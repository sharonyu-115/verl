# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

import torch

try:
    from vllm._custom_ops import scaled_fp8_quant
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
except ImportError as e:
    raise ImportError("FP8 quantization not available") from e

logger = logging.getLogger(__name__)

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])
    kv_cache_patches: list = field(default_factory=lambda: [])


fp8_state: FP8State = FP8State()


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config") and isinstance(vllm_config.quant_config, Fp8Config):
        return True

    return False


def _get_params_in_layers(param_names, layers):
    """Get parameter module names in specified layers.
    
    Args:
        param_names: List of all parameter names in the model
        layers: List of layer indices to extract parameters from
        
    Returns:
        List of parameter module names (without .weight suffix) in the specified layers
    """
    layer_templates = []
    for i in layers:
        # Prefixes used by huggingface model transformer layers.
        # We'll use these to match against the parameter names to determine
        # which layer the parameter is in.
        layer_templates.extend(
            [
                f"transformer.h.{i}.",
                f"layers.{i}.",
                f"layer.{i}.",
            ]
        )
    prefixes = [p for p in layer_templates if any(p in n for n in param_names)]
    if len(prefixes) == 0:
        raise ValueError(f"Could not identify layers {layers} for model.")

    params = []
    for name in param_names:
        if (
            any(p in name for p in prefixes)
            and "bias" not in name
            and "layernorm" not in name
            and "norm" not in name
        ):
            # Convert the param name into vllm's module name
            # Vllm wraps the model with an extra 'model'
            params.append(f"model.{name}".removesuffix(".weight"))
    return params


def get_bf16_layer_names(model_config, num_first_layers=0, num_last_layers=0):
    """Get parameter module names that should remain in BF16.
    
    Args:
        model_config: HuggingFace model configuration
        num_first_layers: Number of first layers to keep in BF16
        num_last_layers: Number of last layers to keep in BF16
        
    Returns:
        List of lists: [[first_layers_params], [last_layers_params]]
        This matches NeMo-RL's structure for vLLM's ignored_layers parameter
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel
    
    if num_first_layers == 0 and num_last_layers == 0:
        return []
    
    # Create empty model to get parameter names
    with init_empty_weights():
        model = AutoModel.from_config(model_config)
    param_names = [name for name, _ in model.named_parameters()]
    
    bf16_params = []  # 二维列表，每个元素是一组层的参数
    
    # Get first N layers
    if num_first_layers > 0:
        first_layers = list(range(num_first_layers))
        bf16_params.append(_get_params_in_layers(param_names, first_layers))
    
    # Get last M layers
    if num_last_layers > 0:
        num_hidden_layers = model_config.num_hidden_layers
        last_layers = list(range(num_hidden_layers - num_last_layers, num_hidden_layers))
        bf16_params.append(_get_params_in_layers(param_names, last_layers))
    
    return bf16_params


def get_module_from_param_name(model, name: str):
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


def is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear layers
            if (
                isinstance(module, LinearBase) 
                and module.weight.dtype == torch.float8_e4m3fn
                or (
                    isinstance(module, FusedMoE) 
                    and module.w13_weight.dtype == torch.float8_e4m3fn
                    and module.w2_weight.dtype == torch.float8_e4m3fn
                )
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def scaled_fp8_blockwise(
    data_hp,
    weight_block_size,
):
    # cast tensor from high precision to FP8 with 128*128 blockwise quantization.
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    )
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
    )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)

    # Use FP32 scale
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    # preserve the behavior for 0 amax case
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    descale_fp = torch.reciprocal(scale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(original_shape)

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp


def quant_weights(weights, model, quant_config):
    weights_quantized = []
    for k, v in weights:
        if not is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        if quant_config.weight_block_size is not None:
            logger.info("Using blockwise quantization")
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(torch.bfloat16),
                weight_block_size=quant_config.weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            weights_quantized.append([k, param_lp])
            if "expert" in k:
                weights_quantized.append([k + "_scale_inv", param_scale])
            else:
                weights_quantized.append([k + "_scale", param_scale])
        else:
            logger.info("Using Per tensor quantization")
            original_shape = v.shape
            # Use per tensor quantization
            quantized_tensor, scale = scaled_fp8_quant(v)
            # Reshape back to original shape
            quantized_tensor = quantized_tensor.view(original_shape)

            scale_k = k.replace(".weight", ".weight_scale")
            scale = scale.view(1)
            weights_quantized.extend([(k, quantized_tensor), (scale_k, scale)])

    return weights_quantized


def load_quanted_weights(weights, model_runner):
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config

    weights_quantized = quant_weights(weights, model, quant_config)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    loaded_params = model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type
    # Add a debug print to print the loaded parameters
    return loaded_params


def process_weights_after_loading(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer.

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr
            for attr in custom_param_dir
            if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    weight_scale = (
        layer.weight_scale_inv
        if hasattr(layer, "weight_scale_inv")
        else layer.weight_scale
    )
    weight, weight_scale = process_fp8_weight_block_strategy(layer.weight, weight_scale)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )

    del layer.weight_scale_inv

    maybe_post_process_fp8_weight_block(layer, self.cutlass_block_fp8_supported)

def process_weights_after_loading_moe(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer.
    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    # Lazy import to avoid importing triton too early.
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        is_rocm_aiter_moe_enabled,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        expert_weight_is_col_major,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import (
        get_col_major_tma_aligned_tensor,
        is_deep_gemm_e8m0_used,
    )

    self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_backend is not None:
        layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        layer.w13_weight_scale_inv.data = swap_w13_to_w31(
            layer.w13_weight_scale_inv.data
        )

    # DeepGemm scales need to be transposed and aligned. We try to do
    # it ahead of time for performance reasons.
    if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w13_weight_scale_inv
            )
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w2_weight_scale_inv
            )

    if is_deep_gemm_e8m0_used():
        assert layer.weight_block_size is not None
        # Re-quantise the expert weights so their scales are UE8M0.
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_sz,
        )
        requant_weight_ue8m0_inplace(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_sz,
        )

        # Ensure column-major TMA alignment expected by DeepGEMM.
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w13_weight_scale_inv
            )
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
                layer.w2_weight_scale_inv
            )


def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

    if self.use_marlin:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    if self.block_quant:
        assert self.quant_config.weight_block_size is not None
        return torch.ops.vllm.apply_w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            use_aiter_and_is_supported=self.use_aiter_and_is_supported,
        )

    weight_scale, weight = requantize_with_max_scale(
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        logical_widths=layer.logical_widths,
    )
    return self.fp8_linear.apply(
        input=x,
        weight=weight.t(),
        weight_scale=weight_scale,
        out_dtype=self.out_dtype,
        input_scale=layer.input_scale,
        bias=bias,
    )


def process_weights_after_loading_kv_cache(self, layer) -> None:
    """Monkey patch for vLLM KV cache FP8 to prevent deletion of scale parameters.
    
    This is a patched version of BaseKVCacheMethod.process_weights_after_loading() that
    keeps k_scale, v_scale, q_scale, and prob_scale parameters instead of deleting them.
    This allows for dynamic updates of FP8 scales during RL training.
    
    Args:
        self: The BaseKVCacheMethod instance
        layer: The attention layer containing the scale parameters
    """
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    
    # Convert scales from model parameters to internal attributes
    if layer.kv_cache_dtype == "fp8":
        k_scale = layer.k_scale
        v_scale = layer.v_scale
        # The default k_scale, v_scale is 1, which is invalid for fp8 KV cache
        # In CUDA, we have to recalculate them if the scale is 1
        if isinstance(k_scale, float):
            k_scale_a = torch.full((1,), k_scale, dtype=torch.float32)
            k_scale_b = k_scale
        else:
            k_scale_a = torch.tensor(k_scale.tolist(), dtype=torch.float32)
            k_scale_b = k_scale.tolist()[0]
        if isinstance(v_scale, float):
            v_scale_a = torch.full((1,), v_scale, dtype=torch.float32)
            v_scale_b = v_scale
        else:
            v_scale_a = torch.tensor(v_scale.tolist(), dtype=torch.float32)
            v_scale_b = v_scale.tolist()[0]
        layer.calculate_kv_scales = k_scale_b == 1 or v_scale_b == 1
        layer._k_scale = k_scale_a.to("cuda")
        layer._v_scale = v_scale_a.to("cuda")
    else:
        layer._k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        layer._v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        layer.calculate_kv_scales = False
    
    if layer.calculate_kv_scales and layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    is_singleton_float = lambda x: isinstance(x, float) or (
        isinstance(x, torch.Tensor) and x.numel() == 1 and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError("Only support per-tensor scaling factor for fp8-quantized Q/prob")

    # These are used in the final Attention.forward()
    layer._q_scale.copy_(q_scale)
    layer._prob_scale.copy_(prob_scale)
    
    if layer.kv_cache_dtype == "fp8" and (q_scale == 1.0 or prob_scale == 1.0):
        logger.warning(
            f"Using uncalibrated q_scale {q_scale} and/or prob_scale "
            f"{prob_scale} with fp8 attention. This may cause accuracy "
            "issues. Please make sure q/prob scaling factors are "
            "available in the fp8 checkpoint."
        )

    # CRITICAL CHANGE: We DON'T delete the parameters here to allow for dynamic updates
    # Original vLLM code deletes: del layer.k_scale, layer.v_scale, layer.q_scale, layer.prob_scale
    # We keep them so they can be updated via load_weights() during training
    logger.debug("[KV_SCALES] Patched process_weights_after_loading: kept k_scale/v_scale parameters for dynamic updates")


def apply_kv_cache_fp8_patch():
    """Apply monkey patch to vLLM KV cache to support dynamic scale updates.
    
    This patch is required for FP8 KV cache support in RL training, where
    we need to recalibrate and update Q/K/V scales after each training step.
    
    Returns:
        The patcher object (for tracking and potential cleanup)
    """
    func_path = "vllm.model_executor.layers.quantization.kv_cache.BaseKVCacheMethod.process_weights_after_loading"
    patcher = patch(func_path, process_weights_after_loading_kv_cache)
    patcher.start()
    fp8_state.kv_cache_patches.append(patcher)
    logger.info("[KV_SCALES] Applied vLLM KV cache FP8 monkey patch for dynamic scale updates")
    return patcher


def apply_vllm_fp8_patches(block_quant=True, enable_kv_cache_fp8=False):
    """Apply vLLM FP8 patches for weight quantization and optionally KV cache.
    
    Args:
        block_quant: Whether to use block-wise quantization (default: True)
        enable_kv_cache_fp8: Whether to apply KV cache FP8 patch for dynamic updates (default: False)
    """
    if block_quant:
        func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher1 = patch(func1_path, process_weights_after_loading)
        patcher1.start()
        fp8_state.vllm_patches.append(patcher1)
        
        func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"
        patcher2 = patch(func2_path, process_weights_after_loading_moe)
        patcher2.start()
        fp8_state.vllm_patches.append(patcher2)
    else:
        func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher1 = patch(func1_path, process_weights_after_loading)
        patcher1.start()
        fp8_state.vllm_patches.append(patcher1)
        
        func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.apply"
        patcher2 = patch(func2_path, apply)
        patcher2.start()
        fp8_state.vllm_patches.append(patcher2)
    
    # Apply KV cache FP8 patch if enabled
    if enable_kv_cache_fp8:
        apply_kv_cache_fp8_patch()
        logger.info("[KV_SCALES] KV cache FP8 support enabled with dynamic scale updates")
