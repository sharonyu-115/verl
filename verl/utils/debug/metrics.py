# Copyright 2025 Individual Contributor: TomQunChaoA
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
import os
from typing import Optional

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def dump_high_diff_tokens(
    data: DataProto,
    tokenizer,
    threshold_percentile: float = 95.0,
    dump_dir: Optional[str] = None,
    step: Optional[int] = None,
) -> dict:
    """
    Dump tokens with high log probability differences between rollout and actor.
    
    This function identifies tokens where rollout_log_probs and old_log_probs differ significantly,
    and saves detailed information about these tokens for analysis.
    
    Args:
        data: DataProto
            the data batch to analyze
            rollout_log_probs: log_probs from rollout forward
            old_log_probs: log_probs from actor forward
            responses: the response token IDs
            response_mask: mask for valid tokens
        tokenizer: tokenizer for decoding token IDs
        threshold_percentile: percentile threshold for identifying high diff tokens (default: 95.0)
        dump_dir: directory to save the dump files (default: ./logprob_diff_dumps)
        step: training step number for filename
        
    Returns:
        dict: statistics about dumped tokens
    """
    if dump_dir is None:
        dump_dir = "./logprob_diff_dumps"
    os.makedirs(dump_dir, exist_ok=True)
    
    rollout_log_probs = data.batch["rollout_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    responses = data.batch["responses"]
    
    if "response_mask" in data.batch:
        response_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        response_mask = data.batch["attention_mask"][:, -responses.size(1):]
    else:
        response_mask = torch.ones_like(responses, dtype=torch.float32)
    
    # Calculate absolute log probability differences
    log_prob_diff = torch.abs(old_log_probs - rollout_log_probs)
    
    # Apply mask
    masked_diff = torch.where(response_mask.bool(), log_prob_diff, torch.tensor(0.0, device=log_prob_diff.device))
    
    # Calculate threshold
    valid_diffs = masked_diff[response_mask.bool()]
    if valid_diffs.numel() == 0:
        logger.warning("No valid tokens to analyze")
        return {"num_high_diff_tokens": 0}
    
    threshold = torch.quantile(valid_diffs, threshold_percentile / 100.0).item()
    
    # Collect high-diff tokens
    high_diff_mask = (masked_diff > threshold) & response_mask.bool()
    
    # Get prompts if available
    prompts = data.batch.get("prompts", None)
    
    # Track unique batches with high diff tokens
    batches_with_high_diff = set()
    for batch_idx in range(responses.size(0)):
        if high_diff_mask[batch_idx].any():
            batches_with_high_diff.add(batch_idx)
    
    # Collect prompt and response information for each unique batch with high diff
    batch_prompt_info = {}
    for batch_idx in batches_with_high_diff:
        prompt_str = None
        prompt_ids_list = None
        
        if prompts is not None:
            if isinstance(prompts, torch.Tensor):
                # prompts is a tensor of token IDs
                prompt_ids_raw = prompts[batch_idx].tolist()
                # Filter out padding tokens (left padding token is 151643, the eos/endoftext token)
                # Left padding means padding tokens are at the beginning
                prompt_ids_list = [pid for pid in prompt_ids_raw if pid != 151643]
                prompt_str = tokenizer.decode(prompt_ids_list, skip_special_tokens=True)
            elif isinstance(prompts, (list, tuple)):
                # prompts is already a list of strings
                prompt_str = prompts[batch_idx]
        
        # Collect response information (without padding tokens)
        response_ids = responses[batch_idx].tolist()
        response_mask_batch = response_mask[batch_idx].tolist()
        # Filter response_ids to only include tokens where mask is True (1)
        valid_response_ids = [token_id for token_id, mask_val in zip(response_ids, response_mask_batch) if mask_val == 1]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        
        batch_prompt_info[batch_idx] = {
            "prompt_ids": prompt_ids_list,
            "prompt_str": prompt_str,
            "response_ids": valid_response_ids,
            "response_str": response_str
        }
    
    dump_data = []
    for batch_idx in range(responses.size(0)):
        for token_idx in range(responses.size(1)):
            if high_diff_mask[batch_idx, token_idx]:
                token_id = responses[batch_idx, token_idx].item()
                rollout_lp = rollout_log_probs[batch_idx, token_idx].item()
                old_lp = old_log_probs[batch_idx, token_idx].item()
                diff = log_prob_diff[batch_idx, token_idx].item()
                
                # Decode token
                token_str = tokenizer.decode([token_id])
                
                dump_data.append({
                    "batch_idx": batch_idx,
                    "token_idx": token_idx,
                    "token_id": token_id,
                    "token_str": repr(token_str),  # Use repr to handle special characters
                    "rollout_log_prob": rollout_lp,
                    "old_log_prob": old_lp,
                    "abs_diff": diff,
                    "rollout_prob": torch.exp(torch.tensor(rollout_lp)).item(),
                    "old_prob": torch.exp(torch.tensor(old_lp)).item(),
                })
    
    # Save to file
    if dump_data:
        step_str = f"step{step}" if step is not None else "unknown_step"
        filename = os.path.join(dump_dir, f"high_diff_tokens_{step_str}.txt")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"High Log-Probability Diff Tokens (threshold={threshold:.6f}, percentile={threshold_percentile})\n")
            f.write("=" * 100 + "\n\n")
            
            # Track which batches we've written prompt info for
            written_batch_prompts = set()
            
            for item in dump_data:
                batch_idx = item['batch_idx']
                
                # Write prompt and response info once per batch
                if batch_idx not in written_batch_prompts and batch_idx in batch_prompt_info:
                    prompt_info = batch_prompt_info[batch_idx]
                    
                    f.write("\n" + "=" * 100 + "\n")
                    f.write(f"BATCH {batch_idx} - PROMPT AND RESPONSE INFORMATION\n")
                    f.write("=" * 100 + "\n")
                    
                    if prompt_info['prompt_ids'] is not None:
                        f.write("\n*** PROMPT IDs (without padding tokens) ***\n")
                        f.write(f"{prompt_info['prompt_ids']}\n")
                    
                    if prompt_info['prompt_str'] is not None:
                        f.write("\n*** PROMPT (decoded, without padding tokens) ***\n")
                        f.write(f"{repr(prompt_info['prompt_str'])}\n")
                    
                    if prompt_info['response_ids'] is not None:
                        f.write("\n*** RESPONSE IDs (without padding tokens) ***\n")
                        f.write(f"{prompt_info['response_ids']}\n")
                    
                    if prompt_info['response_str'] is not None:
                        f.write("\n*** RESPONSE (decoded, without padding tokens) ***\n")
                        f.write(f"{repr(prompt_info['response_str'])}\n")
                    
                    f.write("\n" + "=" * 100 + "\n")
                    f.write(f"BATCH {batch_idx} - HIGH DIFF TOKENS\n")
                    f.write("=" * 100 + "\n\n")
                    
                    written_batch_prompts.add(batch_idx)
                
                # Write token information
                f.write(f"Batch: {item['batch_idx']}, Token Position: {item['token_idx']}\n")
                f.write(f"  Token ID: {item['token_id']}\n")
                f.write(f"  Token: {item['token_str']}\n")
                f.write(f"  Rollout Log Prob: {item['rollout_log_prob']:.6f} (prob: {item['rollout_prob']:.6f})\n")
                f.write(f"  Old Log Prob: {item['old_log_prob']:.6f} (prob: {item['old_prob']:.6f})\n")
                f.write(f"  Absolute Diff: {item['abs_diff']:.6f}\n")
                f.write("-" * 100 + "\n")
        
        logger.info(f"Dumped {len(dump_data)} high-diff tokens to {filename}")
    
    return {
        "num_high_diff_tokens": len(dump_data),
        "threshold": threshold,
        "dump_file": filename if dump_data else None,
    }


def calculate_debug_metrics(data: DataProto, tokenizer=None, dump_high_diff: bool = False, 
                           dump_threshold_percentile: float = 95.0, dump_dir: Optional[str] = None,
                           step: Optional[int] = None) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
        tokenizer: tokenizer for decoding tokens (optional, required if dump_high_diff=True)
        dump_high_diff: whether to dump high-diff tokens to file
        dump_threshold_percentile: percentile threshold for high-diff tokens (default: 95.0)
        dump_dir: directory to save dump files
        step: training step number
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
            "training/num_high_diff_tokens": number of high-diff tokens dumped (if dump_high_diff=True)
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
    
    # Dump high-diff tokens if requested
    if dump_high_diff:
        if tokenizer is None:
            logger.warning("tokenizer is required for dumping high-diff tokens, skipping dump")
        else:
            dump_stats = dump_high_diff_tokens(
                data, tokenizer, 
                threshold_percentile=dump_threshold_percentile,
                dump_dir=dump_dir,
                step=step
            )
            metrics["training/num_high_diff_tokens"] = dump_stats["num_high_diff_tokens"]
    
    return metrics
