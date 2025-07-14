# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import heapq
import importlib
import itertools
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import uuid4

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)


class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return

        # STEP 2: call tools
        tool_calls = completions.choices[0].message.tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        """Call tool and return tool response."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        try:
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
            for j in range(len(roles)):
                if roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 16384,
        abort_callback: callable = None,
        get_load_callback: callable = None,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
            abort_callback: callable, callback function to abort requests immediately.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)
        self.server_addresses = server_addresses

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        # Callback for immediate request abortion
        self.abort_callback = abort_callback

        # Callback for getting load of each server
        self.get_load_callback = get_load_callback

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

        # ==== Length Bias Analysis - Cumulative tracking ====
        self.length_stats_valid_prompts = []  # List of response lengths for valid prompts
        self.length_stats_invalid_prompts = []  # List of response lengths for invalid prompts
        self.num_valid_prompts = 0
        self.num_invalid_prompts = 0
        self.current_step = None  # Track current training step
        # ==== END Length Bias Analysis ====

    def reset_length_bias_stats(self):
        """Reset length bias analysis statistics for a new generation batch."""
        self.length_stats_valid_prompts = []
        self.length_stats_invalid_prompts = []
        self.num_valid_prompts = 0
        self.num_invalid_prompts = 0

    def _calculate_response_length(self, response_content: str) -> int:
        """
        Calculate response length using the same method as response_length/mean.
        This aligns with the calculation in metric_utils.py where response_length 
        is computed as response_mask.sum(-1).float().
        """
        if not response_content.strip():
            return 0
        
        # Tokenize the response content
        tokenized = self.completion_callback.tokenizer(
            response_content, 
            return_tensors="pt", 
            padding=False,
            truncation=False
        )
        
        # Count non-padding tokens (equivalent to attention_mask.sum())
        # For response content without padding, this is just the length
        response_length = tokenized["input_ids"].shape[1]
        
        return response_length

    def _save_length_bias_data(self, prompt_index: int, n: int, response_lengths: list, scores: list, is_valid: bool):
        """Save length bias data to file for batch analysis."""
        import os
        import json
        from datetime import datetime
        
        # Create filename with timestamp if not exists
        filename = "length_bias_analysis.jsonl"
        
        # Prepare data entry with proper type conversion for JSON serialization
        data_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": int(self.current_step) if self.current_step is not None else None,
            "prompt_index": int(prompt_index),
            "n_trajectories": int(n),
            "response_lengths": [int(x) for x in response_lengths],
            "scores": [float(x) for x in scores],
            "is_valid": bool(is_valid),  # Ensure native Python bool type
            "variance": float(np.var(scores)) if scores else 0.0,
            "mean_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
            "std_length": float(np.std(response_lengths)) if len(response_lengths) > 1 else 0.0,
            "max_length": int(np.max(response_lengths)) if response_lengths else 0,
            "min_length": int(np.min(response_lengths)) if response_lengths else 0,
        }
        
        # Append to file
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_entry) + '\n')

    def print_length_bias_summary(self):
        """Print cumulative length bias analysis summary."""
        if not self.length_stats_valid_prompts and not self.length_stats_invalid_prompts:
            return
        
        print(f"[DEBUG LENGTH BIAS] ===== GENERATION SUMMARY =====")
        if self.current_step is not None:
            print(f"Training Step: {self.current_step}")
        
        # Valid prompts statistics
        if self.length_stats_valid_prompts:
            import numpy as np
            valid_mean = np.mean(self.length_stats_valid_prompts)
            valid_max = np.max(self.length_stats_valid_prompts)
            valid_min = np.min(self.length_stats_valid_prompts)
            valid_std = np.std(self.length_stats_valid_prompts) if len(self.length_stats_valid_prompts) > 1 else 0.0
            print(f"Valid prompts ({self.num_valid_prompts} prompts, {len(self.length_stats_valid_prompts)} trajectories):")
            print(f"  Response length - Mean: {valid_mean:.2f}, Max: {valid_max:.0f}, Min: {valid_min:.0f}, Std: {valid_std:.2f}")
        
        # Invalid prompts statistics
        if self.length_stats_invalid_prompts:
            import numpy as np
            invalid_mean = np.mean(self.length_stats_invalid_prompts)
            invalid_max = np.max(self.length_stats_invalid_prompts)
            invalid_min = np.min(self.length_stats_invalid_prompts)
            invalid_std = np.std(self.length_stats_invalid_prompts) if len(self.length_stats_invalid_prompts) > 1 else 0.0
            print(f"Invalid prompts ({self.num_invalid_prompts} prompts, {len(self.length_stats_invalid_prompts)} trajectories):")
            print(f"  Response length - Mean: {invalid_mean:.2f}, Max: {invalid_max:.0f}, Min: {invalid_min:.0f}, Std: {invalid_std:.2f}")
        
        # Comparison analysis
        if self.length_stats_valid_prompts and self.length_stats_invalid_prompts:
            import numpy as np
            valid_mean = np.mean(self.length_stats_valid_prompts)
            invalid_mean = np.mean(self.length_stats_invalid_prompts)
            length_bias = valid_mean - invalid_mean
            print(f"Length bias (valid_mean - invalid_mean): {length_bias:.2f}")
            print(f"Relative length bias: {length_bias / invalid_mean * 100:.2f}%")
        
        # Overall statistics
        all_lengths = self.length_stats_valid_prompts + self.length_stats_invalid_prompts
        total_prompts = self.num_valid_prompts + self.num_invalid_prompts
        if all_lengths:
            import numpy as np
            overall_mean = np.mean(all_lengths)
            overall_max = np.max(all_lengths)
            overall_min = np.min(all_lengths)
            overall_std = np.std(all_lengths) if len(all_lengths) > 1 else 0.0
            print(f"Overall ({total_prompts} prompts, {len(all_lengths)} trajectories):")
            print(f"  Response length - Mean: {overall_mean:.2f}, Max: {overall_max:.0f}, Min: {overall_min:.0f}, Std: {overall_std:.2f}")
        
        if total_prompts > 0:
            filter_ratio = self.num_invalid_prompts / total_prompts * 100
            print(f"Filter ratio: {filter_ratio:.2f}% prompts filtered")
        
        print(f"Detailed data saved to: length_bias_analysis.jsonl")
        print(f"[DEBUG LENGTH BIAS] ===== END SUMMARY =====")

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any], address: str = None):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
            address: Optional specific server address. If None, will randomly select from available servers.
        """
        # If no specific address provided, randomly select one
        if address is None:
            address = random.choice(self.server_addresses)
        
        # Store the selected address in info for later use
        info["__selected_address__"] = address
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info))

        # "fire-and-forget" background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
    ):
        """Submit chat completion request, wait request finish and do callback."""
        # Check if a specific address was pre-selected
        if "__selected_address__" in info:
            address = info.pop("__selected_address__")
        elif request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            # Fallback to load balancing (legacy behavior)
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        # 记录 task 的 request_id 和 server address，用于后续 abort
        task_index = info.get("__task_index__")
        task_request_ids = info.get("__task_request_ids__")
        task_server_addresses = info.get("__task_server_addresses__")
        if task_index is not None and task_request_ids is not None and task_server_addresses is not None:
            task_request_ids[task_index] = request_id
            task_server_addresses[task_index] = address

        completions, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            pass
            # logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                await self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        session = None
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            
            # 设置合理的超时时间，避免无限等待
            timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
            session = aiohttp.ClientSession(timeout=timeout)
            
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                # 检查响应状态
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {error_text}")
                
                # 检查响应类型
                content_type = resp.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    error_text = await resp.text()
                    # 如果是plain text响应，可能是服务器错误或请求被abort
                    raise Exception(f"Expected JSON response but got {content_type}: {error_text}")
                
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            # 只在 finally 块中关闭 session，避免重复关闭
            if session:
                try:
                    # 使用 asyncio.shield 防止 session.close() 被取消
                    await asyncio.shield(session.close())
                except Exception as e:
                    logger.warning(f"Failed to close HTTP session: {e}")
                    # 如果 shield 失败，尝试强制关闭
                    try:
                        if not session.closed:
                            session._connector.close()  # 强制关闭连接器
                    except Exception:
                        pass  # 忽略强制关闭时的异常

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.time()
        
        # Reset length bias statistics at the beginning of each generation
        self.reset_length_bias_stats()
        
        # Get current training step from batch meta_info
        self.current_step = batch.meta_info.get("global_steps", None)
        
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        is_validation = batch.meta_info.get("validate", False)
        n = 1 if is_validation else self.config.n
        batch_conversations = [None] * len(batch) * n
        
        # 早停配置，验证模式下关闭早停机制
        if is_validation:
            min_valid_prompts = len(batch) * n  # 验证模式下等待所有任务完成
            print("[ChatCompletionScheduler] Validation mode: early stopping disabled")
        else:
            # 基于分数方差筛选的合法 prompt 数量
            min_valid_prompts = batch.meta_info.get("needed_valid_prompt_num", len(batch))
            print(f"[ChatCompletionScheduler] Early stop ratio: {min_valid_prompts}/{len(batch)}")
        
        # 跟踪任务的 request_id 和 server address，用于后续 abort
        task_request_ids = {}  # {task_index: request_id}
        task_server_addresses = {}  # {task_index: server_address}
        
        # 初始化批次对话
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            batch_conversations[batch_index] = conversation.tolist()
        
        # 创建任务队列和相关的同步对象
        task_queue = asyncio.Queue()
        task_creation_done = asyncio.Event()
        early_stop_signal = asyncio.Event()  # 早停信号
        
        # 启动动态任务创建器
        task_creator = asyncio.create_task(
            self._dynamic_task_creator(
                batch_conversations=batch_conversations,
                sampling_params=kwargs,
                n=n,
                task_request_ids=task_request_ids,
                task_server_addresses=task_server_addresses,
                task_queue=task_queue,
                task_creation_done=task_creation_done,
                early_stop_signal=early_stop_signal,
            )
        )
        
        # 使用动态任务管理器等待任务完成
        num_prompts = len(batch)
        completed_prompts, valid_prompts, valid_prompt_indices, cancelled_request_ids_by_address = await self._wait_with_dynamic_task_creation(
            task_queue=task_queue,
            task_creation_done=task_creation_done,
            early_stop_signal=early_stop_signal,
            batch_conversations=batch_conversations,
            n=n,
            num_prompts=num_prompts,
            min_valid_prompts=min_valid_prompts,
            task_request_ids=task_request_ids,
            task_server_addresses=task_server_addresses,
            is_validation=is_validation,
        )
        
        # 确保任务创建器完成
        if not task_creator.done():
            task_creator.cancel()
            try:
                await task_creator
            except asyncio.CancelledError:
                pass
        
        if is_validation:
            print(f"[ChatCompletionScheduler] Validation completed: {completed_prompts}/{num_prompts} prompts processed")
        else:
            print(f"[ChatCompletionScheduler] Early stop triggered: {completed_prompts}/{num_prompts} prompts completed, {valid_prompts} valid prompts")
        
        # 在 chat_scheduler 中进行过滤，同时返回合法索引给调用方
        if not is_validation and len(valid_prompt_indices) < num_prompts:
            # 过滤生成的数据
            filtered_batch, filtered_conversations, filtered_n = self._filter_invalid_prompts(
                batch, batch_conversations, valid_prompt_indices, n, num_prompts
            )
            print(f"[ChatCompletionScheduler] Filtered data: kept {len(valid_prompt_indices)}/{num_prompts} valid prompts")
            
            # 使用过滤后的数据进行 postprocess
            output_batch = self.completion_callback.postprocess(filtered_batch, filtered_conversations, n=filtered_n)
            
            # 返回合法的 prompt 索引给调用方，让调用方也过滤 new_batch
            output_batch.meta_info["valid_prompt_indices"] = sorted(list(valid_prompt_indices))
            output_batch.meta_info["need_filtering"] = True
        else:
            # 验证模式或所有 prompt 都合法时，不过滤
            output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
            output_batch.meta_info["valid_prompt_indices"] = list(range(num_prompts))
            output_batch.meta_info["need_filtering"] = False
        
        output_batch.meta_info["timing"] = {
            "generate_sequences": time.time() - t_start,
        }
        output_batch.meta_info["completed_prompts"] = completed_prompts
        output_batch.meta_info["valid_prompts"] = valid_prompts
        output_batch.meta_info["total_prompts"] = num_prompts
        output_batch.meta_info["original_prompts"] = num_prompts
        output_batch.meta_info["cancelled_request_ids_by_address"] = cancelled_request_ids_by_address
        
        # Print length bias analysis summary at the end of generation
        self.print_length_bias_summary()
        
        print("[ChatCompletionScheduler] generate_sequences done")
        return output_batch

    async def _wait_with_dynamic_task_creation(self, task_queue: asyncio.Queue, task_creation_done: asyncio.Event, early_stop_signal: asyncio.Event, batch_conversations: List[List[Dict[str, str]]], n: int, num_prompts: int, min_valid_prompts: int, task_request_ids: Dict[int, str], task_server_addresses: Dict[int, str], is_validation: bool = False) -> tuple[int, int, set, dict]:
        """
        等待任务完成，基于合法 prompt 数量的早停功能，支持动态任务创建
        
        Args:
            task_queue: 任务队列
            task_creation_done: 任务创建完成信号
            early_stop_signal: 早停信号
            batch_conversations: 对话列表，用于为未完成任务提供默认值
            n: 每个输入prompt的生成数量
            num_prompts: 总的 prompt 数量
            min_valid_prompts: 最少需要的合法 prompt 数量
            task_request_ids: 任务索引到request_id的映射
            task_server_addresses: 任务索引到server_address的映射
            is_validation: 是否为验证模式
            
        Returns:
            (实际完成的 prompt 数量, 合法的 prompt 数量, 合法的 prompt 索引集合, 被取消的 request_id 集合)
        """
        # 跟踪每个 prompt 的任务完成情况: {prompt_index: set of completed task_offsets}
        prompt_completion_status = {i: set() for i in range(num_prompts)}
        # 记录合法的 prompt 索引
        valid_prompt_indices = set()
        completed_prompts = 0
        valid_prompts = 0
        pending_tasks = set()
        # 收集被取消的 request_id 按地址分组
        cancelled_request_ids_by_address = {}
        
        try:
            # 动态管理任务：既要从队列获取新任务，也要等待现有任务完成
            while not task_creation_done.is_set() or pending_tasks or not task_queue.empty():
                # 检查是否达到早停条件（仅训练模式）
                if not is_validation and valid_prompts >= min_valid_prompts:
                    print(f"[EarlyStop] 达到早停条件：{valid_prompts}/{min_valid_prompts} 合法 prompts")
                    early_stop_signal.set()  # 设置早停信号，通知任务创建器停止创建新任务
                    break
                
                # 从队列获取新任务（非阻塞）
                new_tasks_added = 0
                while not task_queue.empty():
                    try:
                        task = task_queue.get_nowait()
                        pending_tasks.add(task)
                        new_tasks_added += 1
                    except asyncio.QueueEmpty:
                        break
                
                if new_tasks_added > 0:
                    print(f"[TaskManager] 添加了 {new_tasks_added} 个新任务，当前待处理任务数：{len(pending_tasks)}")
                
                # 如果没有待处理任务，等待新任务或任务创建完成
                if not pending_tasks:
                    if not task_creation_done.is_set():
                        await asyncio.sleep(0.1)  # 短暂等待新任务
                        continue
                    else:
                        break  # 任务创建完成且无待处理任务
                
                # 等待至少一个任务完成
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                # 检查完成的任务并更新 prompt 完成状态
                for task in done_tasks:
                    try:
                        await task  # 获取任务结果，如果有异常会被抛出
                        
                        # 更新对应 prompt 的完成状态
                        prompt_index = getattr(task, '_prompt_index', None)
                        task_offset = getattr(task, '_task_offset', None)
                        
                        if prompt_index is not None and task_offset is not None:
                            prompt_completion_status[prompt_index].add(task_offset)
                            
                            # 检查这个 prompt 是否完全完成（所有 n 个任务都完成）
                            if len(prompt_completion_status[prompt_index]) == n:
                                completed_prompts += 1
                                
                                if is_validation:
                                    # 验证模式下所有完成的 prompt 都算作合法
                                    valid_prompts += 1
                                    valid_prompt_indices.add(prompt_index)
                                    print(f"[Validation] Prompt {prompt_index} 完成 (已完成 {completed_prompts}/{num_prompts} 个)")
                                else:
                                    # 训练模式下检查该 prompt 是否合法（分数方差不为0）
                                    is_valid = self._check_prompt_score_variance(batch_conversations, prompt_index, n)
                                    if is_valid:
                                        valid_prompts += 1
                                        valid_prompt_indices.add(prompt_index)
                                        print(f"[EarlyStop] Prompt {prompt_index} 完成且合法 (已完成 {completed_prompts}/{num_prompts} 个，合法 {valid_prompts} 个)")
                                    else:
                                        print(f"[EarlyStop] Prompt {prompt_index} 完成但不合法 (方差为0) (已完成 {completed_prompts}/{num_prompts} 个，合法 {valid_prompts} 个)")
                                
                    except Exception as e:
                        logger.exception(f"Task failed with exception: {e}")
                        # 为失败的任务提供默认对话
                        task_index = getattr(task, '_task_index', None)
                        if task_index is not None:
                            self._provide_default_conversation(batch_conversations, task_index)
            
            # 如果达到早停条件，取消剩余任务（验证模式下不会早停）
            if not is_validation and valid_prompts >= min_valid_prompts and (pending_tasks or not task_queue.empty()):
                print(f"[EarlyStop] 达到早停条件，取消剩余任务")
                
                # 设置早停信号，确保任务创建器停止
                early_stop_signal.set()
                
                # 取消队列中尚未开始的任务
                cancelled_from_queue = 0
                while not task_queue.empty():
                    try:
                        task = task_queue.get_nowait()
                        task.cancel()
                        task_index = getattr(task, '_task_index', None)
                        if task_index is not None:
                            self._provide_default_conversation(batch_conversations, task_index)
                        cancelled_from_queue += 1
                    except asyncio.QueueEmpty:
                        break
                
                if cancelled_from_queue > 0:
                    print(f"[EarlyStop] 从队列中取消了 {cancelled_from_queue} 个尚未开始的任务")
                
                # 收集被取消任务的 request_id 和 server address
                for task in pending_tasks:
                    task_index = getattr(task, '_task_index', None)
                    if task_index is not None:
                        # 收集 request_id 和 server address
                        if task_index in task_request_ids and task_index in task_server_addresses:
                            request_id = task_request_ids[task_index]
                            server_address = task_server_addresses[task_index]
                            if server_address not in cancelled_request_ids_by_address:
                                cancelled_request_ids_by_address[server_address] = []
                            cancelled_request_ids_by_address[server_address].append(request_id)
                
                # 取消所有待处理的任务
                for task in pending_tasks:
                    task.cancel()
                    task_index = getattr(task, '_task_index', None)
                    if task_index is not None:
                        self._provide_default_conversation(batch_conversations, task_index)
                
                # 等待取消的任务清理完成，使用更短的超时避免无限等待
                print(f"[EarlyStop] 等待 {len(pending_tasks)} 个被取消任务的清理...")
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_tasks, return_exceptions=True),
                        timeout=30.0  # 30秒超时
                    )
                    print("[EarlyStop] 所有被取消任务已完成清理")
                except asyncio.TimeoutError:
                    print("[EarlyStop] 警告：部分任务清理超时，可能存在未完成的服务器端操作")
                except Exception as e:
                    logger.exception(f"[EarlyStop] 任务清理过程中发生错误: {e}")
                
                # 立即abort服务器端请求（在取消客户端任务之前）
                if cancelled_request_ids_by_address and self.abort_callback and callable(self.abort_callback):
                    print(f"[EarlyStop] 立即abort {sum(len(ids) for ids in cancelled_request_ids_by_address.values())} 个服务器端请求")
                    try:
                        # Run abort in a separate thread to avoid blocking the event loop
                        def abort_in_thread():
                            self.abort_callback(cancelled_request_ids_by_address)
                        
                        import threading
                        abort_thread = threading.Thread(target=abort_in_thread, daemon=True)
                        abort_thread.start()
                        
                        # Give abort operations a chance to start
                        await asyncio.sleep(0.2)
                    except Exception as e:
                        logger.exception(f"Error calling abort_callback: {e}")

                # 额外等待时间，确保服务器端操作完成
                print("[EarlyStop] 等待服务器端操作完成...")
                await asyncio.sleep(2.0)  # 给服务器端额外的清理时间
                
                # 清空待处理任务集合
                pending_tasks = set()
            else:
                # 等待所有剩余任务完成
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    # 重新计算完成的 prompt 数量
                    completed_prompts = sum(1 for status in prompt_completion_status.values() if len(status) == n)
                    
                    # 重新计算合法 prompt 数量
                    valid_prompts = 0
                    valid_prompt_indices.clear()
                    for prompt_index in range(num_prompts):
                        if len(prompt_completion_status[prompt_index]) == n:
                            if is_validation:
                                # 验证模式下所有完成的 prompt 都算作合法
                                valid_prompts += 1
                                valid_prompt_indices.add(prompt_index)
                            else:
                                # 训练模式下检查分数方差
                                if self._check_prompt_score_variance(batch_conversations, prompt_index, n):
                                    valid_prompts += 1
                                    valid_prompt_indices.add(prompt_index)
                    
        except Exception as e:
            logger.exception(f"Error in _wait_with_dynamic_task_creation: {e}")
            # 设置早停信号，确保任务创建器停止
            early_stop_signal.set()
            # 确保所有任务都被取消
            for task in pending_tasks:
                task.cancel()
                task_index = getattr(task, '_task_index', None)
                if task_index is not None:
                    self._provide_default_conversation(batch_conversations, task_index)
            # 也要取消队列中的任务
            while not task_queue.empty():
                try:
                    task = task_queue.get_nowait()
                    task.cancel()
                except asyncio.QueueEmpty:
                    break
            raise
            
        return completed_prompts, valid_prompts, valid_prompt_indices, cancelled_request_ids_by_address

    async def _dynamic_task_creator(self, batch_conversations: List[List[Dict[str, str]]], sampling_params: Dict[str, Any], n: int, task_request_ids: Dict[int, str], task_server_addresses: Dict[int, str], task_queue: asyncio.Queue, task_creation_done: asyncio.Event, early_stop_signal: asyncio.Event):
        """
        动态任务创建器：根据服务器负载情况分批次创建任务
        
        Args:
            batch_conversations: 批次对话列表
            sampling_params: 采样参数
            n: 每个 prompt 的回复数量
            task_request_ids: 任务索引到request_id的映射
            task_server_addresses: 任务索引到server_address的映射
            task_queue: 任务队列
            task_creation_done: 任务创建完成信号
            early_stop_signal: 早停信号
        """
        try:
            total_tasks = len(batch_conversations)
            created_tasks = 0
            
            # 默认配置
            batch_creation_interval = getattr(self.config, 'batch_creation_interval', 10)  # 批次创建间隔（秒）
            max_concurrent_per_server = getattr(self.config, 'max_concurrent_per_server', 512)  # 每个服务器最大并发数
            
            print(f"[TaskCreator] 开始动态创建 {total_tasks} 个任务")
            print(f"[TaskCreator] 配置：batch_creation_interval={batch_creation_interval}, max_concurrent_per_server={max_concurrent_per_server}")
            
            while created_tasks < total_tasks:
                # 检查早停信号
                if early_stop_signal.is_set():
                    print(f"[TaskCreator] 收到早停信号，停止创建任务。已创建 {created_tasks}/{total_tasks} 个任务")
                    break
                
                # 第一步：调用 get_load_callback 获取所有服务的负载情况
                server_loads = {}
                if self.get_load_callback and callable(self.get_load_callback):
                    try:
                        server_loads = await self.get_load_callback()
                        print(f"[TaskCreator] 获取服务器负载情况：{server_loads}")
                    except Exception as e:
                        print(f"[TaskCreator] Error calling get_load_callback: {e}")
                        # 如果获取负载失败，使用默认策略
                        server_loads = {}
                
                # 第二步：计算每个服务器可以接收的新任务数量
                server_capacities = {}
                for address in self.server_addresses:
                    current_load = server_loads.get(address, 0)
                    available_capacity = max(0, max_concurrent_per_server - current_load)
                    server_capacities[address] = available_capacity
                
                total_capacity = sum(server_capacities.values())
                print(f"[TaskCreator] 服务器容量情况：{server_capacities}，总容量：{total_capacity}")
                
                # 第三步：根据容量分配任务
                if total_capacity > 0:
                    # 按比例分配任务给各个服务器
                    tasks_to_create = min(total_capacity, total_tasks - created_tasks)
                    
                    for address, capacity in server_capacities.items():
                        if capacity > 0 and created_tasks < total_tasks:
                            # 为这个服务器创建任务
                            tasks_for_this_server = min(
                                capacity,
                                int(tasks_to_create * capacity / total_capacity) + 1,
                                total_tasks - created_tasks
                            )
                            
                            for _ in range(tasks_for_this_server):
                                if created_tasks >= total_tasks:
                                    break
                                    
                                batch_index = created_tasks
                                conversation = batch_conversations[batch_index]
                                prompt_index = batch_index // n
                                task_offset = batch_index % n
                                
                                # 创建任务，但指定使用特定的服务器地址
                                task = asyncio.create_task(
                                    self._submit_chat_completions_semaphore(
                                        messages=conversation,
                                        request_id=None,
                                        sampling_params=sampling_params,
                                        task_index=batch_index,
                                        task_request_ids=task_request_ids,
                                        task_server_addresses=task_server_addresses,
                                        address=address,
                                    )
                                )
                                
                                # 为任务添加索引信息
                                task._task_index = batch_index
                                task._prompt_index = prompt_index
                                task._task_offset = task_offset
                                
                                await task_queue.put(task)
                                created_tasks += 1
                                
                                # print(f"[TaskCreator] 创建任务 {created_tasks}/{total_tasks} 到服务器 {address}")
                    
                    print(f"[TaskCreator] 本轮创建了 {tasks_to_create} 个任务，总进度：{created_tasks}/{total_tasks}")
                else:
                    print("[TaskCreator] 所有服务器都已满载，等待...")
                
                # 第四步：如果还有任务需要创建，等待一段时间
                if created_tasks < total_tasks:
                    await asyncio.sleep(batch_creation_interval)
            
            print(f"[TaskCreator] 完成所有 {total_tasks} 个任务的创建")
            task_creation_done.set()
            
        except Exception as e:
            logger.exception(f"Error in _dynamic_task_creator: {e}")
            task_creation_done.set()  # 即使出错也要设置完成信号
            raise

    def _filter_invalid_prompts(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], 
                               valid_prompt_indices: set, n: int, num_prompts: int) -> tuple[DataProto, List[List[Dict[str, str]]], int]:
        """
        过滤掉非法的 prompt，只保留合法的数据
        
        Args:
            batch: 原始批次数据
            batch_conversations: 所有对话列表
            valid_prompt_indices: 合法的 prompt 索引集合
            n: 每个 prompt 的回复数量
            num_prompts: 总 prompt 数量
            
        Returns:
            (过滤后的批次数据, 过滤后的对话列表, 新的 n 值)
        """
        try:
            # 创建有序的合法 prompt 索引列表
            sorted_valid_prompt_indices = sorted(list(valid_prompt_indices))
            
            # 1. 构建任务级别的索引（用于过滤 batch_conversations 和任务级数据）
            task_indices = []
            for prompt_idx in sorted_valid_prompt_indices:
                for task_offset in range(n):
                    task_index = prompt_idx * n + task_offset
                    if task_index < len(batch_conversations):
                        task_indices.append(task_index)
            
            # 2. 过滤 batch_conversations
            filtered_conversations = [batch_conversations[i] for i in task_indices]
            
            # 3. 处理 batch 数据的过滤
            filtered_batch = batch.select_idxs(sorted_valid_prompt_indices)

            return filtered_batch, filtered_conversations, n
            
        except Exception as e:
            logger.exception(f"Error filtering invalid prompts: {e}")
            # 发生错误时返回原始数据
            return batch, batch_conversations, n
    
    def _check_prompt_score_variance(self, batch_conversations: List[List[Dict[str, str]]], prompt_index: int, n: int) -> bool:
        """
        检查一个 prompt 的 n 个回复的分数方差是否不为 0
        
        Args:
            batch_conversations: 对话列表
            prompt_index: prompt 索引
            n: 每个 prompt 的回复数量
            
        Returns:
            bool: 方差不为 0 则返回 True，否则返回 False
        """
        try:
            scores = []
            response_lengths = []  # 用于长度分析
            
            # 收集该 prompt 对应的 n 个回复的分数和长度
            for task_offset in range(n):
                task_index = prompt_index * n + task_offset
                if task_index < len(batch_conversations) and batch_conversations[task_index] is not None:
                    conversation = batch_conversations[task_index]
                    if len(conversation) > 0:
                        # 找到最后一个 assistant 回复
                        found_valid_assistant = False
                        for message in reversed(conversation):
                            if message.get("role") == "assistant" and "score" in message:
                                scores.append(float(message["score"]))
                                
                                # 计算response长度，与response_length/mean对齐
                                assistant_content = message.get("content", "")
                                response_length = self._calculate_response_length(assistant_content)
                                response_lengths.append(response_length)
                                
                                found_valid_assistant = True
                                break
                        
                        # 如果没找到带分数的 assistant 回复，说明这是一个非法 conversation
                        if not found_valid_assistant:
                            logger.debug(f"Prompt {prompt_index}, task {task_offset}: 没有找到有效的 assistant 回复")
                            return False  # 存在非法 conversation，整个 prompt 无效
                    else:
                        # conversation 为空，非法
                        logger.debug(f"Prompt {prompt_index}, task {task_offset}: conversation 为空")
                        return False
                else:
                    # conversation 不存在或为 None，非法
                    logger.debug(f"Prompt {prompt_index}, task {task_offset}: conversation 不存在")
                    return False
            
            # 检查是否收集到了 n 个分数
            if len(scores) != n:
                logger.warning(f"Prompt {prompt_index} expected {n} scores but got {len(scores)}")
                return False
            
            # 计算方差
            import numpy as np
            variance = np.var(scores)
            
            # 检查方差是否不为 0（考虑浮点数精度）
            is_valid = variance > 1e-8
            
            # ==== DEBUG: Length Bias Analysis ====
            # 累积统计信息并保存到文件
            if response_lengths:
                # 保存详细数据到文件
                self._save_length_bias_data(prompt_index, n, response_lengths, scores, is_valid)
                
                if is_valid:
                    self.length_stats_valid_prompts.extend(response_lengths)
                    self.num_valid_prompts += 1
                    mean_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0
                    print(f"[DEBUG LENGTH BIAS] VALID prompt {prompt_index} ({n} trajectories) - mean_len: {mean_len:.1f}")
                else:
                    self.length_stats_invalid_prompts.extend(response_lengths)
                    self.num_invalid_prompts += 1
                    mean_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0
                    print(f"[DEBUG LENGTH BIAS] INVALID prompt {prompt_index} ({n} trajectories) - mean_len: {mean_len:.1f}")
            # ==== END DEBUG ====
            
            if not is_valid:
                logger.debug(f"Prompt {prompt_index} scores: {scores}, variance: {variance}")
            
            return is_valid
            
        except Exception as e:
            logger.exception(f"Error checking score variance for prompt {prompt_index}: {e}")
            return False
    
    def _provide_default_conversation(self, batch_conversations: List[List[Dict[str, str]]], task_index: int):
        """为未完成或失败的任务提供默认对话"""
        if batch_conversations[task_index] is not None and len(batch_conversations[task_index]) > 0:
            # 如果对话已经有内容但未完成，添加默认助手回复
            if batch_conversations[task_index][-1].get("role") != "assistant":
                batch_conversations[task_index].append({
                    "role": "assistant", 
                    "content": "",  # 空回复
                    "score": 0.0,   # 默认分数
                    "acc": 0.0,     # 默认准确度
                    "pred": ""      # 默认预测
                })
            else:
                # 如果最后一条是助手消息但缺少评分信息，添加默认评分
                last_message = batch_conversations[task_index][-1]
                if "score" not in last_message:
                    last_message["score"] = 0.0
                if "acc" not in last_message:
                    last_message["acc"] = 0.0
                if "pred" not in last_message:
                    last_message["pred"] = ""
        else:
            # 如果对话为空，这通常不应该发生，但为了稳健性还是处理一下
            logger.warning(f"Task {task_index} conversation is None or empty, this should not happen in normal cases")
            # 由于我们无法知道原始的 raw_prompt，只能创建一个最小的默认结构
            batch_conversations[task_index] = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "", "score": 0.0, "acc": 0.0, "pred": ""}
            ]

    async def _submit_chat_completions_semaphore(self, messages: List[Dict[str, str]], request_id: str, sampling_params: Dict[str, Any], task_index: int, task_request_ids: Dict[int, str], task_server_addresses: Dict[int, str], address: str=None):
        """Submit chat completion request to a specific server and wait for completion"""
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
            "__task_index__": task_index,
            "__task_request_ids__": task_request_ids,
            "__task_server_addresses__": task_server_addresses,
        }

        # Use the unified submit_chat_completions method with specific address
        self.submit_chat_completions(messages=messages, request_id=request_id, info=info, address=address)

        # Wait until all completion requests are done
        await done.wait()
