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
"""
The base class for Actor
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from verl import DataProto

__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config):
        """The base class for PPO actor

        Args:
            config (DictConfig): a config passed to the PPOActor. We expect the type to be
                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute logits given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```log_probs```


        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            data (DataProto): an iterator over the DataProto that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.

        """
        pass

    def calibrate_qkv_fp8_scales(
        self,
        data: DataProto,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Calibrate FP8 scales for Q/K/V in attention layers.
        
        This method hooks into attention projection layers to capture Q, K, V activation
        magnitudes, computes percentile-based amax values, and calculates FP8 quantization
        scales. The forward pass is triggered by calling compute_log_prob().
        
        Args:
            data: Calibration data batch (must contain input_ids, attention_mask, position_ids, responses)
            percentile: Percentile for amax computation (default: 99.9)
            margin: Safety margin multiplier (default: 1.05)
            include_q: Whether to include Q scale (default: False, but typically True for KV cache)
        
        Returns:
            Dictionary with format:
            {
                "format": "fp8",
                "percentile": float,
                "margin": float,
                "layers": {
                    "layer_0": {"q_scale": float, "k_scale": float, "v_scale": float},
                    ...
                }
            }
        """
        # Default implementation raises NotImplementedError
        # Subclasses should override this if they support FP8 KV cache calibration
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement calibrate_qkv_fp8_scales(). "
            "This method is required for FP8 KV cache support."
        )
