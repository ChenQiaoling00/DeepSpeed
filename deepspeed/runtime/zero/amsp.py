# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import List

import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.runtime.zero.mics_utils import (MiCS_CommGroups, create_mics_comm_groups, scale_tensors)
from deepspeed.runtime.zero.amsp_utils import create_amsp_o_comm_groups

from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.partition_parameters import Init, AllGatherCoalescedHandle, ZeroParamStatus
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import instrument_w_nvtx, log_dist
from deepspeed.accelerator import get_accelerator
from torch import Tensor
from torch.nn import Parameter 

from deepspeed.runtime.zero.mics import MiCS_Init, MiCS_Optimizer

class AMSP_O_Optimizer(DeepSpeedZeroOptimizer):
    def __init__(self,
                 init_optimizer,
                 param_names,
                 timers,
                 ds_config,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 use_multi_rank_bucket_allreduce=True,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 mpu=None,
                 clip_grad=0.0,
                 gradient_accumulation_dtype=torch.float32,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False):

        log_dist("Init AMSP OS Sharding Optimizer",ranks=[0])
        super().__init__(init_optimizer,param_names,
                 timers,
                 static_loss_scale,
                 dynamic_loss_scale,
                 dynamic_loss_args,
                 verbose,
                 contiguous_gradients,
                 reduce_bucket_size,
                 use_multi_rank_bucket_allreduce,
                 allgather_bucket_size,
                 dp_process_group,
                 expert_parallel_group,
                 expert_data_parallel_group,
                 reduce_scatter,
                 overlap_comm,
                 offload_optimizer_config,
                 mpu,
                 clip_grad,
                 gradient_accumulation_dtype,
                 communication_data_type,
                 postscale_gradients,
                 gradient_predivide_factor,
                 gradient_accumulation_steps,
                 ignore_unused_parameters,
                 partition_grads,
                 round_robin_gradients,
                 has_moe_layers,
                 fp16_master_weights_and_gradients,
                 elastic_checkpoint)

        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(ds_config, mpu)
        self.amsp_o_comm_group = create_amsp_o_comm_groups(
            _ds_config.amsp_o_shard_size,
            dp_process_group,
            mpu=mpu)
        self.real_dp_process_group = [self.amsp_o_comm_group.os_shard_group for i in range(len(self.optimizer.param_groups))]
