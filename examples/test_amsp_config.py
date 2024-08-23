# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Testing on a 8 GPUs node
NDEV_PER_NODE=2 torchrun --nnodes 1 --nproc-per-node 8 test_mics_config.py
"""

import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import deepspeed.comm as dist
from GPT_model import SimpleModel,ModelConfig

from torch.utils.data import DataLoader, TensorDataset


class DummyProfile:
    """
    Dummy Profile.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        pass

    def step(self):
        pass



model_args={
    "block_size": 4096,
    "vocab_size": 50304,
    "n_layer": 32,
    "n_head": 32,
    "n_emb": 768,
    "dropout": 0.0,
    "bias": True
}


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(vocab_size):
    batch_size = 32
    sequence_length = 1024
    input_ids = torch.randint(0,vocab_size, (10000, sequence_length))
    targets = input_ids.clone()
    dataset = TensorDataset(input_ids, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('--zero', type=int, default=3)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--os_shard_size', type=int, default=4)

    parser.add_argument('--p_shard_size', default=4, type=int)
    parser.add_argument('--mics_hierarchical_params_gather', default=False, action='store_true')
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    config_dict["zero_optimization"]["mics_shard_size"] = args.p_shard_size
    config_dict["zero_optimization"]["amsp_o_shard_size"] = args.os_shard_size
    config_dict["zero_optimization"]["mics_hierarchical_params_gather"] = args.mics_hierarchical_params_gather
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path

    return args


def print0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)



# 2. set deepspeed config
config_dict = {
    "train_batch_size": 32,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
        }
    },
    "tensorboard": {
    "enabled": True,
    "output_path": "output/ds_logs/",
    "job_name": "train_AMSP"
    },
    "fp16": {
        "enabled": False,
        "initial_scale_power": 15
    },
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 20,
        "overlap_comm": True,
        "reduce_bucket_size": 5e6,
        "mics_shard_size": 0,
        "amsp_p_shard_size":0,
        "amsp_g_shard_size":0,
        "amsp_o_shard_size":8,
        "mics_hierarchical_params_gather": False,
        "stage3_model_persistence_threshold": 10,
        "contiguous_gradients": False,
    },
}


deepspeed.init_distributed()
args = get_args('/tmp/', config_dict)
hidden_dim = 32


if args.zero ==3 and args.mics_shard_size>0:
    with deepspeed.zero.MiCS_Init(config_dict_or_path=config_dict):
        model = SimpleModel(model_args)
else:
    model = SimpleModel(model_args)

print(f'Total number of parameters: {model.get_num_params()}')

model, _, _, _ = deepspeed.initialize(args=args,
                                      model=model,
                                      model_parameters=model.parameters(),
                                      dist_init_required=True)

def print_params(tag, model):
    if dist.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))

data_loader = get_data_loader(model_args['vocab_size'])

llm_profile = torch.profiler.profile
llm_profile=DummyProfile

with llm_profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,

        ],
        schedule=torch.profiler.schedule(
            skip_first=1, wait=1, warmup=1, active=1, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"tmp"
        ),
        with_stack=True,
        with_modules=True,
        profile_memory=True,
    ) as prof:
    for n, batch in enumerate(data_loader):
        inputs, labels = batch[0].to(model.device), batch[1].to(model.device)
        loss = model(inputs, labels)
        if dist.get_rank() == 0:
            print("LOSS:", loss.item())
        model.backward(loss)
        model.step()
        prof.step()
        #print_params('step={}'.format(n), model)
        if n == 10000: break
