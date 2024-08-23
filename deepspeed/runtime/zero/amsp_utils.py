
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger
from dataclasses import dataclass


@dataclass
class AMSP_OS_CommGroups:
    os_shard_group=None
    os_shard_size=-1
    os_shard_rank=-1


    os_repli_group=None
    os_repli_size=-1
    os_repli_rank=-1


def create_amsp_o_comm_groups(
    shard_size,
    dp_group,
    mpu=None
):
    """create optimizer shard group and repli group from config file

    Returns:
        AMSP_OS_CommGroups
    """

    groups=AMSP_OS_CommGroups

    if mpu is not None:
        assert dp_group==mpu.get_data_parallel_group()

    world_size=dist.get_world_size()

    global_rank=dist.get_rank()

    config=_generate_amsp_o_config(world_size,shard_size,1)

    ranks_of_shard_o_group = config['shard_groups']
    ranks_of_repli_o_group=config['replicate_groups']

    if ranks_of_repli_o_group == 0: 
        assert len(ranks_of_shard_o_group) == 1,f'replicate group are empty only when single shard group'
        for r in range(ranks_of_shard_o_group):
            ranks_of_repli_o_group.append([r])

    global_rank=dist.get_rank()
    for shard_ranks in ranks_of_shard_o_group:
        _group=dist.new_group(shard_ranks)
        if global_rank in shard_ranks:
            groups.os_shard_group=_group
            groups.os_shard_size=len(shard_ranks)
            groups.os_shard_rank=dist.get_rank(_group)
    

    for repli_ranks in ranks_of_repli_o_group:
        if len(repli_ranks) > 1:
            _group=dist.new_group(repli_ranks)
            if global_rank in repli_ranks:
                groups.os_repli_group=_group
                groups.os_repli_rank=dist.get_rank(_group)
                groups.os_repli_size=len(repli_ranks)
        else:
                groups.os_repli_group=None
                groups.os_repli_rank=0
                groups.os_repli_size=1

    return groups


    

def _generate_amsp_o_config(world_size,shard_size,pp_size=1):
    assert (world_size // pp_size) % shard_size == 0 , \
    f'dp group is not dividable by dp_shard_size, ' \
    f' (world size {world_size}, pp size {pp_size}, shard size {shard_size}'

    config={}

    shard_groups=np.arange(world_size).reshape(-1,shard_size)

    replicate_groups=[]

    for i in range(shard_size):
        same_shard_ranks=shard_groups[:,i].tolist()
        n_ranks=len(same_shard_ranks)
        replicate_size=n_ranks // pp_size
        for j in range(0,n_ranks,replicate_size):
            replicate_groups.extend([same_shard_ranks[j:j + replicate_size]])

    config['replicate_groups']=replicate_groups
    config['shard_groups']=shard_groups
    return config
