#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

splits=$1

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

GPUS_PER_NODE=8
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
NNODES=$SLURM_NNODES

# set p,g,os sharding dimension,here we only implement g_shard_size=p_shard_size
# 1. when p_shard_size = 0 , os_shard_size = world_size, will be zero stage-1 configuration
# 2. when p_shard_size = os_shard_size = g_shard_size = world_size, will be zero stage-2 configuration
# 3. when p_shard_size = os_shard_Size = g_shard_size < world_size, will be MiCS configuration
# 4. when p_shard_size < os_shard_size, will be AMSP. https://arxiv.org/abs/2311.00257

p_shard_size=0
g_shard_size=$p_shard_size
os_shard_size=32


torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
test_amsp_config.py --zero=1 --p_shard_size=${p_shard_size} --os_shard_size=${os_shard_size}

# srun -p llm_s --gpus-per-task=8 --nodes=2 --tasks-per-node=1 test_mics.sh > log.log 2>&1
