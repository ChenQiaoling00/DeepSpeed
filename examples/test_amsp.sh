#!/bin/bash

# export NCCL_SOCKET_IFNAME="bond0"
# export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5"

# export CUDA_DEVICE_MAX_CONNECTIONS=1


# __doc_head_address_start__
# Getting the node names
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=($nodes)

# head_node=${nodes_array[0]}

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

splits=$1
# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## env config
GPUS_PER_NODE=8
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
NNODES=$SLURM_NNODES


mics_shard_size=0
os=8
export MICS_SHARD_SIZE=${mics_shard_size}


torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
test_mics_config.py --zero=1 --mics_shard_size=${mics_shard_size} --os=${os}

# srun -p llm_s --gpus-per-task=8 --nodes=2 --tasks-per-node=1 test_mics.sh > log.log 2>&1
