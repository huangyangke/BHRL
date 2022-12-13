#!/usr/bin/env bash

CONFIG="configs/BHRL.py"
GPUS=1
PORT=${PORT:-29500}
seed=100 #固定随机种子
#no_validate='--no-validate'
no_validate=''

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch --seed ${seed} ${no_validate}
