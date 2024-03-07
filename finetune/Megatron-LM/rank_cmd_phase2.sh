set -ex

NODE_RANK=$1
NODE_CNT=$2
MASTER=$3
MASTER_PORT=$4
TP=$5
PP=$6
MBSZ=$7
LOGDIR=$8

outfile=$LOGDIR/$NODE_RANK.$(hostname).stdout
errfile=$LOGDIR/$NODE_RANK.$(hostname).stderr

DATA_PATH=""

CKPT=/workspace/models/phase2_tp${TP}_pp${PP}_dev
SAVE_DIR=${CKPT}_save
ITERS=214301

# if the file "{SAVE_DIR}/latest_checkpointed_iteration.txt" exists, then load from it
# otherwise, load from the directory specified by ${CKPT}

if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Loading from ${SAVE_DIR}"
    NO_LOAD_OPTIM="" # leave no_load_optim to empty string
    LOAD_DIR=${SAVE_DIR}
else
    echo "Loading from ${CKPT}"
    NO_LOAD_OPTIM="--no-load-optim --no-load-rng "
    LOAD_DIR=${CKPT}
fi

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node 4 --nnodes $NODE_CNT --node_rank $NODE_RANK --master_addr $MASTER --master_port $MASTER_PORT \
    /workspace/crystalcoder-train/finetune/Megatron-LM/pretrain_crystalcoder_inst.py --seq-length 2048 \
    --micro-batch-size $MBSZ --global-batch-size 2048 --max-position-embeddings 2048 --tokenizer-type NullTokenizer \
    --load ${LOAD_DIR} --save ${SAVE_DIR} \
    --exit-on-missing-checkpoint --use-checkpoint-args \
    $NO_LOAD_OPTIM --bf16 \
    --no-position-embedding \
    --norm-epsilon 1e-5 \
    --use-flash-attn --no-query-key-layer-scaling --train-iters ${ITERS} \
    --lr 0.0087825 --lr-decay-iters ${ITERS} --lr-decay-style linear --lr-warmup-iters 86 --min-lr 0.00013679 --weight-decay 0.1 --clip-grad 1.0 \
    --data-path ${DATA_PATH} \
    --vocab-size 32032 --dataloader-type cyclic --log-interval 1 --save-interval 100 --eval-interval 2000 --eval-iters 10 \
    --distributed-backend nccl --use-rotary-position-embeddings --rotary-percent 0.25 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-08 --attention-dropout 0.0 --hidden-dropout 0.0 \
    --attention-softmax-in-fp32 --overlap-grad-reduce \
    --tensorboard-dir ${CKPT}_save/logs \
    --seed 42 \
    --sequence-parallel --use-mup --mup-width-scale 0.0625 --mup-output-alpha 2.22 --mup-embeddings-scale 14.6 --mup-scale-qk-dot-by-d \
    --tensorboard-queue-size 10 --log-timers-to-tensorboard --log-memory-to-tensorboard --log-params-norm \
    --rotary-interleave-repeat --rotary-stay-fp32 \
    1> $outfile 2> $errfile


