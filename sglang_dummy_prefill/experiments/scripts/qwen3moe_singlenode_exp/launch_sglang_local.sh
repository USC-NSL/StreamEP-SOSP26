#!/usr/bin/bash

export NUM_GPUS=4
export WORLD_SIZE=$((NUM_GPUS))

export SGLANG_ATTN_MAX_BS=256
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$SGLANG_ATTN_MAX_BS
export SGLANG_TORCH_PROFILER_DIR="/scratch1/wangshao/sglang_profile_local_graph"
export SGLANG_MAX_RUNNING_REQUESTS=$(($SGLANG_ATTN_MAX_BS * $WORLD_SIZE))

ENABLE_FAKE_PREFILL=0
if [ $ENABLE_FAKE_PREFILL -eq 1 ]; then
    FAKE_PREFILL_CONFIG="--enable-fake-prefill"
else
    FAKE_PREFILL_CONFIG=""
fi

PREFILL_CONFIG="--chunked-prefill-size -1 --max-prefill-tokens 16384"

ENABLE_TBO=0
USE_FP8=0

if [ $USE_FP8 -eq 1 ]; then
    export SGLANG_DEEPEP_BF16_DISPATCH=false
    QUANT_CONFIG="--quantization fp8"
else
    export SGLANG_DEEPEP_BF16_DISPATCH=true
    QUANT_CONFIG=""
fi

if [ $ENABLE_TBO -eq 1 ]; then
    TBO_CONFIG="--enable-two-batch-overlap"
else
    TBO_CONFIG=""
fi

MODEL_CONFIG="--model /project2/seojinpa_1660/shaoyuw/models/Qwen3-30B-A3B-Instruct-2507 --trust-remote-code"

USE_DUMMY_WEIGHTS=1

if [ $USE_DUMMY_WEIGHTS -eq 1 ]; then
    MODEL_CONFIG="$MODEL_CONFIG --load-format dummy"
fi

DISABLE_CUDA_GRAPH=1

if [ $DISABLE_CUDA_GRAPH -eq 1 ]; then
    CUDA_GRAPH_CONFIG="--disable-cuda-graph"
else
    CUDA_GRAPH_CONFIG="--cuda-graph-max-bs $SGLANG_ATTN_MAX_BS"
fi

REDUCE_LOG=1

if [ $REDUCE_LOG -eq 1 ]; then
    LOG_LEVEL_HTTP="warning"
    LOG_LEVEL="warning"
else
    LOG_LEVEL_HTTP="info"
    LOG_LEVEL="info"
fi

cd ~/sglang

# EP

if [ $1 == "ep" ]; then

# python -m sglang.compile_deep_gemm \
#     --model /scratch1/wangshao/models/Qwen3-235B-A22B-Instruct-2507/ $QUANT_CONFIG \
#     --disable-radix-cache --chunked-prefill-size -1 --mem-fraction-static 0.7 \
#     --tp-size $WORLD_SIZE --dp-size $WORLD_SIZE --ep-size $WORLD_SIZE \
#     --enable-dp-attention --enable-dp-lm-head --trust-remote-code \
#     --moe-a2a-backend deepep --deepep-mode low_latency \
#     --max-running-requests $SGLANG_MAX_RUNNING_REQUESTS --cuda-graph-max-bs $SGLANG_ATTN_MAX_BS \
#     --load-format dummy --enable-fake-prefill $TBO_CONFIG \
#     --log-level-http warning --log-level warning

USE_DEEPEP=0
if [ $USE_DEEPEP -eq 1 ]; then
    MOE_A2A_BACKEND="--moe-a2a-backend deepep --deepep-mode normal"
else
    MOE_A2A_BACKEND=""
fi

python -m sglang.launch_server \
    $MODEL_CONFIG $QUANT_CONFIG \
    --disable-radix-cache $PREFILL_CONFIG --mem-fraction-static 0.8 \
    --tp-size $WORLD_SIZE --dp-size $WORLD_SIZE --ep-size $WORLD_SIZE \
    --enable-dp-attention --enable-dp-lm-head \
    $MOE_A2A_BACKEND \
    --max-running-requests $SGLANG_MAX_RUNNING_REQUESTS $CUDA_GRAPH_CONFIG \
     $FAKE_PREFILL_CONFIG $TBO_CONFIG \
    --log-level-http $LOG_LEVEL_HTTP --log-level $LOG_LEVEL

elif [ $1 == "tp" ]; then

# TP

python -m sglang.launch_server \
    $MODEL_CONFIG $QUANT_CONFIG \
    --disable-radix-cache $PREFILL_CONFIG --mem-fraction-static 0.8 \
    --tp-size $WORLD_SIZE \
    --max-running-requests $SGLANG_MAX_RUNNING_REQUESTS $CUDA_GRAPH_CONFIG \
     $FAKE_PREFILL_CONFIG \
    --log-level-http $LOG_LEVEL_HTTP --log-level $LOG_LEVEL

else
    echo "Invalid argument"
    exit 1
fi