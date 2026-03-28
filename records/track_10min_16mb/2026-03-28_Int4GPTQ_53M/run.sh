#!/bin/bash
# V27c: 21-level quantization (clip_range=10) + 42M model (dim=576, MHA 8/8)
# Fewer quant levels → better zstd compression → fit bigger model
set -euo pipefail
cd /root/pg_repo

export SEED=${SEED:-1337}
export RUN_ID="v27c_21lvl_42M_seed${SEED}"

# Model architecture
export MODEL_DIM=576
export NUM_HEADS=8
export NUM_KV_HEADS=8
export NUM_LAYERS=11
export MLP_MULT=3.5
export BIGRAM_VOCAB_SIZE=8192
export BIGRAM_DIM=128
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Training
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# Quantization: 21 levels (clip_range=10)
export QUANT_CLIP_RANGE=10
export LATE_QAT_THRESHOLD=0.5
export QAT_ENABLED=0
export PRUNE_PCT=0.02

# SWA + EMA
export SWA_ENABLED=1
export SWA_EVERY=50

# Eval
export EVAL_STRIDE=32
export TTT_TEMPERATURE=0.98

# TTT
export TTT_EPOCHS=3
export TTT_LR=0.0005
export TTT_FREEZE_BLOCKS=2
export TTT_CHUNK_TOKENS=65536
export TTT_OPTIMIZER=adamw

echo "=== V27c: 21-level quant (cr=10) 42M (dim=$MODEL_DIM) seed=$SEED ==="
echo "=== Expected: ~42M params, ~14.7MB artifact (0.349 B/p) ==="

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "logs/${RUN_ID}.log"
