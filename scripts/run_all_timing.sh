#!/bin/bash
# Master timing script for all 10 trajectory inference models on 3 datasets.
# Run this after all other GPU processes have been paused.
# Usage: bash scripts/run_all_timing.sh [model_name] [dataset]
# If no args: runs ALL model x dataset combinations sequentially.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNERS_DIR="$REPO_ROOT/runners"
BASELINES_DIR="$REPO_ROOT/baselines"
DATA_DIR="$REPO_ROOT/OTP-FM/data"
RESULTS_FILE="$REPO_ROOT/scripts/timing_results.csv"
MAX_SECONDS=300  # 5 minute cutoff

if [ ! -f "$RESULTS_FILE" ]; then
    echo "model,dataset,total_time_s,iters_completed,total_iters,time_per_iter_s,extrapolated_total_s,status" > "$RESULTS_FILE"
fi

log_result() {
    local model="$1" dataset="$2" total_time="$3" iters_done="$4" total_iters="$5" time_per_iter="$6" extrap_total="$7" status="$8"
    echo "$model,$dataset,$total_time,$iters_done,$total_iters,$time_per_iter,$extrap_total,$status" >> "$RESULTS_FILE"
    echo "  -> $model | $dataset | ${total_time}s done | ${iters_done}/${total_iters} iters | ${time_per_iter}s/iter | extrap: ${extrap_total}s | $status"
}

run_with_timeout() {
    local cmd="$1"
    timeout --signal=SIGINT ${MAX_SECONDS}s bash -c "$cmd" 2>&1 || true
}

echo "=========================================="
echo " Training Time Benchmark"
echo " Max time per run: ${MAX_SECONDS}s"
echo " Repo root:        $REPO_ROOT"
echo " Results:          $RESULTS_FILE"
echo "=========================================="

# ============================================================
# 1. TrajectoryNet
# ============================================================
run_trajectorynet() {
    local dataset="$1"
    echo ""
    echo ">>> TrajectoryNet - $dataset"
    cd "$BASELINES_DIR/TrajectoryNet"

    local dim niters extra_args=""
    case "$dataset" in
        eb5)    dim=5;   niters=10000; extra_args="--dataset EB-PCA --max_dim 5" ;;
        eb100)  dim=100; niters=10000; extra_args="--dataset EB-PCA --max_dim 100" ;;
        cite50) dim=50;  niters=10000; extra_args="--dataset $DATA_DIR/cite_pca50.npz --embedding_name pca --max_dim 50 --whiten" ;;
    esac

    local cmd="conda run -n env_trajectorynet python -m TrajectoryNet.main $extra_args --niters $niters --batch_size 1000 --dims 64-64-64 --layer_type concatsquash --lr 1e-3 --val_freq 999999 --save_freq 999999 --save /tmp/tnet_timing_${dataset}"
    run_with_timeout "$cmd"
}

# ============================================================
# 2. WLF-UOT
# ============================================================
run_wlf() {
    local dataset="$1"
    echo ""
    echo ">>> WLF-UOT - $dataset"
    cd "$BASELINES_DIR/wl-mechanics"

    local config
    case "$dataset" in
        eb5)    config="configs/embrio/ubot.py" ;;
        eb100)  config="configs/embrio100/ubot.py" ;;
        cite50) config="configs/cite50/ubot.py" ;;
    esac

    local cmd="WANDB_MODE=disabled conda run -n wlf python main.py --config $config --workdir /tmp/wlf_timing_${dataset}"
    run_with_timeout "$cmd"
}

# ============================================================
# 3. SF2M
# ============================================================
run_sf2m() {
    local dataset="$1"
    echo ""
    echo ">>> SF2M - $dataset"
    cd "$RUNNERS_DIR"

    local cmd="conda run -n env_sf2m python train_sf2m_timing.py --dataset $dataset --epochs 100 --iters-per-epoch 100 --batch-size 256"
    run_with_timeout "$cmd"
}

# ============================================================
# 4. DeepRUOT
# ============================================================
run_deepruot() {
    local dataset="$1"
    echo ""
    echo ">>> DeepRUOT - $dataset"
    cd "$RUNNERS_DIR"

    local cmd="conda run -n env_deepruot python train_deepruot_timing.py --dataset $dataset --epochs 60 --device cuda"
    run_with_timeout "$cmd"
}

# ============================================================
# 5. VGFM
# ============================================================
run_vgfm() {
    local dataset="$1"
    echo ""
    echo ">>> VGFM - $dataset"
    cd "$RUNNERS_DIR"

    local cmd="conda run -n env_vgfm python train_vgfm_timing.py --dataset $dataset --n-pretrain-epochs 2000 --n-train-epochs 0"
    run_with_timeout "$cmd"
}

# ============================================================
# 6. DMSB
# ============================================================
run_dmsb() {
    local dataset="$1"
    echo ""
    echo ">>> DMSB - $dataset"
    cd "$BASELINES_DIR/DMSB"

    local extra_args
    case "$dataset" in
        eb5)    extra_args="--problem-name RNAsc --RNA-dim 5 --dir /tmp/dmsb_timing_eb5" ;;
        eb100)  extra_args="--problem-name RNAsc --RNA-dim 100 --dir /tmp/dmsb_timing_eb100" ;;
        cite50) extra_args="--problem-name CITEseq --dir /tmp/dmsb_timing_cite50" ;;
    esac

    local cmd="conda run -n env_dmsb python main.py $extra_args --num-itr 2000 --log-tb"
    run_with_timeout "$cmd"
}

# ============================================================
# 7. JKONet*
# ============================================================
run_jkonet() {
    local dataset="$1"
    echo ""
    echo ">>> JKONet* - $dataset"

    local jko_dataset
    case "$dataset" in
        eb5)    jko_dataset="EB_5D" ;;
        eb100)  jko_dataset="EB_100D" ;;
        cite50) jko_dataset="CITE_50D" ;;
    esac

    local cmd="cd $BASELINES_DIR/jkonet-star && WANDB_MODE=disabled conda run -n env_jkonet python train.py --dataset $jko_dataset --solver jkonet-star-time-potential --epochs 100"
    run_with_timeout "$cmd"
}

# ============================================================
# 8. iJKOnet
# ============================================================
run_ijkonet() {
    local dataset="$1"
    echo ""
    echo ">>> iJKOnet - $dataset"
    cd "$BASELINES_DIR/iJKOnet"

    local extra_args
    case "$dataset" in
        eb5)    extra_args="--dataset RAW_RNA_eb_5 --K 4 --array-tau 0.01,0.01,0.01,0.01" ;;
        eb100)  extra_args="--dataset RAW_RNA_eb_100 --K 4 --array-tau 0.01,0.01,0.01,0.01" ;;
        cite50) extra_args="--dataset RAW_RNA_multi_50 --K 3 --array-tau 0.01,0.01,0.01" ;;
    esac

    local cmd="conda run -n env_ijkonet python train.py --solver inverse-jkonet-multimap-potential $extra_args --epochs 5000 --seed 0"
    run_with_timeout "$cmd"
}

# ============================================================
# 9. MMFM
# ============================================================
run_mmfm() {
    local dataset="$1"
    echo ""
    echo ">>> MMFM - $dataset"
    cd "$RUNNERS_DIR"

    local cmd="conda run -n env_mmfm python train_mmfm_timing.py --dataset $dataset --epochs 300 --iters-per-epoch 100 --batch-size 256"
    run_with_timeout "$cmd"
}

# ============================================================
# 10. OTP-FM
# ============================================================
run_otpfm() {
    local dataset="$1"
    echo ""
    echo ">>> OTP-FM - $dataset"
    cd "$REPO_ROOT"

    local cmd
    case "$dataset" in
        eb5)    cmd="pixi run bash -c 'cd OTP-FM && python experiments/train.py --dataset singlecell --pca-dim 5 --config configs/singlecell/W2.json --epochs 300 --tag timing_eb5'" ;;
        eb100)  cmd="pixi run bash -c 'cd OTP-FM && python experiments/train.py --dataset singlecell --config configs/singlecell/W2.json --epochs 300 --tag timing_eb100'" ;;
        cite50) cmd="pixi run python OTP-FM/experiments/train.py --dataset citeseq --config OTP-FM/configs/singlecell/W2.json --epochs 300 --tag timing_cite50" ;;
    esac
    run_with_timeout "$cmd"
}

# ============================================================
# Main dispatcher
# ============================================================
MODELS="trajectorynet wlf sf2m deepruot vgfm dmsb jkonet ijkonet mmfm otpfm"
DATASETS="eb5 eb100 cite50"

if [ "$1" != "" ]; then
    MODELS="$1"
fi
if [ "$2" != "" ]; then
    DATASETS="$2"
fi

for model in $MODELS; do
    for dataset in $DATASETS; do
        echo ""
        echo "============================================"
        echo "  Running: $model on $dataset"
        echo "  Time: $(date)"
        echo "============================================"

        case "$model" in
            trajectorynet) run_trajectorynet "$dataset" ;;
            wlf)           run_wlf "$dataset" ;;
            sf2m)          run_sf2m "$dataset" ;;
            deepruot)      run_deepruot "$dataset" ;;
            vgfm)          run_vgfm "$dataset" ;;
            dmsb)          run_dmsb "$dataset" ;;
            jkonet)        run_jkonet "$dataset" ;;
            ijkonet)       run_ijkonet "$dataset" ;;
            mmfm)          run_mmfm "$dataset" ;;
            otpfm)         run_otpfm "$dataset" ;;
            *)             echo "Unknown model: $model" ;;
        esac
    done
done

echo ""
echo "=========================================="
echo " All timing runs complete!"
echo " Results saved to: $RESULTS_FILE"
echo "=========================================="
