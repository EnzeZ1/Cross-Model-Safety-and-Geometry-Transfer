#!/bin/bash

set -e
set -o pipefail

export HF_HOME="/nobackup/enzez/hf_cache"
export TRANSFORMERS_CACHE="/nobackup/enzez/hf_cache"

export NCCL_TIMEOUT=600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

QWQ_GPUS="2,3,4"
R1_GPUS="5,6,7"

R1_HF="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
R1_TAG="r1-32b"
QWQ_HF="Qwen/QwQ-32B"
QWQ_TAG="QwQ"

WORK_DIR="$(pwd)"

skip()   { echo "  [skip] $1"; }
header() { echo ""; echo "==============================================================="
           echo "  $1"
           echo "==============================================================="; }
step()   { echo ""; echo "-- $1"; }

overlap_check() {
    local a="$1" b="$2"
    for g in ${a//,/ }; do
        for h in ${b//,/ }; do
            if [ "$g" = "$h" ]; then
                echo "ERROR: GPU $g listed in both QWQ_GPUS=[$a] and R1_GPUS=[$b]"
                exit 1
            fi
        done
    done
}

setup_subdir() {
    local SUB="$1"
    mkdir -p "$SUB"
    for f in "$WORK_DIR"/*.py; do
        ln -sf "$f" "$SUB/$(basename "$f")"
    done
}

run_pipeline() {
    local SUB="$1" GPUS="$2" HF="$3" TAG="$4"
    cd "$WORK_DIR/$SUB"

    {
        echo "[$TAG] === Pipeline start at $(date) ==="

        echo "[$TAG] data prep"
        python -u harmthoughts_data.py "$TAG"

        if ls actsvd_${TAG}*.safetensors 1>/dev/null 2>&1; then
            echo "[$TAG] ActSVD: skip (file exists)"
        else
            echo "[$TAG] ActSVD on GPUs $GPUS"
            python -u work_actsvd.py "$GPUS" "$HF" "$TAG" 100 100
        fi

        if ls dom_${TAG}*.safetensors 1>/dev/null 2>&1; then
            echo "[$TAG] DoM: skip (file exists)"
        else
            echo "[$TAG] DoM on GPUs $GPUS"
            python -u work_dom.py "$GPUS" "$HF" "$TAG"
        fi

        echo "[$TAG] === Pipeline done at $(date) ==="
    } > "$WORK_DIR/log_pipeline_${TAG}.txt" 2>&1
}

header "Sanity checks"

for f in harmthoughts_data.py work_actsvd.py work_dom.py \
         uni.py cross_actsvd.py cross_jsd.py real_transfer.py; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: missing required script: $f"
        exit 1
    fi
done
echo "  All required scripts present."

overlap_check "$QWQ_GPUS" "$R1_GPUS"
echo "  QwQ GPUs: $QWQ_GPUS"
echo "  r1 GPUs:  $R1_GPUS"

header "PHASE 1: parallel pipelines (data + ActSVD + DoM)"

setup_subdir run_qwq
setup_subdir run_r1

echo "  Launching QwQ pipeline (GPUs $QWQ_GPUS)  -> log_pipeline_${QWQ_TAG}.txt"
( run_pipeline run_qwq "$QWQ_GPUS" "$QWQ_HF"  "$QWQ_TAG" ) &
QWQ_PID=$!

sleep 30

echo "  Launching r1 pipeline  (GPUs $R1_GPUS)   -> log_pipeline_${R1_TAG}.txt"
( run_pipeline run_r1 "$R1_GPUS" "$R1_HF" "$R1_TAG" ) &
R1_PID=$!

echo ""
echo "  Both pipelines running. Monitor with:"
echo "    tail -F log_pipeline_${QWQ_TAG}.txt log_pipeline_${R1_TAG}.txt"
echo ""
echo "  Waiting for completion ..."

QWQ_RC=0; R1_RC=0
wait $QWQ_PID || QWQ_RC=$?
wait $R1_PID  || R1_RC=$?

cd "$WORK_DIR"

if [ $QWQ_RC -ne 0 ] || [ $R1_RC -ne 0 ]; then
    echo ""
    echo "  ERROR: pipeline failed:"
    [ $QWQ_RC -ne 0 ] && echo "    QwQ exit $QWQ_RC -- see log_pipeline_${QWQ_TAG}.txt"
    [ $R1_RC -ne 0 ]  && echo "    r1  exit $R1_RC -- see log_pipeline_${R1_TAG}.txt"
    exit 1
fi

echo ""
echo "  Both pipelines done. Collecting outputs ..."
for sub in run_qwq run_r1; do
    for f in "$sub"/actsvd_*.safetensors "$sub"/actsvd_results_*.txt "$sub"/dom_*.safetensors; do
        [ -e "$f" ] && mv -v "$f" "$WORK_DIR/"
    done
done

DA=$(ls dom_${R1_TAG}*.safetensors    2>/dev/null | head -1)
DB=$(ls dom_${QWQ_TAG}*.safetensors   2>/dev/null | head -1)
AA=$(ls actsvd_${R1_TAG}*.safetensors 2>/dev/null | head -1)
AB=$(ls actsvd_${QWQ_TAG}*.safetensors 2>/dev/null | head -1)

for v in DA DB AA AB; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v safetensors not found after Phase 1"; exit 1
    fi
done
echo "  DoM:    $DA  vs  $DB"
echo "  ActSVD: $AA  vs  $AB"

header "PHASE 2: Cross-model geometry (CPU)"

step "uni.py -- CKA / RSA on DoM"
python -u uni.py "$DA" "$DB" "$R1_TAG" "$QWQ_TAG" \
    2>&1 | tee log_cross_dom.txt

step "cross_actsvd.py -- principal angles on ActSVD"
python -u cross_actsvd.py "$AA" "$AB" "$R1_TAG" "$QWQ_TAG" \
    2>&1 | tee log_cross_actsvd.txt

header "PHASE 3: Transfer studies (parallel)"

setup_subdir run_transfer_r1_to_qwq
setup_subdir run_transfer_qwq_to_r1

ln -sf "$WORK_DIR/$DA" "$WORK_DIR/run_transfer_r1_to_qwq/$DA"
ln -sf "$WORK_DIR/$DB" "$WORK_DIR/run_transfer_qwq_to_r1/$DB"

(
    cd run_transfer_r1_to_qwq
    {
        echo "=== r1 -> QwQ transfer (GPUs $QWQ_GPUS) at $(date) ==="
        python -u harmthoughts_data.py "$QWQ_TAG"
        python -u real_transfer.py "$QWQ_GPUS" "$QWQ_HF" "$QWQ_TAG" \
            "$DA" "$R1_TAG" "$QWQ_TAG"
        echo "=== r1 -> QwQ done at $(date) ==="
    } > "$WORK_DIR/log_real_transfer_${R1_TAG}_to_${QWQ_TAG}.txt" 2>&1
) &
T1_PID=$!

sleep 30

(
    cd run_transfer_qwq_to_r1
    {
        echo "=== QwQ -> r1 transfer (GPUs $R1_GPUS) at $(date) ==="
        python -u harmthoughts_data.py "$R1_TAG"
        python -u real_transfer.py "$R1_GPUS" "$R1_HF" "$R1_TAG" \
            "$DB" "$QWQ_TAG" "$R1_TAG"
        echo "=== QwQ -> r1 done at $(date) ==="
    } > "$WORK_DIR/log_real_transfer_${QWQ_TAG}_to_${R1_TAG}.txt" 2>&1
) &
T2_PID=$!

echo "  Both transfers running. Monitor:"
echo "    tail -F log_real_transfer_*.txt"
echo "  Waiting ..."

T1_RC=0; T2_RC=0
wait $T1_PID || T1_RC=$?
wait $T2_PID || T2_RC=$?
cd "$WORK_DIR"

if [ $T1_RC -ne 0 ] || [ $T2_RC -ne 0 ]; then
    echo "  ERROR: transfer failed:"
    [ $T1_RC -ne 0 ] && echo "    r1->QwQ exit $T1_RC"
    [ $T2_RC -ne 0 ] && echo "    QwQ->r1 exit $T2_RC"
    exit 1
fi

for sub in run_transfer_r1_to_qwq run_transfer_qwq_to_r1; do
    for f in "$sub"/real_transfer_*.txt "$sub"/jsd_*.json; do
        [ -e "$f" ] && mv -v "$f" "$WORK_DIR/"
    done
done

header "PHASE 4: Cross-JSD comparison (CPU)"

[ ! -f "jsd_${R1_TAG}.json" ]  && { echo "ERROR: jsd_${R1_TAG}.json missing";  exit 1; }
[ ! -f "jsd_${QWQ_TAG}.json" ] && { echo "ERROR: jsd_${QWQ_TAG}.json missing"; exit 1; }

step "cross_jsd.py"
python -u cross_jsd.py "jsd_${R1_TAG}.json" "jsd_${QWQ_TAG}.json" \
    "$R1_TAG" "$QWQ_TAG" 2>&1 | tee log_cross_jsd.txt

header "All experiments done!"
echo ""
echo "  Per-model artifacts:"
ls -lh actsvd_*.safetensors dom_*.safetensors jsd_*.json 2>/dev/null | sed 's/^/    /'
echo ""
echo "  Transfer summaries:"
ls -lh real_transfer_*.txt 2>/dev/null | sed 's/^/    /'
echo ""
echo "  Cross-model logs:"
ls -lh log_cross_*.txt 2>/dev/null | sed 's/^/    /'
