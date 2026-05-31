#!/bin/bash
# End-to-end SAM2Object runner for ONE dataset, driven by a .txt scan list.
# Runs the README preprocessing + segmentation stages, then the 3D evaluation +
# point-cloud export, for every scan id in $SCAN_LIST. Works for 3RScan and ScanNet
# (launch one dataset per job). Give access: chmod +x scripts/segmentation/run_pipeline.sh
set -euo pipefail

# ============================================================================
# Config  (edit these)
# ============================================================================
DATASET="3RScan"                 # 3RScan | ScanNet  (exact case)
YOUR_ROOT_DIR="/cluster/home/ealegret"
YOUR_SCRATCH_DIR="/cluster/scratch/ealegret"
PROJECT_DIR="$YOUR_ROOT_DIR/3d_vision_project"

RUN_SEGMENTATION=true            # phase 1: preprocess + run SAM2Object (needs GPU + .sam2object venv)
RUN_EVAL=true                    # phase 2: evaluate + export point clouds (.venvv venv)
EVAL_MODE="both"                 # both | objects-only | all  (must include objects-only for top-N)
IOU="0.25"                       # IoU threshold for the colour-matched point clouds
TOP_N=5                          # keep plys + 2D_masks only for the best N scenes (objects-only F1@0.5)
CLEANUP_INTERMEDIATE=true        # after each scene's prediction exists, delete its ~0.9GB of intermediates
CLEANUP_DRYRUN=false             # true = only print what cleanup WOULD delete (no removal)

# Newline-delimited scan-id list (one id per line). Overridable from the env, e.g.
#   SCAN_LIST=/path/to/one_scene.txt bash scripts/segmentation/run_pipeline.sh
# so you can test one scene, then run the full list with the default.
SCAN_LIST="${SCAN_LIST:-$PROJECT_DIR/objectx_complete_scans.txt}"
# other options:
#   $YOUR_SCRATCH_DIR/sam2object/files/sam2object_scans.txt
#   /cluster/project/cvg/data/3RScan/files/val_resplit_scans.txt

# ---- per-dataset roots (input GT meshes + scratch workdir) -----------------
case "$DATASET" in
  3RScan)
    SAM2OBJECT_DATA_PATH="/cluster/project/cvg/data/3RScan/scenes"   # GT meshes (points source)
    DATA_ROOT_DIR="$YOUR_SCRATCH_DIR/sam2object"                     # scratch workdir / outputs
    ;;
  ScanNet)
    SAM2OBJECT_DATA_PATH="/cluster/project/cvg/data/scannet/scans"
    DATA_ROOT_DIR="$YOUR_SCRATCH_DIR/sam2object_scannet"
    ;;
  *) echo "ERROR: DATASET must be '3RScan' or 'ScanNet', got '$DATASET'"; exit 1 ;;
esac
GT_ROOT="$SAM2OBJECT_DATA_PATH"                                      # evaluator reads GT here
PRED_NPY_PATTERN="$DATA_ROOT_DIR/scans/{scan}/results/{scan}_labels_fine_global.npy"
OUT_DIR="$DATA_ROOT_DIR/eval_batch"

# Persistent (HOME, tiny, survives scratch purge) append-only per-scan metrics ledger.
# Never overwritten: every eval appends one row per (scan,mode); accumulates across runs.
METRICS_DIR="$PROJECT_DIR/results/$DATASET"
METRICS_TXT="$METRICS_DIR/metrics_per_scan.txt"

# ============================================================================
# Scan list -> comma-separated SCAN_IDS (single source of truth for all stages)
# ============================================================================
[ -f "$SCAN_LIST" ] || { echo "ERROR: SCAN_LIST not found: $SCAN_LIST"; exit 1; }
SCAN_IDS="$(grep -vE '^[[:space:]]*$' "$SCAN_LIST" | sed 's/[[:space:]]//g' | paste -sd, -)"
[ -n "$SCAN_IDS" ] || { echo "ERROR: SCAN_LIST is empty: $SCAN_LIST"; exit 1; }

echo "DATASET:   $DATASET"
echo "Scans:     $(grep -cvE '^[[:space:]]*$' "$SCAN_LIST") from $SCAN_LIST"
echo "GT/points: $SAM2OBJECT_DATA_PATH"
echo "Workdir:   $DATA_ROOT_DIR"

# ============================================================================
# Phase 1: preprocessing + SAM2Object segmentation  (.sam2object venv, GPU)
# ============================================================================
if [ "$RUN_SEGMENTATION" = true ]; then
  echo "========== Phase 1: SAM2Object segmentation (per-scene) =========="
  source "$YOUR_SCRATCH_DIR/.sam2object/bin/activate"
  cd "$PROJECT_DIR"

  export VLSG_SPACE="$PROJECT_DIR"
  export PYTHONPATH="$VLSG_SPACE:${PYTHONPATH:-}:$VLSG_SPACE/dependencies/gaussian-splatting"
  export DATASET DATA_ROOT_DIR SAM2OBJECT_DATA_PATH
  export TORCH_HOME="$YOUR_SCRATCH_DIR/torch_cache"
  export XDG_CACHE_HOME="$YOUR_SCRATCH_DIR/.cache"
  # Reduce CUDA fragmentation OOM in SAM2 video propagation on long scenes (24GB GPU).
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # NOTE: do NOT export SAM2OBJECT_DIR -- mask_convert/seg_tracking rely on its default.
  mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$DATA_ROOT_DIR/files"

  SEGTRACK="$PROJECT_DIR/dependencies/SAM2Object/segtrack"

  # The bulky per-scene segtrack/outputs (~0.9GB/scene) defaults to HOME (near quota),
  # so we redirect it to scratch via a symlink (the dir is gitignored). The HOME symlink
  # persists but scratch may PURGE its target between runs, leaving a dangling link -- so
  # ALWAYS (re)create the target here, not just when the symlink is missing.
  if [ -L "$SEGTRACK/outputs" ]; then
    mkdir -p "$(readlink "$SEGTRACK/outputs")"
  elif [ ! -e "$SEGTRACK/outputs" ]; then
    mkdir -p "$DATA_ROOT_DIR/segtrack_outputs"
    ln -s "$DATA_ROOT_DIR/segtrack_outputs" "$SEGTRACK/outputs"
    echo "[setup] segtrack/outputs -> $DATA_ROOT_DIR/segtrack_outputs (scratch)"
  fi

  # Run all 5 stages for ONE scene. Explicit `|| return` so a single bad scene
  # cannot abort the whole batch (the model reloads per scene -- the price of
  # per-scene isolation + resumability).
  run_one_scene() {
    local scene="$1"
    export SCAN_IDS="$scene"
    # Make this scene's input reachable to stages 1-2, which read
    # $DATA_ROOT_DIR/scenes/<scene>. Symlink (read-only) to the project source so we
    # neither copy bulk frames nor ever write into the read-only project tree. Never
    # removed by cleanup. A pre-existing real dir (e.g. 5341b7e3) is left as-is.
    mkdir -p "$DATA_ROOT_DIR/scenes"
    [ -e "$DATA_ROOT_DIR/scenes/$scene" ] || ln -s "$SAM2OBJECT_DATA_PATH/$scene" "$DATA_ROOT_DIR/scenes/$scene"
    cd "$SEGTRACK"                          || return 1
    python dataprocess/extract_only_jpg.py || return 1
    python dataprocess/get_posed_images.py || return 1
    python seg_tracking.py                 || return 1
    python mask_convert.py                 || return 1
    ( cd "$SEGTRACK/../graphclustering" && bash scripts/seg_scannet.sh ) || return 1
    return 0
  }

  # Delete a scene's intermediate dirs (NOT results), but only once its prediction
  # exists -- so failed scenes keep their intermediates for debugging. Scene-id
  # scoped paths, no globs. Set CLEANUP_DRYRUN=true to print without deleting.
  cleanup_scene() {
    local scene="$1"
    [ "$CLEANUP_INTERMEDIATE" = true ] || return 0
    local pred="$DATA_ROOT_DIR/scans/$scene/results/${scene}_labels_fine_global.npy"
    [ -f "$pred" ] || { echo "[cleanup-skip] $scene (no prediction yet)"; return 0; }
    # NOTE: 2D_masks is intentionally NOT deleted here -- it is kept until Phase 3,
    # which retains it only for the top-N scenes and prunes the rest.
    local targets=(
      "$DATA_ROOT_DIR/color_images_cluster/$scene"
      "$DATA_ROOT_DIR/posed_images/$scene"
      "$SEGTRACK/outputs/$scene"
    )
    for t in "${targets[@]}"; do
      [ -e "$t" ] || continue
      if [ "$CLEANUP_DRYRUN" = true ]; then
        echo "[cleanup-dryrun] would rm -rf $t"
      else
        rm -rf "$t" && echo "[cleanup] removed $t"
      fi
    done
  }

  SEG_FAILED=()
  while IFS= read -r scene; do
    [ -n "$scene" ] || continue
    pred="$DATA_ROOT_DIR/scans/$scene/results/${scene}_labels_fine_global.npy"
    if [ -f "$pred" ]; then
      echo "[done]  $scene (prediction exists, skip)"
      cleanup_scene "$scene"   # reclaim leftovers from an earlier interrupted run
      continue
    fi
    echo "---------- scene: $scene ----------"
    if run_one_scene "$scene"; then
      echo "[ok]    $scene"
      cleanup_scene "$scene"
    else
      echo "[FAIL]  $scene (logged, continuing)"
      SEG_FAILED+=("$scene")
    fi
  done < <(grep -vE '^[[:space:]]*$' "$SCAN_LIST" | sed 's/[[:space:]]//g')

  echo "[OK] segmentation finished. failed scenes: ${#SEG_FAILED[@]}"
  [ "${#SEG_FAILED[@]}" -eq 0 ] || printf '  [FAIL] %s\n' "${SEG_FAILED[@]}"
  deactivate || true
fi

# ============================================================================
# Phase 2: evaluation -> report.json  (.venvv venv, CPU)
# ============================================================================
if [ "$RUN_EVAL" = true ]; then
  echo "========== Phase 2: evaluation =========="
  source "$YOUR_SCRATCH_DIR/.venvv/bin/activate"
  cd "$PROJECT_DIR"
  mkdir -p "$OUT_DIR/plys"

  mkdir -p "$METRICS_DIR"
  python src/evaluation/evaluate_sam2object_3d.py \
    --dataset "$DATASET" --gt_root "$GT_ROOT" \
    --split_file "$SCAN_LIST" --pred_npy "$PRED_NPY_PATTERN" \
    --mode "$EVAL_MODE" --out "$OUT_DIR/report.json" \
    --append_txt "$METRICS_TXT" --run_tag "$(date +%F_%H%M%S)_${SLURM_JOB_ID:-local}"
  echo "[OK] eval report: $OUT_DIR/report.json ; ledger: $METRICS_TXT"

  # Best scenes (.txt) + general average over ALL accumulated scenes (.txt)
  python src/evaluation/summarize_metrics.py \
    --ledger "$METRICS_TXT" --mode both --metric f1 --iou 0.5 --top "$TOP_N" \
    --out_best "$METRICS_DIR/best_metrics.txt" \
    --out_avg "$METRICS_DIR/general_metrics.txt"

  # ==========================================================================
  # Phase 3: keep plys + 2D_masks only for the top-N scenes; prune the rest
  # ==========================================================================
  if [ -f "$OUT_DIR/report.json" ]; then
    echo "========== Phase 3: top-$TOP_N selection =========="
    MANIFEST="$OUT_DIR/top${TOP_N}_scenes.txt"
    python src/evaluation/select_top_scenes.py \
      --report "$OUT_DIR/report.json" --mode objects-only \
      --metric f1 --iou 0.5 --top "$TOP_N" > "$MANIFEST"
    mapfile -t TOP < <(awk -F'\t' '{print $1}' "$MANIFEST")

    if [ "${#TOP[@]}" -gt 0 ]; then
      echo "[top-$TOP_N] $(printf '%s ' "${TOP[@]}")"
      # export point clouds for the winners only
      python src/evaluation/export_segmented_ply.py \
        --dataset "$DATASET" --gt_root "$GT_ROOT" \
        --scans "${TOP[@]}" --pred_npy "$PRED_NPY_PATTERN" \
        --out_dir "$OUT_DIR/plys" --mode objects-only --iou "$IOU" --points_only

      # prune 2D_masks: keep the winners, delete every other listed scene's
      while IFS= read -r scene; do
        [ -n "$scene" ] || continue
        keep=no
        for t in "${TOP[@]}"; do [ "$t" = "$scene" ] && keep=yes && break; done
        if [ "$keep" = no ]; then rm -rf "$DATA_ROOT_DIR/2D_masks/$scene"; fi
      done < <(grep -vE '^[[:space:]]*$' "$SCAN_LIST" | sed 's/[[:space:]]//g')
      echo "[OK] top-$TOP_N plys: $OUT_DIR/plys/ ; 2D_masks kept for winners; manifest: $MANIFEST"
    else
      echo "[WARN] no scenes ranked (empty report); skipping top-N export/prune."
    fi
  fi
fi

echo "========== DONE ($DATASET) =========="
