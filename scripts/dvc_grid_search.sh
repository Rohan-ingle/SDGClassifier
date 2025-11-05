#!/bin/bash
# Grid search for Random Forest hyperparameters using DVC experiments

set -e

echo "=========================================="
echo "DVC Grid Search - Random Forest"
echo "=========================================="
echo ""

# Define parameter ranges
DEPTHS=(5 10)
ESTIMATORS=(50)
MIN_SAMPLES_SPLIT=(2 10)

echo "Grid configuration:"
echo "  Max depths: ${DEPTHS[@]}"
echo "  N estimators: ${ESTIMATORS[@]}"
echo "  Min samples split: ${MIN_SAMPLES_SPLIT[@]}"
echo ""

TOTAL=$((${#DEPTHS[@]} * ${#ESTIMATORS[@]} * ${#MIN_SAMPLES_SPLIT[@]}))
echo "Total experiments to queue: $TOTAL"
echo ""

# Ask for confirmation
read -p "Queue all $TOTAL experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Queue all experiments
COUNT=0
for depth in "${DEPTHS[@]}"; do
  for est in "${ESTIMATORS[@]}"; do
    for split in "${MIN_SAMPLES_SPLIT[@]}"; do
      COUNT=$((COUNT + 1))
      EXP_NAME="rf-d${depth}-e${est}-s${split}"
      
      echo "[$COUNT/$TOTAL] Queueing: $EXP_NAME"
      
      dvc exp run --queue \
        --set-param "model.max_depth=$depth" \
        --set-param "model.n_estimators=$est" \
        --set-param "model.min_samples_split=$split" \
        --name "$EXP_NAME"
    done
  done
done

echo ""
echo "=========================================="
echo "All experiments queued!"
echo "=========================================="
echo ""
echo "To run experiments:"
echo "  Sequential: dvc queue start"
echo "  Parallel (4 workers): dvc queue start --jobs 4"
echo ""
echo "To view queue: dvc queue status"
echo "To view results: dvc exp show"
echo ""
