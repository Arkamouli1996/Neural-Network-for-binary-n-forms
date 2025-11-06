#!/bin/bash
# Master script to run the entire pipeline

echo "======================================================================"
echo "TRIPLE-ROOT SURFACE NEURAL NETWORK PIPELINE"
echo "======================================================================"

# Step 1: Generate data (if not already done)
if [ ! -f "data/X_train.csv" ]; then
    echo ""
    echo "[1/4] Generating dataset..."
    julia julia/generate_data.jl
    if [ $? -ne 0 ]; then
        echo "Error generating data!"
        exit 1
    fi
else
    echo "[1/4] Dataset already exists, skipping generation"
fi

# Step 2: Visualize data
echo ""
echo "[2/4] Visualizing data..."
python3 python/visualize.py
if [ $? -ne 0 ]; then
    echo "Error visualizing data!"
    exit 1
fi

# Step 3: Train model
echo ""
echo "[3/4] Training neural network..."
python3 python/train.py
if [ $? -ne 0 ]; then
    echo "Error training model!"
    exit 1
fi

# Step 4: Test model
echo ""
echo "[4/4] Testing model..."
python3 python/test.py
if [ $? -ne 0 ]; then
    echo "Error testing model!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - data/X_train.csv, y_train.csv (training data)"
echo "  - plot/data_visualization.png (data visualization)"
echo "  - plot/tsne_visualization.png (t-SNE plot)"
echo "  - plot/training_curves.png (training curves)"
echo "  - python/triple_root_model.pth (trained model)"
echo ""
