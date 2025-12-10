#!/bin/bash
# Test script for Stage 1 Inference

echo "🧪 Testing Stage 1 Animation Cleanup Inference"
echo "================================================"
echo ""

# Check if checkpoint exists
if [ ! -f "epoch_014.pth" ]; then
    echo "⚠️  Warning: epoch_014.pth not found in current directory"
    echo "   Using default checkpoint path..."
fi

# Create output directory
mkdir -p test_output

echo "📁 Input folder: test_input/"
echo "📁 Output folder: test_output/"
echo "🔄 Phase: rough → clean"
echo ""

# Run inference
python stage1_inference.py \
    --input test_input/ \
    --phase rough \
    --target clean \
    --out test_output/ \
    --ckpt epoch_014.pth

echo ""
echo "✅ Test completed!"
echo "📂 Check results in: test_output/"

