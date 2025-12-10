#!/bin/bash
# Quick start script for Streamlit app

echo "🎨 Starting Stage 1 Animation Cleanup Web Interface..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit is not installed."
    echo "Please install it with: pip install streamlit"
    exit 1
fi

# Check NumPy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ -n "$NUMPY_VERSION" ]; then
    MAJOR_VERSION=$(echo $NUMPY_VERSION | cut -d. -f1)
    if [ "$MAJOR_VERSION" -ge 2 ]; then
        echo "⚠️  Warning: NumPy $NUMPY_VERSION detected (requires < 2.0.0)"
        echo "   Run './fix_numpy.sh' to fix this issue"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check if checkpoint exists
if [ ! -f "epoch_014.pth" ]; then
    echo "⚠️  Warning: epoch_014.pth not found in current directory"
    echo "   You can specify a different checkpoint in the web interface"
    echo ""
fi

echo "✅ Starting Streamlit server..."
echo "🌐 Open your browser to: http://localhost:8501"
echo ""

# Run streamlit
streamlit run app.py --server.headless true
