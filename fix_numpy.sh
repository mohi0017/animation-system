#!/bin/bash
# Fix NumPy compatibility issue for Streamlit

echo "🔧 Fixing NumPy compatibility issue..."
echo ""

echo "Current NumPy version:"
python -c "import numpy; print(f'  {numpy.__version__}')" 2>/dev/null || echo "  NumPy not found"

echo ""
echo "Downgrading NumPy to < 2.0.0 for compatibility..."
pip install "numpy<2.0.0" --upgrade --quiet

echo ""
echo "✅ NumPy fixed!"
echo ""
echo "New NumPy version:"
python -c "import numpy; print(f'  {numpy.__version__}')"

echo ""
echo "Now you can run Streamlit:"
echo "  streamlit run app.py"

