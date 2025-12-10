# Setup Instructions for Streamlit App

## ⚠️ NumPy Compatibility Fix Required

Streamlit requires NumPy < 2.0.0. If you see NumPy errors, run:

```bash
./fix_numpy.sh
```

Or manually:
```bash
pip install "numpy<2.0.0" --upgrade
pip install "protobuf<5" "importlib-metadata<7" --upgrade
```

## Quick Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Fix NumPy (if needed)
```bash
./fix_numpy.sh
```

### Step 3: Run Streamlit
```bash
streamlit run app.py
```

Or use the script:
```bash
./run_streamlit.sh
```

## Troubleshooting

### Issue: NumPy 2.x Error
**Solution:**
```bash
pip install "numpy<2.0.0" --upgrade
```

### Issue: Protobuf Error
**Solution:**
```bash
pip install "protobuf<5" --upgrade
```

### Issue: Import Metadata Error
**Solution:**
```bash
pip install "importlib-metadata<7" --upgrade
```

### Issue: Streamlit Won't Start
**Check:**
1. Streamlit installed: `pip show streamlit`
2. NumPy version: `python -c "import numpy; print(numpy.__version__)"`
3. Should be < 2.0.0

**Fix all at once:**
```bash
pip install "numpy<2.0.0" "protobuf<5" "importlib-metadata<7" --upgrade
```

## Verification

After fixing, verify:
```bash
python -c "import streamlit; print('✅ Streamlit OK')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
```

## Alternative: Virtual Environment

For clean setup:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
./fix_numpy.sh
streamlit run app.py
```

