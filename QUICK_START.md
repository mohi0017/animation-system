# Quick Start Guide

## CLI Usage (Command Line)

### Single Image Processing
```bash
python stage1_inference.py \
    --input test_cases/case_0000/input.png \
    --phase rough \
    --target clean \
    --out output.png
```

### Batch Processing (Folder)
```bash
python stage1_inference.py \
    --input test_cases/case_0000/ \
    --phase rough \
    --target clean \
    --out results/
```

## Streamlit Web Interface

### Option 1: Direct Command
```bash
streamlit run app.py
```

### Option 2: Using Script (Linux/Mac)
```bash
chmod +x run_streamlit.sh
./run_streamlit.sh
```

Then open browser: `http://localhost:8501`

## Available Phases

- `rough` - Initial rough sketches
- `tiedown` - Refined tie-down drawings  
- `line` - Clean line art
- `clean` - Final cleaned artwork
- `color` - Colored artwork
- `skeleton` - Skeleton/pose reference

## Example Commands

```bash
# Rough to Clean
python stage1_inference.py --input input.png --phase rough --target clean --out clean.png

# Tiedown to Color
python stage1_inference.py --input input.png --phase tiedown --target color --out colored.png

# Line to Clean
python stage1_inference.py --input input.png --phase line --target clean --out output.png
```

## Troubleshooting

**Import Error?**
```bash
pip install -r requirements.txt
```

**Checkpoint not found?**
- Make sure `epoch_014.pth` is in the same directory
- Or specify path: `--ckpt /path/to/checkpoint.pth`

**GPU not working?**
- The script will automatically use CPU if GPU is unavailable
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

