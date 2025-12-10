# Test Commands for Stage 1 Inference

## ✅ Quick Test Commands

### 1. Single Image Test
```bash
python stage1_inference.py \
    --input test_input/AIAnimationStudio_Roughs_005.png \
    --phase rough \
    --target clean \
    --out test_single.png \
    --ckpt epoch_014.pth
```

### 2. Batch Folder Test
```bash
python stage1_inference.py \
    --input test_input/ \
    --phase rough \
    --target clean \
    --out test_output/ \
    --ckpt epoch_014.pth
```

### 3. Using Test Script
```bash
./test_command.sh
```

## 🎨 Different Phase Combinations

### Rough to Clean
```bash
python stage1_inference.py --input test_input/ --phase rough --target clean --out test_output/
```

### Rough to Color
```bash
python stage1_inference.py --input test_input/ --phase rough --target color --out test_output/
```

### Rough to Tiedown
```bash
python stage1_inference.py --input test_input/ --phase rough --target tiedown --out test_output/
```

### Rough to Line
```bash
python stage1_inference.py --input test_input/ --phase rough --target line --out test_output/
```

## 📝 Notes

- **NumPy Warnings**: You may see NumPy compatibility warnings, but these don't affect functionality. The script works correctly.
- **Output**: All processed images are saved to the specified output directory.
- **Checkpoint**: Default is `epoch_014.pth` in current directory.

## ✅ Expected Output

```
Using device: cpu
Loading checkpoint from: epoch_014.pth
Model loaded successfully!
[OK] AIAnimationStudio_Roughs_005.png
[OK] AIAnimationStudio_Roughs_006.png
[OK] AIAnimationStudio_Roughs_007.png
[OK] AIAnimationStudio_Roughs_008.png
Done.
```

## 🔧 Troubleshooting

**If you see import errors:**
```bash
pip install -r requirements.txt
```

**If checkpoint not found:**
- Make sure `epoch_014.pth` is in the project directory
- Or specify full path: `--ckpt /path/to/checkpoint.pth`

**If NumPy warnings appear:**
- These are harmless - the script still works
- To fix: `pip install "numpy<2.0.0"`

