# 🎨 Streamlit Web Interface Guide

## Quick Start

### Option 1: Using Script (Recommended)
```bash
./run_streamlit.sh
```

### Option 2: Direct Command
```bash
streamlit run app.py
```

### Option 3: Custom Port
```bash
streamlit run app.py --server.port 8502
```

## Features

### ✨ Main Features
- **Drag & Drop Upload** - Easy image upload interface
- **Phase Selection** - Choose input and target phases from dropdown
- **Real-time Processing** - See results instantly
- **Download Results** - Save processed images
- **Side-by-side Comparison** - Compare original vs processed
- **GPU Support** - Automatic GPU detection and usage

### 🎯 Supported Phases
- `rough` - Initial rough sketches
- `tiedown` - Refined tie-down drawings
- `line` - Clean line art
- `clean` - Final cleaned artwork
- `color` - Colored artwork
- `skeleton` - Skeleton/pose reference

### ⚙️ Configuration Options

**In Sidebar:**
- **Checkpoint Selection** - Choose model checkpoint
- **Input Phase** - Current phase of uploaded image
- **Target Phase** - Desired output phase
- **Advanced Options:**
  - Strong Preprocessing - For scanned roughs
  - GPU Toggle - Enable/disable GPU usage

## How to Use

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload Image:**
   - Click "Browse files" or drag & drop
   - Supported formats: PNG, JPG, JPEG

3. **Configure Settings:**
   - Select input phase (e.g., "rough")
   - Select target phase (e.g., "clean")
   - Adjust advanced options if needed

4. **Process:**
   - Click "🚀 Process Image" button
   - Wait for processing (usually 2-5 seconds)

5. **Download:**
   - View result in right column
   - Click "💾 Download Result" to save

## UI Layout

```
┌─────────────────────────────────────────┐
│  🎨 Stage 1 Animation Cleanup          │
├──────────────┬────────────────────────┤
│              │                          │
│  Sidebar     │   Main Content          │
│  - Config    │   - Upload Area         │
│  - Phases    │   - Output Area         │
│  - Options   │   - Download Button    │
│              │                          │
└──────────────┴─────────────────────────┘
```

## Troubleshooting

### App Won't Start
```bash
# Check if streamlit is installed
pip install streamlit

# Or install all requirements
pip install -r requirements.txt
```

### Model Not Loading
- Ensure `epoch_014.pth` is in the project directory
- Or specify full path in sidebar

### NumPy Warnings
- These are harmless and don't affect functionality
- To fix: `pip install "numpy<2.0.0"`

### GPU Not Working
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- App will automatically fall back to CPU

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Production Deployment
```bash
streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect to share.streamlit.io
3. Deploy!

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Tips

- **Batch Processing**: Use CLI for multiple images
- **Best Results**: Use high-quality input images (512x512 or larger)
- **Performance**: GPU is 5-10x faster than CPU
- **File Size**: Large images may take longer to process

## Support

For issues or questions, check:
- `README.md` - Full documentation
- `TEST_COMMANDS.md` - CLI testing guide
- `DEPLOYMENT_SUMMARY.md` - Deployment info

