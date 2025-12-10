# Stage 1 Animation Cleanup System

AI-powered animation phase enhancement using conditional GANs. This system can transform animation frames from one phase (e.g., rough) to another (e.g., clean).

## Features

- 🎨 Multi-phase animation enhancement (rough → clean, tiedown → color, etc.)
- 🖥️ Command-line interface for batch processing
- 🌐 Streamlit web interface for interactive use
- 🚀 Ready for deployment

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd stage1_cleanup
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure you have a trained model checkpoint:**
   - Place your `.pth` checkpoint file in the project directory
   - Default expected: `epoch_014.pth`

## Usage

### Command Line Interface (CLI)

Process a single image:
```bash
python stage1_inference.py \
    --input path/to/input.png \
    --phase rough \
    --target clean \
    --out path/to/output.png \
    --ckpt epoch_014.pth
```

Process a folder of images:
```bash
python stage1_inference.py \
    --input path/to/input_folder/ \
    --phase rough \
    --target clean \
    --out path/to/output_folder/ \
    --ckpt epoch_014.pth
```

**Arguments:**
- `--input`: Input image file or folder path (required)
- `--phase`: Input phase - one of: `rough`, `tiedown`, `line`, `clean`, `color`, `skeleton` (required)
- `--target`: Target phase - one of: `rough`, `tiedown`, `line`, `clean`, `color`, `skeleton` (default: `clean`)
- `--out`: Output path for processed image(s) (required)
- `--ckpt`: Path to model checkpoint file (default: `epoch_014.pth`)

### Streamlit Web Interface

Launch the web interface:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload images via drag-and-drop
- Select input and target phases
- Real-time processing
- Download results
- GPU acceleration (if available)

## Project Structure

```
stage1_cleanup/
├── app.py                      # Streamlit web interface
├── stage1_inference.py         # CLI inference script
├── models.py                   # Shared model definitions
├── train_stage1.py             # Training script
├── utils/
│   ├── __init__.py
│   ├── io_utils.py             # I/O utilities
│   └── preprocess_utils.py     # Preprocessing functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── epoch_014.pth               # Model checkpoint (your file)
```

## Supported Phase Transitions

The system supports the following animation phases:
- `rough` - Initial rough sketches
- `tiedown` - Refined tie-down drawings
- `line` - Clean line art
- `clean` - Final cleaned artwork
- `color` - Colored artwork
- `skeleton` - Skeleton/pose reference

You can transform between any of these phases (e.g., `rough → clean`, `tiedown → color`).

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: Recommended (CUDA-compatible) for faster processing
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~2GB for dependencies

## Troubleshooting

### Model not loading
- Ensure checkpoint file exists and path is correct
- Check that checkpoint contains `G` key (generator weights)

### CUDA out of memory
- Reduce batch size (CLI processes one image at a time)
- Use CPU mode: set `use_gpu=False` in Streamlit

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory

## Deployment

### Local Deployment
1. Install dependencies
2. Run Streamlit: `streamlit run app.py`
3. Access at `http://localhost:8501`

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set Python version to 3.8+
4. Deploy!

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## License

[Your License Here]

## Support

For issues or questions, please contact the development team.

