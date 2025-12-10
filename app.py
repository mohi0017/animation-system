"""
Streamlit Web Interface for Stage 1 Animation Cleanup
"""

import os
import sys
import tempfile
import warnings

# Suppress harmless warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io

# Suppress PyTorch internal messages
import logging
logging.getLogger("torch").setLevel(logging.ERROR)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PhaseEmbedder, UNetGenerator
from utils.io_utils import save_tensor_as_png
from stage1_inference import preprocess_for_model, run_inference_single, PHASES

# Page config
st.set_page_config(
    page_title="Stage 1 Animation Cleanup",
    page_icon="🎨",
    layout="wide"
)

# Title and Header
st.title("🎨 Stage 1 Animation Cleanup")
st.markdown("### AI-Powered Animation Phase Enhancement")
st.markdown("Transform your animation frames from one phase to another using advanced conditional GANs")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model loading method
    model_source = st.radio(
        "Model Source",
        ["Local File", "Download from URL"],
        index=0,
        help="Choose how to load the model"
    )
    
    checkpoint_path = None
    
    if model_source == "Local File":
        # Checkpoint selection from local files
        checkpoint_dir = os.path.dirname(__file__)
        try:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            
            if checkpoint_files:
                default_ckpt = "epoch_014.pth" if "epoch_014.pth" in checkpoint_files else checkpoint_files[0]
                selected_ckpt = st.selectbox(
                    "Select Checkpoint",
                    checkpoint_files,
                    index=checkpoint_files.index(default_ckpt) if default_ckpt in checkpoint_files else 0
                )
                checkpoint_path = os.path.join(checkpoint_dir, selected_ckpt)
            else:
                checkpoint_path = st.text_input(
                    "Checkpoint Path",
                    value=os.path.join(checkpoint_dir, "epoch_014.pth"),
                    help="Path to model checkpoint file"
                )
        except Exception:
            checkpoint_path = st.text_input(
                "Checkpoint Path",
                value="epoch_014.pth",
                help="Path to model checkpoint file"
            )
    else:
        # Download from URL
        model_url = st.text_input(
            "Model URL",
            value="",
            help="Enter URL to download model checkpoint (.pth file)"
        )
        
        if model_url:
            # Download model if not already downloaded
            @st.cache_resource
            def download_model(url, filename="epoch_014.pth"):
                import urllib.request
                checkpoint_dir = os.path.dirname(__file__)
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                
                if not os.path.exists(checkpoint_path):
                    try:
                        with st.spinner("Downloading model... This may take a few minutes."):
                            urllib.request.urlretrieve(url, checkpoint_path)
                        st.success("Model downloaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to download model: {e}")
                        return None
                return checkpoint_path
            
            checkpoint_path = download_model(model_url)
        else:
            st.info("Enter a URL to download the model checkpoint")
    
    # Phase selection
    st.subheader("Phase Selection")
    input_phase = st.selectbox(
        "Input Phase",
        PHASES,
        index=0,
        help="Current phase of the input image"
    )
    
    target_phase = st.selectbox(
        "Target Phase",
        PHASES,
        index=3 if "clean" in PHASES else 0,
        help="Desired output phase"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        strong_preprocessing = st.checkbox(
            "Strong Preprocessing",
            value=False,
            help="Use stronger thresholding for scanned roughs"
        )
        
        use_gpu = st.checkbox(
            "Use GPU (if available)",
            value=torch.cuda.is_available(),
            disabled=not torch.cuda.is_available(),
            help="Use CUDA if available"
        )

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.embedder = None
    st.session_state.device = None
    st.session_state.model_loaded = False

# Load model function
@st.cache_resource
def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        E = 16
        in_ch = 4 + 2 * E
        
        embedder = PhaseEmbedder(PHASES, embed_dim=E).to(device)
        model = UNetGenerator(in_ch=in_ch, out_ch=4).to(device)
        
        # Load weights
        if "G" in ckpt:
            model.load_state_dict(ckpt["G"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        
        model.eval()
        return model, embedder, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Input Image")
    st.markdown("Upload your animation frame (PNG, JPG, or JPEG)")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a rough animation frame",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image")
        
        # Show image info
        st.caption(f"📏 Size: {image.size[0]} × {image.size[1]} px | Format: {image.format}")
        
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_input_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process button
        st.markdown("")  # Spacing
        if st.button("🚀 Process Image", type="primary"):
            if checkpoint_path is None or not checkpoint_path:
                st.error("⚠️ Please configure model checkpoint in sidebar first!")
                st.stop()
            
            device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
            
            # Load model
            with st.spinner("Loading model... This may take 10-20 seconds on first load."):
                model, embedder, device = load_model(checkpoint_path, device)
            
            if model is None:
                st.error("Failed to load model. Please check the checkpoint path.")
            else:
                # Process image
                with st.spinner("Processing image..."):
                    try:
                        # Preprocess
                        x = preprocess_for_model(temp_input_path, size=512, strong=strong_preprocessing)
                        x = x.unsqueeze(0).to(device)
                        B, _, H, W = x.shape
                        
                        # Generate conditioning
                        cond = embedder([input_phase], [target_phase], B, H, W, device)
                        x_cond = torch.cat([x, cond], dim=1)
                        
                        # Inference
                        with torch.no_grad():
                            pred = model(x_cond)[0]
                        
                        # Save to session state
                        st.session_state.prediction = pred.cpu()
                        st.session_state.processed = True
                        
                        st.success("✅ Image processed successfully!")
                        st.balloons()  # Celebration!
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                        st.exception(e)  # Show full error details
                        st.session_state.processed = False
                
                # Clean up temp file
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)

with col2:
    st.subheader("📥 Output Result")
    
    if 'processed' in st.session_state and st.session_state.processed:
        pred = st.session_state.prediction
        
        # Convert tensor to image
        pred_np = pred.numpy()
        if pred_np.shape[0] == 4:
            pred_np = pred_np.transpose(1, 2, 0)
        
        # Denormalize
        rgb = (pred_np[..., :3] + 1.0) * 127.5
        alpha = pred_np[..., 3:4] * 255.0 if pred_np.shape[2] == 4 else None
        
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # Display RGB result
        if alpha is not None:
            alpha = np.clip(alpha, 0, 255).astype(np.uint8)
            rgba = np.concatenate([rgb, alpha], axis=2)
            output_image = Image.fromarray(rgba, 'RGBA')
        else:
            output_image = Image.fromarray(rgb, 'RGB')
        
        st.image(output_image, caption=f"Output: {input_phase} → {target_phase}")
        
        # Show output info
        st.caption(f"📏 Size: {output_image.size[0]} × {output_image.size[1]} px | Format: PNG")
        
        # Download button
        buf = io.BytesIO()
        output_image.save(buf, format='PNG')
        st.download_button(
            label="💾 Download Result",
            data=buf.getvalue(),
            file_name=f"output_{input_phase}_to_{target_phase}.png",
            mime="image/png"
        )
        
        # Side-by-side comparison option
        st.markdown("---")
        if st.checkbox("Show side-by-side comparison"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(image, caption="Original")
            with col_b:
                st.image(output_image, caption="Processed")
    else:
        st.info("👈 Upload an image and click 'Process Image' to see results here")
        st.markdown("""
        ### How to use:
        1. **Upload** an animation frame (rough, tiedown, line, etc.)
        2. **Select** input and target phases in the sidebar
        3. **Click** "Process Image" to transform
        4. **Download** your enhanced result!
        """)

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**🎨 Stage 1 Cleanup**")
with col_f2:
    st.markdown("Multi-phase Conditional GAN")
with col_f3:
    device_status = "🟢 GPU" if torch.cuda.is_available() else "⚪ CPU"
    st.markdown(f"Device: {device_status}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Stage 1 Animation Cleanup System | AI-Powered Animation Enhancement</small>
</div>
""", unsafe_allow_html=True)

