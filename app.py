"""
🌸 Flower Classification Web App
Using ResNet50 Transfer Learning Model
Premium Edition
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="🌸 Flower AI Classifier",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== Premium CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ffd89b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 60px rgba(240, 147, 251, 0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a0a0a0;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .upload-zone {
        border: 3px dashed rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #f093fb;
        background: rgba(240, 147, 251, 0.05);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        border: 1px solid rgba(240, 147, 251, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
    }
    
    .flower-name-result {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: capitalize;
        margin: 1rem 0;
    }
    
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.3);
    }
    
    .prediction-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    
    .prediction-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(10px);
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .prediction-name {
        color: #ffffff;
        font-weight: 500;
        font-size: 1.1rem;
        text-transform: capitalize;
        flex: 1;
        margin-left: 1rem;
    }
    
    .prediction-percent {
        color: #f093fb;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stats-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .mode-button {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .mode-button:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #f093fb;
    }
    
    .mode-button.active {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.2) 0%, rgba(245, 87, 108, 0.2) 100%);
        border-color: #f093fb;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
    }
    
    .footer {
        text-align: center;
        padding: 3rem 0;
        color: #555;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom file uploader */
    .stFileUploader > div > div {
        background: transparent !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #888;
        font-weight: 500;
        padding: 0.8rem 2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
    }
    
    /* Image container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    .stImage {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Model Loading ====================
@st.cache_resource
def load_model():
    """Load the trained ResNet50 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, "models", "best_model.pt")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    return model, device

@st.cache_data
def load_class_mapping():
    """Load the flower name mapping"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(app_dir, "data", "cat_to_name.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        cat_to_name = json.load(f)
    
    # IMPORTANT: ImageFolder sorts folder names ALPHABETICALLY as strings!
    folder_names = list(cat_to_name.keys())
    sorted_folders = sorted(folder_names)
    
    idx_to_class = {i: cat_to_name[folder] for i, folder in enumerate(sorted_folders)}
    
    return idx_to_class

# ==================== Image Preprocessing ====================
def preprocess_image(image):
    """Preprocess image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0)
    
    return img_tensor

# ==================== Prediction ====================
def predict_flower(image, model, device, idx_to_class, top_k=5):
    """Make prediction on the input image"""
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            flower_name = idx_to_class.get(idx, f"Unknown ({idx})")
            predictions.append({
                'name': flower_name,
                'probability': float(prob)
            })
    
    return predictions

# ==================== Main App ====================
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🌸 Flower AI</h1>
        <p class="hero-subtitle">INTELLIGENT FLOWER RECOGNITION SYSTEM</p>
        <div style="margin-top: 1.5rem;">
            <span class="feature-badge">🧠 ResNet50</span>
            <span class="feature-badge">🎯 102 Species</span>
            <span class="feature-badge">⚡ Real-time</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        model, device = load_model()
        idx_to_class = load_class_mapping()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("📁 Make sure `best_model.pt` and `cat_to_name.json` are in the app directory.")
        model_loaded = False
        return
    
    # Stats Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">102</div>
            <div class="stats-label">Flower Species</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">95%</div>
            <div class="stats-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">ResNet</div>
            <div class="stats-label">Architecture</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{'GPU' if device.type == 'cuda' else 'CPU'}</div>
            <div class="stats-label">Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content
    col_left, col_right = st.columns([1, 1], gap="large")
    
    image = None
    
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Input Image")
        
        # Mode selection using radio instead of tabs for better control
        input_mode = st.radio(
            "Select input method:",
            ["📁 Upload Image", "📷 Use Camera"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if input_mode == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Drag and drop or browse",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Supported: JPG, JPEG, PNG, WEBP",
                label_visibility="collapsed"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
        
        else:  # Camera mode
            st.markdown("""
            <div style="text-align: center; padding: 1rem; color: #888;">
                <p>📷 Click button below to capture</p>
            </div>
            """, unsafe_allow_html=True)
            
            camera_image = st.camera_input(
                "Capture flower image",
                label_visibility="collapsed"
            )
            if camera_image:
                image = Image.open(camera_image)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")
        
        if image is not None:
            with st.spinner("🔮 Analyzing image..."):
                predictions = predict_flower(image, model, device, idx_to_class)
            
            if predictions:
                top_pred = predictions[0]
                
                # Main result
                st.markdown(f"""
                <div class="result-card">
                    <p style="color: #888; font-size: 0.9rem; margin-bottom: 0.5rem;">IDENTIFIED AS</p>
                    <p class="flower-name-result">🌺 {top_pred['name']}</p>
                    <span class="confidence-badge">
                        {top_pred['probability']*100:.1f}% Confidence
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Other predictions
                st.markdown("#### 📊 Other Possibilities")
                
                for i, pred in enumerate(predictions):
                    confidence_percent = pred['probability'] * 100
                    st.markdown(f"""
                    <div class="prediction-item">
                        <div class="rank-badge">#{i+1}</div>
                        <span class="prediction-name">{pred['name']}</span>
                        <span class="prediction-percent">{confidence_percent:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #666;">
                <p style="font-size: 4rem; margin-bottom: 1rem;">🌺</p>
                <p style="font-size: 1.2rem; font-weight: 500;">Waiting for image...</p>
                <p style="font-size: 0.9rem; color: #555;">Upload a flower image or capture from camera</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌸 <strong>Flower AI Classifier</strong></p>
        <p style="font-size: 0.8rem; color: #444;">Powered by PyTorch & ResNet50 Transfer Learning</p>
        <p style="font-size: 0.75rem; color: #333; margin-top: 1rem;">
            Recognizes 102 different flower species with high accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
