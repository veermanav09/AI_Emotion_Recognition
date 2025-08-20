import streamlit as st
import torch
from models.vision_model import VisionBackbone
from models.audio_model import AudioBackbone
from models.text_model import TextBackbone
from models.fusion_model import FusionClassifier

DEVICE = "cpu"

# load models
vnet, anet, tnet, clf = VisionBackbone(), AudioBackbone(), TextBackbone(), FusionClassifier()
ckpt = torch.load("checkpoint.pth", map_location=DEVICE)
vnet.load_state_dict(ckpt["vnet"]); anet.load_state_dict(ckpt["anet"])
tnet.load_state_dict(ckpt["tnet"]); clf.load_state_dict(ckpt["clf"])

vnet.eval(); anet.eval(); tnet.eval(); clf.eval()

st.title("AI Emotion Recognition for Tailored Ads")

uploaded = st.file_uploader("Upload an image (dummy only for now)", type=["png", "jpg", "jpeg"])
if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    st.write("👉 In a real setup, we’d extract features & run inference")
    st.success("Predicted Emotion: Happy 🎉")
    st.info("Recommended Ad: Travel Package 🌍")
