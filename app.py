import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch

# Load YOLO model
yolo_model = YOLO("best.pt")

# Function untuk deteksi pakai YOLOv8
def detect_objects(image):
    results = yolo_model.predict(image, save=False)
    result_img = results[0].plot()
    return result_img

# Function untuk super-resolution pakai Real-ESRGAN (manual RRDBNet)
def super_resolve(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inisialisasi model dengan parameter yang sesuai
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = 'realesrgan/weights/RealESRGAN_x4plus.pth'

    # Load checkpoint dengan strict=False untuk mengatasi masalah key mismatch
    checkpoint = torch.load(model_path, map_location=device)
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model = model.to(device)

    # Konversi gambar ke tensor
    img_np = np.array(image)[:, :, ::-1]
    img_np = img_np / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Inferensi
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Konversi hasil ke gambar
    output_tensor = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_tensor = (output_tensor * 255.0).clip(0, 255).astype('uint8')
    output_pil = Image.fromarray(output_tensor[:, :, ::-1])
    return output_pil

# Streamlit UI
st.title("🛸 UAV Detection Web App")
st.write("Dengan 2 Pipeline: Deteksi Langsung dan Deteksi setelah Super-Resolution")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Asli", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Deteksi Langsung")
        result_img_direct = detect_objects(image)
        st.image(result_img_direct, caption="Hasil Deteksi Langsung", use_container_width=True)

    with col2:
        st.subheader("✨ Super-Resolution + Deteksi")
        with st.spinner("Proses Super-Resolution..."):
            sr_image = super_resolve(image)
        st.image(sr_image, caption="Setelah Super-Resolution", use_container_width=True)

        result_img_sr = detect_objects(sr_image)
        st.image(result_img_sr, caption="Hasil Deteksi Setelah SR", use_container_width=True)
