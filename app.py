"""
Deployment App — Deteksi Penyakit Daun (Plant Pathology)
Model: Custom EfficientNetB0 (PyTorch, Multi-label Classification)
Kelas: healthy, multiple_diseases, rust, scab
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

# ============================================================
# 1. KONFIGURASI
# ============================================================
MODEL_PATH = "best_model.pt"          # <-- sesuaikan nama/path file model kamu
LABEL2CAT = ["healthy", "multiple_diseases", "rust", "scab"]  # urutan HARUS sama persis seperti saat training
IMG_SIZE = 224
THRESHOLD = 0.5

st.set_page_config(page_title="Deteksi Penyakit Daun", page_icon="🌿")


# ============================================================
# 2. DEFINISI ARSITEKTUR MODEL
# (harus identik dengan class CustomEfficientNetB0 di notebook training)
# ============================================================
class CustomEfficientNetB0(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.model = efficientnet_b0(weights=None)  # weights=None karena kita load state_dict sendiri
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# 3. LOAD MODEL (cached supaya tidak reload setiap interaksi)
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CustomEfficientNetB0(output_size=len(LABEL2CAT))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# 4. PREPROCESSING (harus sama persis dengan test_transform di training)
# ============================================================
transform = transforms.Compose([
    transforms.Resize(230),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def predict(model, image: Image.Image):
    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor).squeeze().tolist()
    if isinstance(output, float):  # jaga-jaga kalau cuma 1 kelas
        output = [output]
    return output


# ============================================================
# 5. UI STREAMLIT
# ============================================================
st.title("🌿 Deteksi Penyakit Daun")
st.write(
    "Upload gambar daun untuk mendeteksi kondisi: **healthy**, "
    "**multiple diseases**, **rust**, atau **scab**."
)
st.caption("Model: Custom EfficientNetB0 — Multi-label classification")

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

    with col2:
        with st.spinner("Menganalisis gambar..."):
            probs = predict(model, image)

        st.subheader("Hasil Prediksi")

        detected = [label for label, p in zip(LABEL2CAT, probs) if p > THRESHOLD]

        if detected:
            st.success(f"**Terdeteksi:** {', '.join(detected)}")
        else:
            st.warning("Tidak ada kelas yang melewati threshold confidence.")

        st.write("---")
        st.write("**Detail Probabilitas tiap Kelas:**")
        for label, p in zip(LABEL2CAT, probs):
            st.write(f"{label}")
            st.progress(min(max(p, 0.0), 1.0))
            st.caption(f"{p:.2%}")

else:
    st.info("Silakan upload gambar daun (.jpg / .jpeg / .png) untuk memulai.")

st.write("---")
st.caption(
    "⚠️ Catatan: model ini bersifat multi-label — satu gambar bisa terdeteksi "
    "lebih dari satu kondisi sekaligus."
)