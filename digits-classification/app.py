import numpy as np
import streamlit as st
from PIL import Image
from sklearn.datasets import load_digits

from main import load_or_train_models

st.set_page_config(page_title="Digits ANN vs CNN", layout="wide")


@st.cache_resource
def get_models():
    return load_or_train_models()


def image_to_digit(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    arr = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    return arr / 255.0 * 16.0


def predict_vector(ann, cnn, scaler, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = scaler.transform(vector.reshape(1, -1))
    ann_proba = ann.predict(scaled, verbose=0)[0]
    cnn_proba = cnn.predict(scaled.reshape(1, 8, 8, 1), verbose=0)[0]
    return ann_proba, cnn_proba


def show_result(title: str, proba: np.ndarray, test_acc: float) -> None:
    pred = int(proba.argmax())
    st.subheader(title)
    st.metric("Tahmin", str(pred), f"güven %{proba[pred] * 100:.1f}")
    st.caption(f"Test accuracy: {test_acc:.2%}")
    st.bar_chart({str(i): float(p) for i, p in enumerate(proba)})


ann, cnn, scaler, ann_acc, cnn_acc = get_models()
digits = load_digits()

st.title("Rakam sınıflandırma")
st.caption("Görsel yükle → ANN ve CNN ayrı ayrı sonuç üretir")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = None

uploaded = st.file_uploader("Rakam görseli yükle", type=["png", "jpg", "jpeg"])

st.write("veya veri setinden örnek seç")
sample_cols = st.columns(10)
for digit, col in enumerate(sample_cols):
    idx = int(np.where(digits.target == digit)[0][0])
    if col.button(str(digit), use_container_width=True, key=f"sample_{digit}"):
        st.session_state.sample_idx = idx

vector = None
preview = None
source_image = None

if uploaded is not None:
    st.session_state.sample_idx = None
    source_image = Image.open(uploaded)
    preview = image_to_digit(source_image)
    vector = preview.reshape(64)
elif st.session_state.sample_idx is not None:
    preview = digits.images[st.session_state.sample_idx]
    vector = digits.data[st.session_state.sample_idx]
    source_image = Image.fromarray((preview / 16.0 * 255.0).astype(np.uint8)).resize(
        (128, 128), Image.Resampling.NEAREST
    )

if vector is None:
    st.info("Bir görsel yükle veya örnek rakama bas.")
    st.stop()

ann_proba, cnn_proba = predict_vector(ann, cnn, scaler, vector)

top = st.columns(2)
with top[0]:
    st.image(source_image, caption="Yüklenen / seçilen görsel", width=180)
with top[1]:
    st.image(
        Image.fromarray((preview / 16.0 * 255.0).astype(np.uint8)).resize(
            (128, 128), Image.Resampling.NEAREST
        ),
        caption="Modele giden 8×8",
        width=180,
    )

left, right = st.columns(2)
with left:
    show_result("ANN", ann_proba, ann_acc)
with right:
    show_result("CNN", cnn_proba, cnn_acc)
