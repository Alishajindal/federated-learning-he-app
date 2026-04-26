import streamlit as st
from PIL import Image

from predict import load_model, predict_image

st.set_page_config(page_title="BloodMNIST App", layout="centered")

# Labels
LABELS = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature Granulocyte",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet"
}

# Load model
@st.cache_resource
def get_model():
    model, device = load_model()
    return model, device

model, device = get_model()

# UI
st.title("🩸 Blood Cell Classification")

uploaded_file = st.file_uploader("Upload a blood cell image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        label, confidence = predict_image(model, device, image)

        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("Please upload an image to start.")