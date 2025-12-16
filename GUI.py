# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# ===============================
# Load the exported model
# ===============================
MODEL_PATH = "handwritten_digit_cnn_savedmodel"
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# ===============================
# Streamlit Page Config & Styling
# ===============================
st.set_page_config(
    page_title="Minimal Futuristic Digit Recognizer",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body {
    background-color: #0f1115;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2 {
    color: #00d4ff;
    font-weight: 600;
}
div.stButton > button {
    background-color: #00d4ff;
    color: #0f1115;
    border-radius: 8px;
    height: 40px;
    width: 100%;
    font-weight: 600;
    transition: 0.2s;
}
div.stButton > button:hover {
    background-color: #00a1cc;
    color: #ffffff;
}
.stFileUploader > label {
    color: #00d4ff;
    font-weight: 500;
}
.stMarkdown p {
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# App Title
# ===============================
st.title("🤖 Handwritten Digit Recognizer")

# ===============================
# Prediction Function
# ===============================
def predict_image(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1,28,28,1)
    input_tensor = tf.convert_to_tensor(img_array)
    output = infer(input_tensor)
    pred_probs = list(output.values())[0].numpy()[0]
    digit = int(np.argmax(pred_probs))
    return digit, pred_probs

# ===============================
# Function to Plot Confidence using Matplotlib
# ===============================
def plot_confidence_matplotlib(probs, predicted_digit):
    digits = list(range(10))
    colors = ['#00d4ff' if i == predicted_digit else '#888888' for i in digits]
    plt.figure(figsize=(8,4))
    plt.bar(digits, probs, color=colors)
    plt.xticks(digits)
    plt.ylabel("Confidence")
    plt.xlabel("Digit")
    plt.ylim(0,1)
    plt.title("Prediction Confidence")
    for i, v in enumerate(probs):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', color='white', fontweight='bold')
    st.pyplot(plt)
    plt.clf()

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3 = st.tabs(["Draw Digit", "Upload Image", "About"])

# -------------------------------
# Canvas Tab
# -------------------------------
with tab1:
    st.subheader("Draw a Digit")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        if st.button("Predict from Canvas"):
            digit, probs = predict_image(img)
            st.success(f"Predicted Digit: {digit}")
            st.image(img, caption="Your Drawing", width=140)
            st.subheader("Confidence Levels")
            plot_confidence_matplotlib(probs, digit)

# -------------------------------
# Upload Tab
# -------------------------------
with tab2:
    st.subheader("Upload a Digit Image")
    uploaded_file = st.file_uploader("Choose a digit image...", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if st.button("Predict from Upload"):
            digit, probs = predict_image(img)
            st.success(f"Predicted Digit: {digit}")
            st.image(img, caption="Uploaded Image", width=140)
            st.subheader("Confidence Levels")
            plot_confidence_matplotlib(probs, digit)

# -------------------------------
# About Tab
# -------------------------------
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    **Project Overview:**  
    This project is a **Handwritten Digit Recognizer** built using a **Convolutional Neural Network (CNN)**. It can identify digits from hand-drawn inputs or uploaded images. The goal is to create a simple, intuitive interface for digit recognition.

    **GUI (Graphical User Interface):**  
    - Built using **Streamlit** for a minimalistic and futuristic design.  
    - Features two main functionalities:  
        1. **Draw Digit:** Draw a digit on a canvas and get predictions.  
        2. **Upload Image:** Upload an image of a digit and get predictions.  
    - A separate **About tab** provides details about the project and model.

    **CNN Model:**  
    - The model is trained on the **MNIST dataset** for handwritten digit recognition.  
    - It consists of convolutional layers, pooling layers, and fully connected layers to achieve robust recognition.  
    - The trained model is exported and loaded in **SavedModel format**, allowing inference in this app.

    **Usage:**  
    1. Select a tab (Draw or Upload).  
    2. Provide input (draw or upload).  
    3. Click the predict button to see the recognized digit along with a **highlighted confidence bar chart** for all digits.
    """)
