import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# ---- Load your trained model here ----
model = tf.keras.models.load_model("model/model.h5")

st.title("🖌️ MNIST Digit Recognizer")

st.write("Draw a digit (0–9) below and click **Predict**")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=15,
    stroke_color="#FFFFFF",  # White drawing (important!)
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Extract image
        img = canvas_result.image_data
        img = Image.fromarray(np.uint8(img))

        # Convert to grayscale, resize to 28x28 (MNIST format)
        img = img.convert("L").resize((28, 28))

        # Normalize
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 784)

        # Predict
        preds = model.predict(img_arr)[0]
        top = np.argmax(preds)

        st.write(f"## 🧠 Model prediction: {top}")
        st.write("### Probability scores")

        # Create dataframe
        df = pd.DataFrame({
            "Digit": list(range(10)),
            "Probability": preds
        })

        # Optional: round values
        df["Probability"] = df["Probability"].round(4)

        st.dataframe(df,use_container_width=True,hide_index=True)

        st.image(img, width=140, caption="Processed 28×28 image")
    else:
        st.write("❗ Draw something first!")
