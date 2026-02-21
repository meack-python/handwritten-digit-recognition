import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")


@st.cache(allow_output_mutation=True)
def load_trained_model(path="model/model.h5"):
    return load_model(path)


def _otsu_threshold(arr):
    hist = np.bincount(arr.flatten(), minlength=256)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def preprocess_canvas_image(image_data, canvas_size=(280, 280)):
    if image_data is None:
        return None

    img = image_data.astype(np.uint8)

    # Composite RGBA over white background if needed
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        comp = (rgb * alpha[:, :, None]) + (255.0 * (1 - alpha[:, :, None]))
        img_rgb = comp.astype(np.uint8)
    else:
        img_rgb = img[:, :, :3]

    pil = Image.fromarray(img_rgb)
    gray_pil = pil.convert("L")

    arr = np.array(gray_pil)

    # If background is light and strokes dark, invert so digit is white on black
    if arr.mean() > 127:
        arr = 255 - arr

    # Binarize using Otsu
    thresh = _otsu_threshold(arr)
    bw = (arr > thresh).astype(np.uint8) * 255

    # Find bounding box of the digit
    coords = np.argwhere(bw > 0)
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        margin = 4
        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, bw.shape[1] - 1)
        y1 = min(y1 + margin, bw.shape[0] - 1)
        digit = bw[y0 : y1 + 1, x0 : x1 + 1]
    else:
        digit = bw

    # Convert to PIL image for resizing
    digit_pil = Image.fromarray(digit)

    # Resize to fit in 20x20 while preserving aspect ratio
    w, h = digit_pil.size
    if w == 0 or h == 0:
        resized = Image.new("L", (28, 28), color=0)
    else:
        if h > w:
            new_h = 20
            new_w = max(1, int(round((w * 20) / h)))
        else:
            new_w = 20
            new_h = max(1, int(round((h * 20) / w)))
        resized = digit_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    # Create 28x28 image and paste centered
    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - resized.size[0]) // 2
    paste_y = (28 - resized.size[1]) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # Normalize to [0,1]
    norm = (np.array(canvas).astype(np.float32) / 255.0).reshape(1, 784)
    return norm


def predict_and_display(model, input_tensor):
    if input_tensor is None:
        st.warning("Draw a digit on the canvas first.")
        return

    preds = model.predict(input_tensor)[0]
    digits = list(range(10))
    df = pd.DataFrame({"Digit": digits, "Probability": preds})
    df_sorted = df.sort_values("Probability", ascending=False).reset_index(drop=True)

    st.subheader("Predicted probabilities")
    # Format probabilities as percentages
    df_sorted["Probability"] = (df_sorted["Probability"] * 100).round(2).astype(str) + "%"
    st.table(df_sorted)

    top_digit = int(df_sorted.loc[0, "Digit"])
    st.success(f"Top prediction: {top_digit}")


def main():
    st.title("Handwritten Digit Recognition")
    st.write("Draw a digit (0-9) in the box below, then click Predict.")

    # Sidebar
    
    # ================= SIDEBAR =================
    st.sidebar.markdown("## 🎛️ Drawing Options")

    stroke_width = st.sidebar.slider("Stroke width", 1, 50, 12)
    stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
    bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")

    st.sidebar.markdown("---")

    st.sidebar.markdown("## 📌 App Information")

    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        **Handwritten Digit Recognition App**

        This app uses a trained **MNIST neural network model**
        to recognize digits from 0 to 9.

        ### 🧠 How It Works
        - Draw a digit in the canvas.
        - Click **Predict**.
        - The image is preprocessed (cropped, centered, resized).
        - The model predicts the digit.
        - Probability scores are displayed.

        ### ✍️ Tips for Better Accuracy
        - Draw clearly and centered.
        - Avoid very small digits.
        - Use thicker strokes.
        - Keep the background clean.
        """)
    with st.sidebar.expander("📬 Source Code"):
        st.markdown("""  
 
        🌐 [GitHub Repository](https://github.com/meack-python/handwritten-digit-recognition/blob/main/MNIST.ipynb)
        """)
    with st.sidebar.expander("📬 Contact Author"):
        st.markdown("""
        👨‍💻 **Meack Python**  

        👍 [Facebook](https://www.facebook.com/profile.php?id=61586847368051)  
        🌐 [GitHub](https://github.com/meack-python)  
        📺 [YouTube](https://www.youtube.com/@meack-python)

        Feel free to reach out for collaboration or improvements!
        """)

    st.sidebar.markdown("---")

    # Styled footer
    st.sidebar.markdown(
        """
        <div style='text-align:center; font-size:12px; color:gray;'>
        2026 Handwritten Digit Recognizer <br>
        Built with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict"):
            model = load_trained_model()
            input_tensor = preprocess_canvas_image(canvas_result.image_data)
            predict_and_display(model, input_tensor)

    with col2:
        if st.button("Clear"):
            # Rerun clears the canvas because of key use
            st.experimental_rerun()

    st.markdown("---")
    st.write("Model source: `model/model.h5`")


if __name__ == "__main__":
    main()
