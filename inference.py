import os
import requests
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
import streamlit as st
from PIL import Image

# Cache the model and processor to load them only once
@st.cache_resource
def load_model_and_processor(model_name):
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = TrOCRProcessor.from_pretrained(model_name)
    return model, processor

model_name = "Wasanthak/trocr-khmer-printed"
model, processor = load_model_and_processor(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optionally, use half-precision floats for faster computation if supported
if device.type == "cuda":
    model = model.half()

model.eval()

# Streamlit app
st.title("Enadoc Khmer OCR using modified TrOCR model!")
st.write(
    "Upload a PNG file, or select a file from the `selected_images` folder for prediction. "
    "Mark predictions as correct or incorrect to log results for further analysis."
)

# Directory for preloaded images
selected_images_dir = "selected_images"
if not os.path.exists(selected_images_dir):
    os.makedirs(selected_images_dir)

# Dropdown for preloaded images
image_files = [f for f in os.listdir(selected_images_dir) if f.lower().endswith(".png")]
image_choice = st.selectbox("Select an image from the `selected_images` folder", ["None"] + image_files)

# File uploader
uploaded_file = st.file_uploader("Or upload a PNG file", type=["png"])

# Process the selected or uploaded image
if uploaded_file is not None or image_choice != "None":
    try:
        # Load the image (uploaded file has higher priority)
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            source_name = uploaded_file.name
        else:
            image_path = os.path.join(selected_images_dir, image_choice)
            image = Image.open(image_path).convert("RGB")
            source_name = image_choice

        # Display the selected/uploaded image
        st.image(image, caption="Selected Image", use_container_width=True)

        # Preprocess the image for the model
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_start_token_id=model.config.decoder_start_token_id,
                max_length=80,  # Adjust as needed
            )

        # Decode the prediction
        predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Display the result
        st.markdown(
            f"<h2 style='font-size:300%; color:green;'>Prediction:</h2><p style='font-size:300%; color:#00FF00;'>{predicted_text}</p>",
            unsafe_allow_html=True,
        )

        # Add thumbs up and thumbs down buttons
        col1, col2 = st.columns(2)

        correct_clicked = col1.button("👍 Correct")
        incorrect_clicked = col2.button("👎 Incorrect")

        if correct_clicked and not incorrect_clicked:
            # Log correct predictions
            log_file = "correct_log.txt"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Correct Prediction:\n")
                f.write(f"Image: {source_name}\n")
                f.write(f"Prediction: {predicted_text}\n")
                f.write("-" * 50 + "\n")
            st.success(f"Marked as correct and logged to {log_file}")

        elif incorrect_clicked and not correct_clicked:
            # Log incorrect predictions
            log_file = "error_log.txt"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Incorrect Prediction:\n")
                f.write(f"Image: {source_name}\n")
                f.write(f"Prediction: {predicted_text}\n")
                f.write("-" * 50 + "\n")
            st.success(f"Marked as incorrect and logged to {log_file}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please select an image from the dropdown or upload a PNG file.")