# TrOCR Khmer OCR Model

This repository contains a trained TrOCR (Transformer-based OCR) model for recognizing Khmer text from images. The model is fine-tuned to work specifically with Khmer text documents and images, enabling accurate OCR functionality.

## Features
- Supports Khmer text recognition from PNG images.
- Efficiently processes and predicts text using a trained VisionEncoderDecoderModel.
- Integrated with a Streamlit-based web interface for ease of use.

---

## Getting Started

### Prerequisites
To use this model, ensure the following dependencies are installed:

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers (Hugging Face)
- Streamlit
- Pillow
- Requests

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Folder Structure
Ensure the repository has the following structure:
```
project-directory/
├── 1m_final_model/           # Directory containing the model files
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── selected_images/          # Directory for preloaded images
├── inference.py              # Main Streamlit application file
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## How to Use the Model

### Running the Application Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repository>/trocr-khmer-ocr.git
   cd trocr-khmer-ocr
   ```

2. Ensure the `1m_final_model` directory contains all necessary model files. If not, the application will automatically download them from the configured Azure Blob Storage URLs.

3. Run the application using Streamlit:
   ```bash
   streamlit run inference.py --server.port 8501
   ```

4. Access the web interface at:
   ```
   http://localhost:8501
   ```

### Uploading or Selecting Images
- **Upload a PNG Image**: Use the file uploader to upload a Khmer text image.
- **Select from Preloaded Images**: Choose an image from the `selected_images` folder.

The OCR prediction will be displayed in the interface.

---

## Deployment

### Deploying to Azure App Service
1. Ensure your repository is prepared with a `startup.txt` file containing the startup command:
   ```bash
   streamlit run inference.py --server.port $PORT --server.address 0.0.0.0
   ```

2. Deploy the app using one of the following methods:
   - **Git Deployment**:
     ```bash
     git push azure master
     ```
   - **ZIP Deployment**:
     ```bash
     zip -r app.zip .
     az webapp deploy --resource-group <resource-group-name> --name <app-name> --src-path app.zip
     ```

3. Ensure the Azure App Service is configured with the correct Python version and dependencies.

---

## Optimizations

### Improving Performance
- **Caching**: The model and processor are cached using Streamlit's caching mechanism for faster inference.
- **GPU Acceleration**: The app leverages GPU (if available) for faster predictions.
- **Preprocessing**: Images are resized to optimize model input and reduce latency.

### Future Enhancements
- Model quantization to reduce size and improve speed.
- Support for additional image formats and batch processing.
- Fine-tuning the model for other Southeast Asian languages.

---

## Technical Details

### Model Architecture
This model is based on the TrOCR architecture provided by Hugging Face. It combines a Vision Transformer (ViT) encoder with an autoregressive GPT-2 decoder for end-to-end text recognition.

### Training
The model was fine-tuned on a dataset of Khmer text images to achieve optimal performance for Khmer OCR tasks.

### Inference Workflow
1. Input images are preprocessed and converted into pixel values.
2. The pixel values are passed through the VisionEncoderDecoderModel.
3. Predicted text is decoded and displayed in the interface.

---

## Known Issues
- Performance may degrade on large or high-resolution images.
- The app relies on internet connectivity for downloading model files if they are missing locally.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- [Hugging Face](https://huggingface.co) for providing the TrOCR architecture.
- [Azure](https://azure.microsoft.com) for hosting the model and application.
