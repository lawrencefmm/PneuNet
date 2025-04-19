# PneuNet: Pneumonia Classification from Chest X-Rays

PneuNet is a deep learning system for detecting and classifying pneumonia types from chest X-ray images using convolutional neural networks. This project addresses a critical need for automated analysis of medical imaging to support clinical decision-making.

## The Problem: Pneumonia Diagnosis Challenges
Pneumonia causes over 2.5 million deaths annually worldwide. Key challenges include:
- Distinguishing between bacterial and viral pneumonia (critical for treatment decisions)
- High inter-class similarity in X-ray manifestations
- Time-consuming manual analysis by radiologists
- Limited access to expert diagnostics in developing regions

PneuNet automates the classification process with three distinct categories:
1. 🦠 **Bacterial Pneumonia**
2. 🩺 **Normal** 
3. 🦠 **Viral Pneumonia** 

<!-- Example X-rays for each class -->

### Example X-rays for Each Class

Below are representative X-ray images for each classification category:

1. **Bacterial Pneumonia**  
![Bacterial Pneumonia X-ray](assets/person1_bacteria_1.jpeg)

2. **Normal**  
![Normal X-ray](assets/IM-0001-0001.jpeg)

3. **Viral Pneumonia**  
![Viral Pneumonia X-ray](assets/person16_virus_47.jpeg)


<!--
Bacterial: assets/bacterial_example.png
Normal: assets/normal_example.png
Viral: assets/viral_example.png
-->

## Dataset Preparation
The model uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, processed into three classes:

| Class               | Samples |
|---------------------|---------|
| Bacterial Pneumonia | 2780   | 
| Viral Pneumonia      | 1493   | 
| Normal              | 1,583   | 

## Image Preprocessing Pipeline
Our preprocessing ensures optimal input for the neural network:

1. **Grayscale Conversion**  
   `transforms.Grayscale()` - Convert to single-channel images

2. **Quality Control**  
   - Remove images <256px in any dimension
   - Verify file integrity through hash checks

3. **Resizing**  
   `transforms.Resize((256, 256))` using lanczos resampling:
   ```python
   image = image.resize((img_dimension, img_dimension), Image.LANCZOS)
    ```

## Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone git@github.com:lawrencefmm/PneuNet.git
   cd PneuNet
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Download the Kaggle Dataset

1. **Install Kaggle CLI**  
   ```bash
   pip install kaggle
   ```

2. **Get your Kaggle API token**  
   - Go to your Kaggle account settings and create a new API token.
   - Place the downloaded `kaggle.json` in `~/.kaggle/`.

3. **Download the dataset**  
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d PneuNet/
   ```

## Preprocess the Data

Run the preprocessing script to prepare the dataset:
```bash
python src/preprocess.py
```

## Train the Model

Train the model using the provided training script:
```bash
python src/train_model.py
```

## Predict on a New Image

Use the `predict.py` script to predict the class of a chest X-ray image.  
Pass the image path and model checkpoint path as arguments:

```bash
python src/predict.py --image path/to/image.jpeg --model_path path/to/model.ckpt
```

- `--image`: Path to the image you want to classify.
- `--model_path`: Path to the trained model checkpoint (usually found in the `lightning_logs` directory, e.g., `lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt`).

The script will output the predicted class for the image.

---

Adjust paths and filenames as needed for your setup.