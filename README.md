# PneuNet

PneuNet is a deep learning project for detecting pneumonia from chest X-ray images using convolutional neural networks. It provides scripts for data preprocessing, model training, and prediction on new images.

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