from cnn import PneuNet
from torchvision import transforms
import torch
import argparse
from PIL import Image
from pathlib import Path

def __main__(image_path, model_path):
    model_path = Path(model_path)

    checkpoint = [ckpt for ckpt in model_path.glob("**/*.ckpt")][0]
    print(checkpoint, model_path)
    checkpoint_path = str(checkpoint)

    hparams = [hp for hp in model_path.glob("**/hparams.yaml")][0]
    hparams_path = str(hparams)

    model = PneuNet.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_path,
        map_location=None,
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), 
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    image = image.to(device)
    
    output = model(image)
    _, predicted_class = torch.max(output, 1)

    return output, predicted_class.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using the PneuNet model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("model_path", type=str, help="Path to the model directory.")
    args = parser.parse_args()
    output = __main__(args.image_path, args.model_path)
    predict = output[1]

    mp = {
        0: "PNEUMONIA BACTERIA",
        1: "NORMAL",
        2: "PNEUMONIA VIRUS"
    }
    print(output[0])
    print(mp[predict])