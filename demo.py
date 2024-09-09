import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from matplotlib import pyplot as plt
import os
import glob
from main_model import HyperMamba_Base as create_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class_names = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

checkpoint_path = ""

model = create_model(num_classes=7).to(device)
assert os.path.exists(checkpoint_path), "Checkpoint file does not exist."
checkpoint = torch.load(checkpoint_path, map_location=device)
filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.endswith(('total_ops', 'total_params'))}
model.load_state_dict(filtered_checkpoint)
model.eval()

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def process_image(img_path):
    image = Image.open(img_path).convert('RGB')

    input_tensor = data_transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    top_prob = probabilities[0][predicted].item() * 100

    predicted_class_name = class_names[predicted.item()]

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=30)
    text = f"Class: {predicted_class_name}, Prob: {top_prob:.2f}%"

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    image_width, image_height = image.size
    text_margin_ratio = 0.05
    text_margin_from_top = int(image_height * text_margin_ratio)

    text_position_y = text_margin_from_top
    margin = 10
    position = (margin, text_position_y)
    draw.text(position, text, fill=(0, 0, 255), font=font)

    return image

input_folder = ""
output_folder = ""

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = glob.glob(os.path.join(input_folder, '*.*'))

for img_path in image_files:
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        continue

    print(f"Processing {img_path}...")

    processed_image = process_image(img_path)

    output_filename = os.path.basename(img_path)
    output_path = os.path.join(output_folder, output_filename)

    processed_image.save(output_path, format='TIFF', dpi=(300, 300))

    print(f"Processed image saved to {output_path}")

print("All images processed.")