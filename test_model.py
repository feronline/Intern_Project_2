import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from model import FoInternNet

IMAGE_PATH = "sample_data/cfc_000254.png"
MODEL_PATH = "trained_model.pth"
INPUT_SHAPE = (224, 224)
N_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor(),
])
input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

model = FoInternNet(input_size=INPUT_SHAPE, n_classes=N_CLASSES).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    output = output.squeeze().cpu().numpy()

freespace_mask = output[1]

binary_mask = (freespace_mask > 0.0043999999).astype(np.uint8)

original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
original_cv_resized = cv2.resize(original_cv, INPUT_SHAPE)


color_mask = np.zeros_like(original_cv_resized)
color_mask[:, :, 1] = binary_mask * 255


overlayed = cv2.addWeighted(original_cv_resized, 0.7, color_mask, 0.3, 0)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original_cv_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Freespace")
plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

os.makedirs("output", exist_ok=True)  # klasör yoksa oluştur
output_path = "output/freespace_overlay.jpg"
cv2.imwrite(output_path, overlayed)
print(f"Overlay görseli kaydedildi: {output_path}")
