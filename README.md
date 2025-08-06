# 🚗 Freespace Segmentation with FCNN

This project was developed as part of an internship focused on freespace segmentation for autonomous vehicles. The goal is to identify drivable areas in images using a Fully Convolutional Neural Network (FCNN).

## 📁 Project Structure
SAT_Intern/
│
├── data/
│ ├── images/ # Original images
│ └── masks/ # Generated masks
│
├── output/
│ ├── freespace_overlay.jpg # Overlayed segmentation result
│ └── show.png # Raw predicted mask
│
├── training/ # Training-related scripts and model
├── json2mask/ # JSON to mask conversion
├── test_model.py # Script to test trained model
└── trained_model.pth # Final model file

## 🧠 Model

- Architecture: Fully Convolutional Neural Network (FCNN)
- Framework: PyTorch
- Output: Binary segmentation masks (Freespace / Not Freespace)

## 🛠️ Pipeline

1. **Annotation → Mask Conversion** using OpenCV
2. **Tensorization** of images and masks
3. **Model Training** with loss tracking and validation
4. **Prediction** on test images
5. **Visualization** of results

---

## 🖼️ Example Outputs

| Predicted Mask | Original Image with Overlay |
|-----------------------------|----------------|
| ![Overlay](output/freespace_overlay.jpg) | ![Mask](output/show.png) |

---

## 📈 Training Performance

- 469 annotated image-mask pairs
- Trained for 20 epochs
- Achieved validation loss: **~0.17**

## 👨‍💻 Intern Summary
This project was conducted over 20 days and covered:

-JSON annotation parsing
-Preprocessing and mask generation
-PyTorch model building & training
-Augmentation and hyperparameter tuning
-Final evaluation and testing



---
