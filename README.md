# Leukemia Detection System
Deep learning model for leukemia detection from blood cell images.

## Tech Stack
Python, TensorFlow, Keras, OpenCV, MobileNetV2, Scikit-learn, Google Colab

## Dataset
Download from Google Drive: [Click Here](https://drive.google.com/drive/folders/1mUpqIvuqo9OcKggi4vaVeIFeim3UHqHX?usp=drive_link)

Place the dataset in your Google Drive under `MyDrive/Leukemiadata/` with the following structure:
```
Leukemiadata/
├── H_100X_C1/
├── H_100X_C2/
└── L_100X_C1/
```

## Features
- Binary classification: **Healthy** vs **Leukemia** blood cells
- Classifies leukemia cell types: lymphoblast, monoblast
- Classifies healthy cell types: lymphocyte, neutrophil, myelocyte, monocyte, eosinophil, basophil
- Transfer learning with MobileNetV2 (pretrained on ImageNet)
- Fine-tuning of the last 50 layers
- Data augmentation (rotation, zoom, horizontal flip)
- Confidence score output per prediction

## Model Architecture
```
MobileNetV2 (pretrained, partial fine-tune)
    └── GlobalAveragePooling2D
    └── Dense(256, relu)
    └── Dropout(0.4)
    └── Dense(1, sigmoid)
```
- Input size: 224 × 224 × 3
- Optimizer: Adam (lr=1e-5)
- Loss: Binary Crossentropy
- Trained for 10 epochs

## How to Run
1. Open `Leukemia_Detector.ipynb` in Google Colab
2. Mount Google Drive and place the dataset at `MyDrive/Leukemiadata/`
3. Run all cells to preprocess data, train, and evaluate the model
4. Saved model will be exported to `MyDrive/leukemia_binary_best_model.h5`

## Prediction
To run inference on a new image:
```python
img = cv2.imread("your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]
label = "LEUKEMIA" if pred > 0.5 else "HEALTHY"
confidence = pred if pred > 0.5 else 1 - pred
print(f"{label} — {confidence * 100:.2f}% confidence")
```
