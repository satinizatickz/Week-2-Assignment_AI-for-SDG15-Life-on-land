# 🌍 Deforestation Detection Using CNNs  
**AI for Sustainable Development — SDG 15: Life on Land**  
> “AI can help us see the forest *and* the trees — before it’s too late.”

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🧩 Project Overview  
Deforestation is one of the leading causes of climate change, biodiversity loss, and land degradation.  
This project leverages **Convolutional Neural Networks (CNNs)** to classify satellite imagery and detect deforestation patterns.  
By integrating AI with sustainability, it supports **SDG 15 — Life on Land**, contributing to global conservation efforts.

---

## 🎯 Objectives  
- Build and train a **CNN model** to classify satellite images into:
  - 🌲 Forest areas  
  - 🏜️ Deforested areas  
  - 🌾 Other land covers  
- Demonstrate how **AI supports sustainable land management**.  
- Provide a **reproducible and open-source** workflow using Python + TensorFlow.  
- Promote **ethical and fair AI** for environmental monitoring.

---

## 🌱 Alignment with SDG 15 — Life on Land  

| SDG Target | Role of AI | Impact |
|-------------|-------------|--------|
| **15.1** — Ensure conservation and sustainable use of terrestrial ecosystems | Automated deforestation detection | Enables data-driven monitoring and policy support |

---

## 🧠 Machine Learning Approach  

- **Type:** Supervised Learning  
- **Algorithm:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  
- **Task:** Image Classification  
- **Metrics:** Accuracy, Precision, Recall, F1-Score  

**Example Architecture:**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Forest / Deforested / Other
])Week-2-Assignment_AI-for-SDG15-Life-on-land/
├── README.md
├── requirements.txt
├── src/
│   └── deforestation_cnn.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── cnn_model.h5
├── demo_screenshots/
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── model_architecture.png
├── Deforestation_Detection_CNN.ipynb
├── report.pdf
└── pitch_deck.pdf# Clone repository
git clone https://github.com/satinizatickz/Week-2-Assignment_AI-for-SDG15-Life-on-land.git
cd Week-2-Assignment_AI-for-SDG15-Life-on-land

# Install dependencies
pip install -r requirements.txt

# Train model
python src/deforestation_cnn.py --train

# Run prediction
python src/deforestation_cnn.py --predict test_images/https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflowhttps://img.shields.io/badge/Status-Active-success
