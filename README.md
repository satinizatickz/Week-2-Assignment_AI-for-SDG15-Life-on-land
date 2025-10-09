🌍 Deforestation Detection Using CNNs

AI for Sustainable Development — SDG 15: Life on Land

> “AI can help us see the forest and the trees — before it’s too late.” 🌳🤖




---

📘 Project Overview

Deforestation remains a major environmental challenge, leading to biodiversity loss, soil degradation, and increased greenhouse gas emissions. This project leverages Convolutional Neural Networks (CNNs) to automatically detect deforested regions from satellite images.

By combining machine learning with sustainability goals, this project contributes to SDG 15 – Life on Land, helping to protect, restore, and promote sustainable use of terrestrial ecosystems.


---

🎯 Objectives

Develop a supervised CNN model to classify satellite images into:

🌲 Forest areas

🪵 Deforested areas

🌾 Other land covers


Demonstrate how AI can support sustainable land management.

Showcase reproducible code using Python and TensorFlow.

Reflect on the ethical implications of using AI for environmental monitoring.



---

🌱 Alignment with SDG 15 – Life on Land

Aspect	Description

SDG Target	15.1 – Ensure the conservation and sustainable use of terrestrial ecosystems
AI Role	Automated monitoring and early detection of deforestation
Impact	Supports policymakers, NGOs, and conservation groups with data-driven insights



---

🧠 Machine Learning Approach

Type: Supervised Learning

Algorithm: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Task: Image Classification

Metrics: Accuracy, Precision, Recall, F1-Score


Model Architecture

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Forest / Deforested / Other
])


---

📊 Dataset Information

Dataset Source:
Kaggle – Forest and Non-Forest Images

Description:

High-resolution satellite images labeled as forest, non-forest, or other.

Dataset split: 80% training, 20% validation/test.

Data augmentation applied to improve generalization.



---

⚙️ Workflow

1️⃣ Data Preprocessing

Resize images to 128×128

Normalize pixel values (0–1)

Augment using rotation, flip, zoom


2️⃣ Model Training

Train CNN using Adam optimizer

Monitor validation accuracy and loss

Save best model weights


3️⃣ Evaluation

Compute metrics (Accuracy, Precision, Recall, F1-score)

Display confusion matrix and classification report

Plot accuracy/loss graphs


4️⃣ Visualization

Example outputs to include:

Model Accuracy & Loss Plot



Confusion Matrix Heatmap



Sample Predictions (Forest vs Deforested)



Model Architecture Summary



Training Output Demo (Google Colab)




---

💻 How to Run Locally (or on Google Colab)

1. Clone the repository

git clone https://github.com/<your-username>/deforestation-detection-cnn.git
cd deforestation-detection-cnn

2. Install dependencies

pip install -r requirements.txt

3. Run training

python src/deforestation_cnn.py --train

4. Run inference

python src/deforestation_cnn.py --predict test_images/

Or open the notebook version:

Deforestation_Detection_CNN.ipynb


---

📦 Repository Structure

deforestation-detection-cnn/
│
├── README.md                    # This file
├── requirements.txt              # Dependencies
├── src/
│   └── deforestation_cnn.py      # Main Python script
├── data/
│   ├── train/                    # Training images
│   └── test/                     # Testing images
├── models/
│   └── cnn_model.h5              # Saved trained model
├── demo_screenshots/             # Output visuals (add your own)
├── report.pdf                    # 1-page report (optional)
└── pitch_deck.pdf                # Pitch presentation (optional)


---

📈 Results Summary

Metric	Value

Accuracy	~91%
Precision	0.88
Recall	0.90
F1-Score	0.89


(Add screenshots inside the demo_screenshots/ folder and ensure file names match the placeholders above.)


---

🧩 Ethical Reflection

Bias Risks:

Unequal data distribution (certain regions underrepresented).

Seasonal variations may cause false deforestation alerts.


Fairness & Sustainability:

Use globally diverse datasets.

Validate predictions with local forestry experts.

Encourage open access and transparency in data and model sharing.



---

🚀 Future Improvements

Integrate Google Earth Engine API for real-time monitoring.

Deploy a Streamlit web app for interactive visualization.

Experiment with Transfer Learning (ResNet50 / MobileNet) for higher accuracy.

Add time-series prediction for early warning of deforestation trends.



---

🪪 License

This project is open-source under the MIT License.
You are free to use, modify, and share it for educational and research purposes.


---

🧾 Credits

Developed as part of the PLP Academy Week 2 Assignment – “Machine Learning Meets the UN SDGs.”
Author: Zatickz Satini
Instructor: PLP AI for Sustainable Development Program


---



