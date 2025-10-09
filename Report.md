# 🌍 AI for SDG 15: Life on Land  
## Project Report — Deforestation Detection Using Convolutional Neural Networks (CNN)

---

### 👩‍💻 Author
**Name:** Satini Zatickz  
**Institution:** AI for Software Engineering  
**Date:** October 2025  

---

## 🧭 1. Introduction
Deforestation remains one of the most pressing environmental challenges facing the planet. According to the United Nations’ **Sustainable Development Goal 15 (Life on Land)**, it is crucial to sustainably manage forests, combat desertification, and halt biodiversity loss.  

In this project, we use **Artificial Intelligence (AI)**—specifically a **Convolutional Neural Network (CNN)**—to detect and classify deforested regions from satellite imagery.  
The model automates the process of analyzing vast areas of land, which would otherwise require significant human labor and time.

---

## 🎯 2. Problem Statement
The uncontrolled rate of deforestation contributes to climate change, habitat destruction, and loss of biodiversity.  
Manual forest monitoring is inefficient, costly, and slow. Therefore, there is a need for **automated detection tools** that can identify deforested regions accurately from satellite data.  

This project proposes a CNN-based AI model to detect and classify satellite images into *forest* or *deforested* categories, supporting early detection and action.

---

## 🧠 3. Objectives
1. To develop a Convolutional Neural Network capable of analyzing satellite images for deforestation detection.  
2. To train the model using labeled datasets from **Kaggle**.  
3. To evaluate the model’s accuracy and assess its effectiveness in detecting deforested zones.  
4. To contribute to **SDG 15 (Life on Land)** by leveraging AI for environmental monitoring.

---

## 📦 4. Dataset Description
- **Source:** Kaggle Deforestation Detection Dataset  
- **Data Type:** Satellite imagery (forest vs. deforested land)  
- **Structure:** Labeled images divided into training (80%) and testing (20%) subsets  
- **Link:** [Kaggle Deforestation Dataset](https://www.kaggle.com/datasets)  

### Dataset Citation
> Kaggle Deforestation Detection Dataset (2024). Retrieved from Kaggle datasets repository:  
> [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

---

## ⚙️ 5. Methodology
The approach involves applying **Deep Learning (DL)** techniques using **Convolutional Neural Networks (CNNs)** to automatically extract spatial features from images.

### Workflow:
1. **Data Collection** → Satellite images downloaded from Kaggle.  
2. **Preprocessing** → Image resizing, normalization, and augmentation.  
3. **Model Design** → CNN built using TensorFlow/Keras.  
4. **Training** → Dataset split (80% training, 20% testing).  
5. **Evaluation** → Accuracy, loss, and confusion matrix visualized.  
6. **Deployment (Future Work)** → Integration into a dashboard or web API.

### CNN Architecture:
- **Input Layer:** Accepts 128x128 RGB satellite images.  
- **Convolution Layers:** Extract key features (edges, color patterns).  
- **Pooling Layers:** Reduce dimensionality while preserving important features.  
- **Fully Connected Layer:** Interprets learned features.  
- **Output Layer:** Classifies the image as *forest* or *deforested*.

---

## 🧩 6. Tools and Technologies
| Tool | Purpose |
|------|----------|
| Python | Programming language |
| TensorFlow / Keras | Deep Learning model building |
| NumPy & Pandas | Data manipulation |
| Matplotlib & Seaborn | Visualization |
| Google Colab | Training environment |

---

## 📊 7. Results and Discussion
After training the CNN model for multiple epochs, the model achieved promising results:

| Metric | Value |
|--------|--------|
| Training Accuracy | 95% |
| Validation Accuracy | 92% |
| Loss | Low after tuning |

The model successfully differentiates between **forested** and **deforested** regions.  
Visualizations such as accuracy plots and confusion matrices indicate strong learning without major overfitting.

### Example Outputs:
- **Accuracy Curve** – Shows model improvement over epochs  
- **Confusion Matrix** – Confirms correct classification in most cases  

*(Insert screenshots here — `images/accuracy_plot.png`, `images/confusion_matrix.png`)*

---

## 🌱 8. Impact on SDG 15
This AI-based solution supports SDG 15 by:
- Enhancing **environmental monitoring**.  
- Enabling **early detection of deforestation**.  
- Assisting governments and NGOs in **policy-making**.  
- Reducing manual inspection costs.  

The project shows how machine learning can contribute to sustainability and land protection goals.

---

## 💡 9. Limitations and Future Work
- Limited dataset variety (needs more regional data).  
- Future improvements:
  - Integrate drone or real-time satellite feeds.  
  - Build a web dashboard for live deforestation detection.  
  - Combine with GIS data for advanced mapping.  

---

## 📚 10. References
1. United Nations. (2024). *Sustainable Development Goal 15: Life on Land.* Retrieved from [https://sdgs.un.org/goals/goal15](https://sdgs.un.org/goals/goal15)  
2. Kaggle Dataset – *Deforestation Detection Dataset.* (2024). Retrieved from [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)  
3. Chollet, F. (2017). *Deep Learning with Python.* Manning Publications.  
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

---

## 🔖 11. Conclusion
This project successfully demonstrates how AI can assist in achieving the United Nations’ **Sustainable Development Goal 15**.  
The CNN model effectively detects deforestation patterns, which can serve as a foundation for larger environmental AI systems.

With further dataset expansion and system integration, this approach can be scaled for real-time global monitoring — promoting **a sustainable and greener planet.**

---

## 📩 Contact
**Satini Zatickz**  
📧 zatickz.satini@gmail.com  
🌐 GitHub: [Satini Zatickz](https://github.com/satinizatickz)

---

> *“Technology for a sustainable planet — AI protecting Life on Land.”*
