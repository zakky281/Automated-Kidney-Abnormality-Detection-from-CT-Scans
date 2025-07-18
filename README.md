# Automated Kidney Abnormality Detection from CT Scans

This project uses a deep learning approach to detect kidney abnormalities (Normal, Cyst, Tumor, and Stone) from CT scan images. Built with Python and powered by a ResNet50 model, the application includes a Streamlit-based web interface for user-friendly interaction.

For Model KERAS : https://drive.google.com/drive/folders/1u290_Jcsg8wTTXtvlWJ9EcTuJ0lI9D7J?usp=sharing

## 🧠 Overview

- **Dataset**: [CT Kidney Dataset (Normal, Cyst, Tumor, and Stone)](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- **Model**: Transfer learning with ResNet50
- **Interface**: Web-based GUI using Streamlit
- **Environment**: Python virtual environment
- **Goal**: Classify CT kidney images into one of four categories to assist in early diagnosis and reduce manual errors

---

## 🛠️ Installation (FOR WINDOWS)

1. **Clone the repository:**

```bash
git clone https://github.com/zakky281/Automated-Kidney-Abnormality-Detection-from-CT-Scans.git
cd Automated-Kidney-Abnormality-Detection-from-CT-Scans
```

2. **Create and activate a virtual environment**

```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\Activate.ps1

# Activate it (macOS/Linux)
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
---

## 🚀 Usage (FOR WINDOWS)
1. **Run the Streamlit App**
```bash
streamlit run app.py
```
Upload a CT scan image through the GUI and get real-time predictions.

2. **Train the Model**
```bash
python train_resnet.py
```

if you want to train baseline model
```bash
python train_baseline.py
```

3. **Evaluate Model**
```bash
python evaluate_model.py
```

## Performance Summary
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Accuracy           | \~95%                       |
| Precision          | High                        |
| Recall             | High                        |
| Confusion Matrix   | See `confusion_matrix.png`  |
| Misclassifications | See `misclassified_images/` |

## 📌 Dataset
Source: Kaggle - CT Kidney Dataset
Categories: Normal, Cyst, Tumor, Stone
Preprocessing: Images resized and normalized for ResNet50 input

## 📈 Future Improvements
- Add Grad-CAM explainability
- Integrate segmentation + classification
- Add user authentication for clinical deployments
- Enable batch predictions through GUI
