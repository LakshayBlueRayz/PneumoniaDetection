# 🩺 Pneumonia Detection System

An AI-powered web application for detecting **pneumonia from chest X-ray images** using **deep learning** and **transfer learning (ResNet50)**. Built with **Streamlit** for a user-friendly interface and **TensorFlow** for model inference.

---

## 📸 How It Works

1. Upload a chest X-ray image (JPG, PNG, or JPEG format).
2. The image is preprocessed and passed through a trained ResNet50-based model.
3. The model outputs probabilities for:
   - **Normal**
   - **Pneumonia**
4. Results are displayed with confidence levels, visual progress, and warnings when the model is uncertain.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **Model:** TensorFlow with ResNet50 (Transfer Learning)
- **Image Handling:** PIL, NumPy
- **Prediction File:** `transfer_pneumonia_model.h5`

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

### 2. Create a Virtual Environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

### 4. Add the Pretrained Model
Place your trained model file transfer_pneumonia_model.h5 in the root directory of the project.


🚀 Run the App
bash
Copy
Edit
streamlit run app.py
