# 🧠 Sign Language Detection using Streamlit

## 📘 Overview
This project is a **Sign Language Detection Web App** built with **Streamlit** and a **Machine Learning / Deep Learning model**.  
It helps bridge communication between individuals with speech or hearing impairments and normal users by **translating sign language gestures into text** in real time.

---

## 🎯 Features
- ✋ Detects hand gestures or sign language symbols from webcam or uploaded images  
- 🧩 Uses a trained ML/DL model (e.g., CNN or YOLOv8) for sign recognition  
- 🗣️ Displays the **predicted text** corresponding to the sign  
- 💬 User-friendly web interface powered by **Streamlit**  
- 📷 Option to **capture live video** or **upload images** for detection  
- 🔁 Real-time prediction and output display  

---

## 🧰 Tech Stack
| Category     | Technology Used |
|--------------|----------------|
| **Frontend** | Streamlit |
| **Backend**  | Python |
| **Model**    | YOLOv8 / CNN (custom trained on sign language dataset) |
| **Libraries** | OpenCV, NumPy, TensorFlow / PyTorch, Ultralytics, Pillow |
| **Data**     | Custom sign language image dataset |

---

## 🏗️ Project Structure

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/Paramasivam-2005/mini-project.git
cd mini-project
python -m venv venv
venv\Scripts\activate     # For Windows
source venv/bin/activate  # For Linux/Mac
pip install -r requirements.txt
streamlit run app.py

---



