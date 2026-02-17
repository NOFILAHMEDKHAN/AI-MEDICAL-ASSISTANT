# 🏥 College Doctor — AI Symptom Checker

> An intelligent disease prediction web app built with Flask and Machine Learning that predicts diseases from symptoms and recommends the right specialist doctors in Pakistan.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview
College Doctor is a machine learning-powered web application that allows users to select 3 to 5 symptoms and receive disease predictions from three different ML models. For each predicted disease, the app recommends the most relevant specialist and provides direct links to find doctors on Pakistani medical platforms.

---

## ✨ Features
- 🔍 Symptom-based Disease Prediction — Select 3–5 symptoms from 94 available options
- 🤖 3 ML Models — Decision Tree, Random Forest, and Naive Bayes running in parallel
- 📊 Confidence Scores — Each prediction shows a confidence percentage
- 👨‍⚕️ Specialist Recommendation — Maps each disease to the most accurate medical specialist
- 🔗 Doctor Directory Links — Direct links to Marham.pk, OlaDoc, and Healthwire
- ⚠️ Contradictory Symptom Detection — Prevents logically impossible symptom combinations
- 🎨 Modern UI — Clean, responsive design with animated prediction cards

---

## 🧠 ML Models Used

| Model | Description |
|-------|-------------|
| Decision Tree | Fast, interpretable rule-based classifier |
| Random Forest | Ensemble of decision trees for higher accuracy |
| Naive Bayes | Probabilistic classifier based on Bayes' theorem |

---

## 🗂️ Project Structure
```
medical assistant/
│
├── app.py                  # Flask backend — models, routes, doctor links
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── templates/
    └── index.html          # Frontend UI
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/college-doctor.git
cd college-doctor
```

### 2. Create & Activate Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Dataset
Download from Kaggle: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

Place both files and update paths in app.py:
```python
df = pd.read_csv("path/to/Training.csv")
tr = pd.read_csv("path/to/Testing.csv")
```

### 5. Run the App
```bash
python app.py
```
Open browser at: http://127.0.0.1:5000

---

## 🖥️ How to Use
1. Open the app in your browser
2. Select 3 to 5 symptoms from the dropdown menus
3. Click Predict Disease
4. View predictions from all 3 models with confidence scores
5. Click specialist links to find doctors on Marham.pk, OlaDoc, or Healthwire

---

## 🩺 Diseases Covered (41 Total)

| Category | Diseases |
|----------|----------|
| Liver | Jaundice, Hepatitis A/B/C/D/E, Alcoholic Hepatitis, Chronic Cholestasis |
| Infectious | Malaria, Dengue, Typhoid, Chicken Pox, AIDS, Tuberculosis, Impetigo |
| Respiratory | Common Cold, Pneumonia, Bronchial Asthma |
| Digestive | GERD, Gastroenteritis, Peptic Ulcer, Piles |
| Cardiovascular | Heart Attack, Hypertension, Varicose Veins |
| Endocrine | Diabetes, Hypothyroidism, Hyperthyroidism, Hypoglycemia |
| Neurological | Migraine, Paralysis, Vertigo, Cervical Spondylosis |
| Skin | Acne, Psoriasis, Fungal Infection, Allergy, Drug Reaction |
| Other | Arthritis, Osteoarthritis, UTI |

---

## 🔒 Constraints & Validations
- Minimum 3 symptoms required for prediction
- Maximum 5 symptoms allowed
- Duplicate symptoms are blocked
- Contradictory symptoms are detected and rejected

---

## 📦 Dependencies
```
flask
numpy
pandas
scikit-learn
```

---

## 🌐 Doctor Platforms Integrated

| Platform | URL |
|----------|-----|
| Marham.pk | https://www.marham.pk |
| OlaDoc | https://oladoc.com |
| Healthwire | https://healthwire.pk |

---

## 🚀 Future Improvements
- [ ] Add user login and symptom history
- [ ] Show disease descriptions and precautions
- [ ] Deploy on Heroku or Render
- [ ] Add multilingual support (Urdu)

---

## 👨‍💻 Author

**Developed by:** Nofil Ahmed Khan  
Computer Science | NED University of Engineering and Technology  

📧 **Email:** nofil2012@gmail.com  
🌐 **LinkedIn:** [linkedin.com/in/khannofil](https://linkedin.com/in/khannofil)  

💬 *Engineering practical AI solutions that merge intelligence, interaction, and innovation.*

---

## 📜 License & Usage Policy

⚠️ **Important Notice:**  
This project is open-source for **educational viewing**, but reproduction, commercial use, or copying of this code without explicit permission is **strictly prohibited**.

Please contact the author (**Nofil Ahmed Khan**) via email or LinkedIn to request permission before using any part of this repository.

---


---

<div align="center">

**Created with 💙 by Nofil Ahmed Khan — where AI meets real-world productivity.**

</div>