# 🏥 AI-MedAssist

**AI-MedAssist** is an AI-powered medical diagnosis web application that predicts possible diseases based on user-reported symptoms.  
It leverages **machine learning models** to classify diseases, provides **confidence scores**, and recommends relevant **medical specialists**.  
Additionally, it performs **live Google search integration** to suggest real doctors’ profiles from trusted platforms like **Marham** and **Oladoc**.

---

## 🚀 Features

- **Symptom-based diagnosis** using:
  - Decision Tree  
  - Random Forest  
  - Naive Bayes  

- **94 unique symptoms** for user input  
- **40+ possible diseases** with mapped specialists  
- **Confidence scores** for each model prediction  
- **Doctor recommendations**:
  - Maps each disease → relevant specialist  
  - Fetches top specialist links in real-time via Google search  
- **User-friendly interface** built with Flask + Bootstrap 5  
- **Error handling**:
  - Prevents duplicate symptoms  
  - Detects contradictory symptoms (e.g., `constipation` vs `diarrhoea`)  

---

## 🖼️ Demo UI

### 🔹 Symptom Selection
- Users can choose up to **5 symptoms** from dropdowns.  
- Duplicate or contradictory symptoms are flagged before prediction.

### 🔹 Prediction Results
- Displays predicted diseases from 3 ML models.  
- Shows **confidence percentages**.  
- Suggests relevant **specialists**.  
- Provides **clickable doctor links** fetched live.  

---

## 📂 Project Structure

AI-MedAssist/
├── dataset/
│ ├── Training.csv # Training data
│ └── Testing.csv # Testing data
├── templates/
│ └── index.html # UI (Bootstrap + Jinja2)
├── static/
│ └── (optional CSS/JS files)
├── app.py # Main Flask application
├── requirements.txt # Project dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-MedAssist.git
cd AI-MedAssist
2. Create Virtual Environment (Recommended)
bash
Copy code
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy code
pip install Flask numpy pandas scikit-learn googlesearch-python
4. Add Datasets
Place Training.csv and Testing.csv inside the dataset/ folder.

5. Run the App
bash
Copy code
python app.py
6. Access Locally
Open in your browser:
👉 http://127.0.0.1:5000

📊 How It Works
User selects up to 5 symptoms from dropdown menus.

System validates inputs (no duplicates or contradictions).

Models predict most probable diseases:

Decision Tree

Random Forest

Naive Bayes

Confidence scores are calculated for each prediction.

Relevant specialists are suggested for the predicted disease.

Real doctor links are fetched live using Google search integration.

Results are displayed in a modern Bootstrap dashboard.

🔐 Notes & Limitations
❌ This project is not a replacement for professional medical advice. Always consult a licensed doctor.

🌍 Doctor results may vary by location, time, and availability (Google search dependent).

⚡ For real-world deployment, consider:

Adding geolocation-based searches

Using verified APIs (e.g., health databases, Google Maps API)

📌 Future Improvements
🔑 Add user authentication & personal health history tracking

📊 Enhance UI with interactive charts & graphs for confidence visualization

⚡ Optimize models with hyperparameter tuning

📍 Support geolocation-based doctor search

🔗 Integrate with trusted medical APIs (WebMD, Mayo Clinic, etc.)

📑 Export results as PDF medical reports

🤝 Contributing
Contributions are welcome!

Fork the repository

Create a new feature branch (feature-new)

Commit your changes

Push and open a Pull Request

📜 License
This project is open-source under the MIT License.

👨‍💻 Author
Developed by Your Name
📧 Email: your.email@example.com
🌐 GitHub: your-username
