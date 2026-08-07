# ❤️ Cardiovascular Risk Predictor

## 📌 Project Overview

Cardiovascular Risk Predictor is a Machine Learning-based web application that predicts an individual's risk of developing cardiovascular disease using health-related parameters. The model is trained on the Framingham Heart Study dataset and deployed using Flask, providing an intuitive web interface for real-time risk prediction.

This project demonstrates the complete Machine Learning workflow, including data preprocessing, model training, evaluation, and deployment.

---

## 🚀 Features

- Predicts cardiovascular disease risk based on patient health information.
- User-friendly Flask web application.
- Data preprocessing and feature scaling.
- Logistic Regression-based prediction model.
- Trained model saved using Pickle.
- Real-time prediction through an interactive web interface.
- Displays prediction results instantly.

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- Pickle

---

## 📊 Dataset

The project uses the **Framingham Heart Study Dataset**, which contains various medical and lifestyle attributes used to estimate an individual's cardiovascular disease risk.

Some important features include:

- Age
- Gender
- Smoking Status
- Blood Pressure
- Cholesterol Level
- Diabetes
- Body Mass Index (BMI)
- Heart Rate
- Glucose Level

---

## 🤖 Machine Learning Model

- **Algorithm:** Logistic Regression
- **Data Preprocessing:**
  - Missing value handling
  - Feature scaling
  - Train-Test Split
- **Model Evaluation:**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

---

## 📁 Project Structure

```text
Cardiovascular-Risk-Predictor/
│
├── app.py
├── train.py
├── test_predict.py
├── requirements.txt
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_info.json
│
├── data/
│   └── framingham.csv
│
├── templates/
├── static/
├── assets/
├── screenshots/
│
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/Cardiovascular-Risk-Predictor.git
```

### Navigate to the Project Folder

```bash
cd Cardiovascular-Risk-Predictor
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will be available at:

```
http://127.0.0.1:5000
```

---

## 📈 Workflow

1. Load the Framingham Heart Study dataset.
2. Clean and preprocess the data.
3. Scale numerical features.
4. Train the Logistic Regression model.
5. Save the trained model using Pickle.
6. Deploy the model with Flask.
7. Predict cardiovascular disease risk from user input.

---

## 📸 Application Preview

Add screenshots of your application after uploading them to the `screenshots` folder.

Example:

- Home Page
- Risk Prediction Form
- Prediction Result

---

## 🎯 Future Enhancements

- Improve prediction accuracy using advanced machine learning algorithms.
- Integrate Deep Learning models.
- Add user authentication.
- Store prediction history in a database.
- Deploy the application on cloud platforms such as Render, Railway, or AWS.
- Provide personalized health recommendations based on prediction results.

---

## 📚 Learning Outcomes

This project helped in understanding:

- Machine Learning workflow
- Data preprocessing techniques
- Logistic Regression
- Feature scaling
- Model serialization using Pickle
- Flask web development
- Frontend and backend integration
- Real-time machine learning model deployment

---

## 👩‍💻 Author

**NESHWIN R A**

**B.Tech Information Technology**

Passionate about Machine Learning, Artificial Intelligence, and Data Analytics.

---

## 📄 License

This project is developed for educational and learning purposes.
