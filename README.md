# 🏢 Big Data-Driven Apartment Rent Price Prediction

A machine learning web application that predicts apartment rental prices using property characteristics and data-driven analysis. The project uses Apache Spark (PySpark) for big data processing and machine learning, with a Streamlit interface for interactive predictions and visualizations.

## 🌐 Live Demo

**Try the application:** https://rent-prediction-pyspark.streamlit.app/

## 📌 Project Overview

The goal of this project is to estimate monthly apartment rent based on factors such as property type, location, number of bedrooms and bathrooms, floor area, property condition, furnishing, parking, and available amenities.

The project uses a Ghana apartment rental dataset containing approximately 17,890 listings.

## ✨ Key Features

- **Rent Prediction:** Estimate monthly rent using user-selected property details.
- **Data Insights:** Explore rental data and understand property-related trends.
- **Model Comparison:** Compare the performance of multiple regression models.
- **Feature Engineering:** Prepare and transform input features for machine learning.
- **Interactive Dashboard:** Use a Streamlit web interface for predictions and visualizations.
- **Saved Model Deployment:** Serve predictions using a saved model bundle without retraining the model whenever the app starts.

## 🛠️ Technologies Used

- Python
- Apache Spark (PySpark)
- PySpark MLlib
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib and Plotly
- GitHub
- Streamlit Community Cloud

## 🤖 Machine Learning Models

The project compares the following regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient-Boosted Trees (GBT)
5. XGBoost
6. LightGBM

Model performance is evaluated using regression metrics, including R² score. Hyperparameter tuning is also included in the project workflow.

## 📊 Model Performance

The model comparison results obtained during the initial evaluation were:

| Model | R² Score |
|---|---:|
| Random Forest | 0.8345 |
| Decision Tree | 0.8008 |
| Linear Regression | 0.7322 |

**Note:** These are initial comparison results. Final tuned-model performance may differ.

## 🖥️ Application Sections

- **Predict:** Enter property details and estimate monthly rent.
- **Data Insights:** Explore rental data and visualizations.
- **Model Comparison:** Review regression model performance.
- **About:** Learn more about the project and its methodology.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/abhipandey070/rent-prediction-pyspark.git
cd rent-prediction-pyspark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

Make sure `model_bundle.pkl` is present in the project directory alongside `app.py`.

## 📁 Project Structure

```text
rent-prediction-pyspark/
├── app.py
├── model_bundle.pkl
├── requirements.txt
└── README.md
```

The training notebook contains the data processing, feature engineering, model comparison, and model development workflow.

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud and connected to GitHub for source-code management and deployment updates.

**Live application:** https://rent-prediction-pyspark.streamlit.app/

## 🎯 Project Objective

To demonstrate how big data processing and machine learning can be combined to analyze rental listings, build predictive models, compare their performance, and provide an accessible web-based rent estimation tool.

## 👨‍💻 Author

**Abhishek Pandey**

B.Tech — Information Technology  
Vellore Institute of Technology (VIT), Vellore

GitHub: https://github.com/abhipandey070

---

*Academic project focused on big data analytics, machine learning, and interactive application development.*
