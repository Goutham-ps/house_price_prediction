# 🏠 AI House Price Prediction & Recommendation App

An AI-powered web application that not only predicts house prices but also **recommends properties, visualizes market trends, and provides location-based insights**.

---

## 🚀 Live Demo

🔗 https://houseprediction-dn5iutwpnaqwacjffg5uxf.streamlit.app/

---

## 🧠 Features

* 📊 Predict house prices instantly
* 🏠 Smart property recommendation system
* 🧠 Explainable AI using SHAP
* 🗺️ Interactive Bangalore price map
* 📈 Data visualization & insights dashboard
* 🎯 Accuracy score (R²)
* 💡 Price category (Affordable / Mid-range / Premium)
* 🔍 Advanced filters (Location, Price, BHK)
* 📍 Location-based analysis
* 📂 Bulk prediction using CSV upload

---

## 🏠 Recommendation System

* Suggests similar properties based on:

  * Size
  * BHK
  * Rooms
* Location-aware recommendations
* Displays top matching properties with key details

---

## 🗺️ Map Features

* 📍 Visualizes properties across Bangalore
* 🎨 Color-coded pricing (Affordable → Premium)
* 📏 Dynamic marker size based on price
* 🧭 Smart zoom based on selected location
* 🖱️ Hover tooltips with formatted price
* 🎛️ Filters:

  * Location
  * Price range
  * BHK

---

## 📊 Dashboard Insights

* 📈 Average price per location
* 📊 Maximum price
* 🏘️ Number of listings
* 📍 Compare locations
* 🔍 Real-time filtering

---

## ⚙️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* SHAP
* Pandas / NumPy
* PyDeck (Map Visualization)
* Joblib

---

## 📊 Model Details

* Model: `HistGradientBoostingRegressor`

### Features Used:

* Size
* BHK
* Rooms
* Location
* Area Type
* Engineered Features:

  * bhk_per_size
  * total_space_per_room

---

## 📂 How to Run Locally

```bash
🔗 GitHub Repository: https://github.com/goutham-ps/house-price-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 CSV Format for Bulk Prediction

```csv
size,bhk,rooms,location,area_type
1200,2,2,Bangalore,Super built-up Area
1500,3,3,Bangalore,Built-up Area
```

---

## 📌 Future Improvements

* 🤖 Advanced ML recommender (KNN / cosine similarity)
* 🌐 Real-time property data (API integration)
* 📱 Mobile UI optimization
* 🔐 User authentication
* 📊 Model comparison (XGBoost, Random Forest)

---

## 👨‍💻 Author

Goutham P S

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
