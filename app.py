import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ----------------------------
# ⚙️ PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Predictor")
st.markdown("### 🧠 AI-powered Real Estate Price Prediction")

# ----------------------------
# 🚀 CACHING
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("House_Data.xlsx")

@st.cache_resource
def load_model():
    model = joblib.load("house_model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

@st.cache_resource
def get_explainer(_model):
    return shap.Explainer(_model)

# ----------------------------
# 📂 LOAD
# ----------------------------
df = load_data()
model, feature_columns = load_model()
explainer = get_explainer(model)

st.success("✅ Model loaded successfully")

# ----------------------------
# 🧹 DATA CLEANING
# ----------------------------
df = df[['size', 'bhk', 'rooms', 'location', 'area_type', 'price']]

df['location'] = df['location'].astype(str).str.strip()
df['area_type'] = df['area_type'].astype(str).str.strip()

def convert_size(x):
    try:
        x = str(x)
        if '-' in x:
            vals = x.split('-')
            return (float(vals[0]) + float(vals[1])) / 2
        return float(''.join([c for c in x if c.isdigit() or c == '.']))
    except:
        return None

df['size'] = df['size'].apply(convert_size)
df = df[df['size'] < df['size'].quantile(0.99)]

df['bhk'] = df['bhk'].astype(str).str.extract(r'(\d+)')
df['bhk'] = pd.to_numeric(df['bhk'], errors='coerce')
df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df = df.dropna(subset=['size', 'bhk', 'rooms', 'location', 'price'])

df['bhk_per_size'] = df['bhk'] / df['size']
df['total_space_per_room'] = df['size'] / df['rooms']

# ----------------------------
# LOCATION GROUPING
# ----------------------------
location_counts = df['location'].value_counts()

df['location'] = df['location'].apply(
    lambda x: x if location_counts[x] > 30 else 'other'
)

df = pd.get_dummies(df, columns=['location', 'area_type'])

# ----------------------------
# MODEL ACCURACY
# ----------------------------
X = df.drop('price', axis=1)
y = df['price']

X = X.reindex(columns=feature_columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preds = model.predict(X_test)
score = r2_score(y_test, preds)

# ----------------------------
# 🧭 TABS
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Data", "ℹ️ About"])

# ============================
# 🏠 TAB 1: PREDICTION
# ============================
with tab1:

    st.sidebar.header("🔧 Input Parameters")

    size = st.sidebar.slider("Size (sq ft)", 500, 5000, 1000)
    bhk = st.sidebar.slider("BHK", 1, 5, 2)
    rooms = st.sidebar.slider("Rooms", 1, 5, 2)

    location_list = [col.replace("location_", "") for col in feature_columns if "location_" in col]
    area_type_list = [col.replace("area_type_", "") for col in feature_columns if "area_type_" in col]

    selected_location = st.sidebar.selectbox("Location", location_list)
    selected_area = st.sidebar.selectbox("Area Type", area_type_list)

    st.write(f"📊 Model Accuracy (R²): **{score:.2f}**")

    if st.button("Predict"):

        input_data = pd.DataFrame(columns=feature_columns)
        input_data.loc[0] = 0

        input_data['size'] = size
        input_data['bhk'] = bhk
        input_data['rooms'] = rooms
        input_data['bhk_per_size'] = bhk / size
        input_data['total_space_per_room'] = size / rooms if rooms != 0 else 0

        loc_col = f'location_{selected_location}'
        if loc_col in input_data.columns:
            input_data[loc_col] = 1

        area_col = f'area_type_{selected_area}'
        if area_col in input_data.columns:
            input_data[area_col] = 1

        with st.spinner("Predicting..."):
            prediction = model.predict(input_data)

        # 🎯 DISPLAY
        st.markdown("## 💰 Estimated Price")
        st.markdown(
            f"<h1 style='text-align:center;color:#00cc66;'>₹ {prediction[0]:.2f} Lakhs</h1>",
            unsafe_allow_html=True
        )

        lower = prediction[0] * 0.9
        upper = prediction[0] * 1.1

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Price", f"{prediction[0]:.2f}")

        with col2:
            st.metric("Range", f"{lower:.2f} - {upper:.2f}")

        with col3:
            if prediction[0] < 75:
                category = "💸 Affordable"
            elif prediction[0] < 200:
                category = "🏠 Mid-range"
            else:
                category = "💎 Premium"

            st.metric("Category", category)

        st.markdown("---")

        # ============================
        # 🧠 SHAP
        # ============================
        st.subheader("🧠 Why this price?")

        shap_values = explainer(input_data)

        shap_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Impact": shap_values.values[0]
        })

        shap_df["Feature"] = shap_df["Feature"].str.replace("location_", "Location: ")
        shap_df["Feature"] = shap_df["Feature"].str.replace("area_type_", "Area: ")

        shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

        st.markdown("### 🔍 Top Influencing Factors")

        for _, row in shap_df.head(5).iterrows():
            color = "green" if row["Impact"] > 0 else "red"
            arrow = "⬆️" if row["Impact"] > 0 else "⬇️"

            st.markdown(
                f"{arrow} **{row['Feature']}** → <span style='color:{color}'>{row['Impact']:.2f}</span>",
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Feature Impact")
        st.bar_chart(shap_df.head(8).set_index("Feature")["Impact"])

# ============================
# 📊 TAB 2: DATA
# ============================
with tab2:
    st.subheader("📊 Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Price vs Size")
    sample_df = df[['size', 'price']].sample(500)
    st.scatter_chart(sample_df.rename(columns={"size": "x", "price": "y"}))

# ============================
# ℹ️ TAB 3: ABOUT
# ============================
with tab3:
    st.markdown("""
    ## 📌 About This Project

    This app predicts house prices using Machine Learning.

    ### 🔧 Features:
    - HistGradientBoostingRegressor model
    - Feature engineering
    - SHAP explainability
    - Interactive UI

    ### 📊 Inputs:
    - Size
    - BHK
    - Rooms
    - Location
    - Area Type

    Built using Streamlit 🚀
    """)