import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ----------------------------
# ⚙️ PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Real Estate Analyzer", layout="wide")

st.title("🏠 Smart Real Estate Analyzer")
st.markdown("### 🧠 AI-powered Price Prediction + Market Insights")

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
    score = joblib.load("score.pkl")  # ✅ load score
    return model, columns, score

@st.cache_resource
def get_explainer(_model):
    return shap.Explainer(_model)

# ----------------------------
# LOAD
# ----------------------------
df = load_data()
model, feature_columns, score = load_model()
explainer = get_explainer(model)

st.success("✅ Model loaded successfully")

# ----------------------------
# CLEANING
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

df_encoded_full = pd.get_dummies(df, columns=['location', 'area_type'])

# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Dashboard", "ℹ️ About"])

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

    st.write(f"📊 Model Accuracy (R²): **{score:.2f}**")  # ✅ from file

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

        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # 🔮 Prediction
    prediction = model.predict(input_data)[0]

    # ----------------------------
    # 💰 DISPLAY
    # ----------------------------
    st.markdown("## 💰 Estimated Price")
    st.markdown(
        f"<h1 style='text-align:center;color:#00cc66;'>₹ {prediction:.2f} Lakhs</h1>",
        unsafe_allow_html=True
    )

    # Range
    lower = prediction * 0.9
    upper = prediction * 1.1

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Price", f"{prediction:.2f}")

    with col2:
        st.metric("Range", f"{lower:.2f} - {upper:.2f}")

    with col3:
        if prediction < 75:
            category = "💸 Affordable"
        elif prediction < 200:
            category = "🏠 Mid-range"
        else:
            category = "💎 Premium"

        st.metric("Category", category)

    # Insight
    st.markdown("### 💡 Insight")
    if prediction > df['price'].mean():
        st.warning("This property is above average market price")
    else:
        st.info("This property is below average market price")

    st.markdown("---")

    # ----------------------------
    # 🧠 SHAP EXPLANATION
    # ----------------------------
    st.subheader("🧠 Why this price?")

    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact": shap_values.values[0]
    })

    shap_df["Feature"] = shap_df["Feature"].str.replace("location_", "Location: ")
    shap_df["Feature"] = shap_df["Feature"].str.replace("area_type_", "Area: ")

    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

    # 🔍 Top factors
    st.markdown("### 🔍 Top Influencing Factors")

    for _, row in shap_df.head(5).iterrows():
        color = "green" if row["Impact"] > 0 else "red"
        arrow = "⬆️" if row["Impact"] > 0 else "⬇️"

        st.markdown(
            f"{arrow} **{row['Feature']}** → <span style='color:{color}'>{row['Impact']:.2f}</span>",
            unsafe_allow_html=True
        )

    # 📊 Chart
    st.markdown("### 📊 Feature Impact")
    st.bar_chart(shap_df.head(8).set_index("Feature")["Impact"])    
    # CSV
    st.markdown("---")
    st.subheader("📂 Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        input_df['bhk_per_size'] = input_df['bhk'] / input_df['size']
        input_df['total_space_per_room'] = input_df['size'] / input_df['rooms']

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        preds = model.predict(input_encoded)
        input_df['Predicted Price'] = preds

        st.dataframe(input_df)

# ============================
# 📊 DASHBOARD
# ============================
with tab2:
    st.subheader("📊 Market Insights")

    st.bar_chart(df['price'])

    loc_avg = df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(loc_avg)

# ============================
# ABOUT
# ============================
with tab3:
    st.write("Smart real estate prediction system")