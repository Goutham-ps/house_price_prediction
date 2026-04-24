import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ----------------------------
# ⚙️ PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("Enter house details to estimate price using AI")

# ----------------------------
# 📂 LOAD DATA
# ----------------------------
df = pd.read_excel("House_Data.xlsx")

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

# Remove outliers
df = df[df['size'] < df['size'].quantile(0.99)]

# Fix bhk
df['bhk'] = df['bhk'].astype(str).str.extract(r'(\d+)')
df['bhk'] = pd.to_numeric(df['bhk'], errors='coerce')

df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df = df.dropna(subset=['size', 'bhk', 'rooms', 'location', 'price'])

# Feature engineering
df['bhk_per_size'] = df['bhk'] / df['size']
df['total_space_per_room'] = df['size'] / df['rooms']

# ----------------------------
# LOCATION GROUPING
# ----------------------------
location_counts = df['location'].value_counts()

df['location'] = df['location'].apply(
    lambda x: x if location_counts[x] > 30 else 'other'
)

# ----------------------------
# ENCODING
# ----------------------------
df = pd.get_dummies(df, columns=['location', 'area_type'])

# ----------------------------
# MODEL
# ----------------------------
X = df.drop('price', axis=1)
y = df['price']

feature_columns = X.columns.tolist()

try:
    model = joblib.load("house_model.pkl")
    feature_columns = joblib.load("columns.pkl")
    st.success("✅ Model loaded successfully")

except:
    st.warning("⚙️ Training model for first time...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_depth=10,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=10
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "house_model.pkl")
    joblib.dump(feature_columns, "columns.pkl")

    st.success("💾 Model saved!")

# ----------------------------
# ACCURACY
# ----------------------------
X = X.reindex(columns=feature_columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preds = model.predict(X_test)
score = r2_score(y_test, preds)

st.write(f"📊 Model Accuracy (R²): {score:.2f}")

# ----------------------------
# INPUT
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

size = st.sidebar.slider("Size (sq ft)", 500, 5000, 1000)
bhk = st.sidebar.slider("BHK", 1, 5, 2)
rooms = st.sidebar.slider("Rooms", 1, 5, 2)

location_list = [col.replace("location_", "") for col in feature_columns if "location_" in col]
area_type_list = [col.replace("area_type_", "") for col in feature_columns if "area_type_" in col]

selected_location = st.sidebar.selectbox("Location", location_list)
selected_area = st.sidebar.selectbox("Area Type", area_type_list)

# ----------------------------
# PREDICTION
# ----------------------------
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

    prediction = model.predict(input_data)

    # ----------------------------
    # UI DISPLAY
    # ----------------------------
    st.subheader("💰 Predicted Price")

    lower = prediction[0] * 0.9
    upper = prediction[0] * 1.1

    if prediction[0] < 50:
        category = "💸 Low"
    elif prediction[0] < 150:
        category = "🏠 Medium"
    else:
        category = "💎 High"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Price", f"₹ {prediction[0]:.2f} Lakhs")

    with col2:
        st.metric("Range", f"{lower:.2f} - {upper:.2f}")

    with col3:
        st.metric("Category", category)

    if prediction[0] > 150:
        st.warning("⚠️ This property is relatively expensive")
    elif prediction[0] < 50:
        st.info("💡 This property is relatively affordable")

    st.markdown("---")

    # ----------------------------
    # SHAP
    # ----------------------------
    st.subheader("🧠 Why this price?")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact": shap_values.values[0]
    })

    shap_df["Feature"] = shap_df["Feature"].str.replace("location_", "Location: ")
    shap_df["Feature"] = shap_df["Feature"].str.replace("area_type_", "Area: ")

    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False).reset_index(drop=True)

    # Key factors
    st.markdown("### 🔍 Key Factors")

    for _, row in shap_df.head(3).iterrows():
        if row["Impact"] > 0:
            st.markdown(f"🟢 **{row['Feature']} increased price** (+{row['Impact']:.2f})")
        else:
            st.markdown(f"🔴 **{row['Feature']} decreased price** ({row['Impact']:.2f})")

    # Table
    st.markdown("### 📊 Detailed Impact")
    st.dataframe(shap_df.head(8), hide_index=True, use_container_width=True)
                        
    # Chart
    st.markdown("### 📈 Feature Impact")
    st.bar_chart(shap_df.head(8).set_index("Feature")["Impact"])

# ----------------------------
# DATA VIEW
# ----------------------------
with st.expander("📊 View Cleaned Dataset"):
    st.write(df)

# ----------------------------
# VISUALIZATION
# ----------------------------
st.subheader("📈 Price vs Size")
