import streamlit as st
import pandas as pd
import joblib
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

# 🔥 REMOVE OUTLIERS
df = df[df['size'] < df['size'].quantile(0.99)]

df['bhk'] = df['bhk'].astype(str).str.extract('(\d+)')
df['bhk'] = pd.to_numeric(df['bhk'], errors='coerce')

df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df = df.dropna(subset=['size', 'bhk', 'rooms', 'location', 'price'])

# ----------------------------
# 🔥 LOCATION GROUPING
# ----------------------------
location_counts = df['location'].value_counts()

df['location'] = df['location'].apply(
    lambda x: x if location_counts[x] > 10 else 'other'
)

# ----------------------------
# 🔢 ENCODING
# ----------------------------
df = pd.get_dummies(df, columns=['location', 'area_type'])

# ----------------------------
# 🤖 MODEL TRAIN / LOAD
# ----------------------------
try:
    model = joblib.load("house_model.pkl")
    feature_columns = joblib.load("columns.pkl")
    st.success("✅ Model loaded successfully")

except:
    st.warning("⚙️ Training model for first time...")

    model = HistGradientBoostingRegressor(max_depth=6)

   
    # Save model + columns
    joblib.dump(model, "house_model.pkl")
    joblib.dump(feature_columns, "columns.pkl")

    st.success("💾 Model saved!")

X = df.drop('price', axis=1)
y = df['price']

# align columns
X = X.reindex(columns=feature_columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preds = model.predict(X_test)
score = r2_score(y_test, preds)

st.write(f"📊 Model Accuracy (R²): {score:.2f}")

# ----------------------------
# 🎛️ USER INPUT
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

size = st.sidebar.slider("Size (sq ft)", 500, 5000, 1000)
bhk = st.sidebar.slider("BHK", 1, 5, 2)
rooms = st.sidebar.slider("Rooms", 1, 5, 2)

# Extract location & area_type options
location_list = [col.replace("location_", "") for col in feature_columns if "location_" in col]
area_type_list = [col.replace("area_type_", "") for col in feature_columns if "area_type_" in col]

selected_location = st.sidebar.selectbox("Location", location_list)
selected_area = st.sidebar.selectbox("Area Type", area_type_list)

# ----------------------------
# 🔮 PREDICTION
# ----------------------------
if st.button("Predict"):
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0

    input_data['size'] = size
    input_data['bhk'] = bhk
    input_data['rooms'] = rooms

    # location
    loc_col = f'location_{selected_location}'
    if loc_col in input_data.columns:
        input_data[loc_col] = 1

    # area type
    area_col = f'area_type_{selected_area}'
    if area_col in input_data.columns:
        input_data[area_col] = 1

    prediction = model.predict(input_data)

    st.subheader("💰 Predicted Price")
    st.success(f"₹ {prediction[0]:.2f} Lakhs")
    st.metric("Estimated Price", f"{prediction[0]:.2f}")

# ----------------------------
# 📊 DATA VIEW
# ----------------------------
with st.expander("📊 View Cleaned Dataset"):
    st.write(df)

# ----------------------------
# 📈 VISUALIZATION
# ----------------------------
st.subheader("📈 Price vs Size")
st.scatter_chart(df[['size', 'price']])