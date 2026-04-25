import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import pydeck as pdk

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

df['lat'] = 12.90 + np.random.rand(len(df)) * 0.2
df['lon'] = 77.50 + np.random.rand(len(df)) * 0.2



# ----------------------------
# LOCATION GROUPING
# ----------------------------
location_counts = df['location'].value_counts()

df['location'] = df['location'].apply(
    lambda x: x if location_counts[x] > 30 else 'other'
)

df_encoded_full = pd.get_dummies(df, columns=['location', 'area_type'])


def recommend_properties(df, input_data, location, top_n=5):

    df_filtered = df[df['location'] == location].copy()

    if len(df_filtered) == 0:
        df_filtered = df.copy()

    df_filtered['score'] = (
        abs(df_filtered['size'] - input_data['size']) +
        abs(df_filtered['bhk'] - input_data['bhk']) * 10 +
        abs(df_filtered['rooms'] - input_data['rooms']) * 5
    )

    return df_filtered.sort_values('score').head(top_n)[
        ['location', 'size', 'bhk', 'rooms', 'price']
    ]
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

        st.markdown("---")
        st.subheader("🏠 Recommended Properties")

        input_dict = {
            "size": size,
            "bhk": bhk,
            "rooms": rooms
        }

        recommendations = recommend_properties(df, input_dict,selected_location)
        st.write("Selected location:", selected_location)
        st.dataframe(recommendations, use_container_width=True,hide_index=True)
        st.success("💡 Top result is the closest match to your requirements")

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
    st.markdown("### 📍 Bangalore Property Price Map")

    # ----------------------------
    # 🔍 SEARCH LOCATION
    # ----------------------------
    st.markdown("### 🔍 Search Location")

    location_options = sorted(df['location'].unique())

    selected_loc = st.selectbox(
        "Select a location",
        ["All"] + location_options
    )

    if selected_loc != "All":
        filtered_df = df[df['location'] == selected_loc]
    else:
        filtered_df = df.copy()

    # ----------------------------
    # 💰 FORMAT PRICE
    # ----------------------------
    def format_price(x):
        if x >= 100:
            return f"₹ {x/100:.2f} Cr"
        return f"₹ {x:.2f} L"

    # ----------------------------
    # 📊 INSIGHTS
    # ----------------------------
    st.markdown("### 📊 Location Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Price", format_price(filtered_df['price'].mean()))
    col2.metric("Max Price", format_price(filtered_df['price'].max()))
    col3.metric("Listings", len(filtered_df))

    # ----------------------------
    # 📍 MAP DATA (IMPORTANT FIX)
    # ----------------------------
    map_df = filtered_df.groupby('location').agg({
        'price': 'mean',
        'lat': 'mean',
        'lon': 'mean'
        }).reset_index()
    # ----------------------------
    # 🎛️ PRICE FILTER (FIXED)
    # ----------------------------
    min_val = int(map_df['price'].min())
    max_val = int(map_df['price'].max())

    if min_val == max_val:
        st.info(f"Only one price available: ₹ {min_val} L")
    else:
        min_price = st.slider(
            "Filter by Price",
            min_val,
            max_val,
            min_val
        )
        map_df = map_df[map_df['price'] >= min_price]   

    # ----------------------------
    # 🎛️ BHK FILTER (NEW)
    # ----------------------------
    bhk_filter = st.selectbox(
        "Filter by BHK",
        ["All"] + sorted(df['bhk'].dropna().unique().astype(int).tolist())
    )

    if bhk_filter != "All":
        filtered_df = filtered_df[filtered_df['bhk'] == bhk_filter]
        map_df = filtered_df.groupby('location').agg({
            'price': 'mean',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()

    # ----------------------------
    # 🎨 COLOR LOGIC (BETTER)
    # ----------------------------
    def get_color(price):
        if price > 150:
            return [255, 0, 0, 180]      # 🔴 Expensive
        elif price > 80:
            return [255, 165, 0, 160]    # 🟠 Mid
        else:
            return [0, 0, 255, 140]      # 🔵 Affordable

    map_df['color'] = map_df['price'].apply(get_color)

    # ----------------------------
    # 📏 RADIUS (REDUCE CLUTTER)
    # ----------------------------
    max_price = map_df['price'].max()

    if max_price == 0:
        map_df['radius'] = 100
    else:
        map_df['radius'] = (map_df['price'] / max_price) * 500

    # ----------------------------
    # 🎯 SMART ZOOM
    # ----------------------------
    if selected_loc != "All" and selected_loc in map_df['location'].values:
        lat = map_df[map_df['location'] == selected_loc]['lat'].values[0]
        lon = map_df[map_df['location'] == selected_loc]['lon'].values[0]

        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=13,
        )
    else:
        view_state = pdk.ViewState(
            latitude=12.9716,
            longitude=77.5946,
            zoom=11,
        )


    # ----------------------------
    # 🚀 MAP DISPLAY
    # ----------------------------
    def format_price_label(x):
        if x >= 100:
            return f"{x/100:.2f} Cr"
        return f"{x:.2f} L"

    map_df['price_label'] = map_df['price'].apply(format_price_label)

    
    # ----------------------------
    # 🗺️ MAP LAYER
    # ----------------------------
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius='radius',
        get_fill_color='color',
        pickable=True
    )



    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip = {
            "html": "<b>{location}</b><br/>Avg Price: ₹ {price_label}",
            "style": {"backgroundColor": "black", "color": "white"}
        }
    ))

    # ----------------------------
    # 🔥 INSIGHT BELOW MAP
    # ----------------------------
    if len(map_df) > 0:
        top_area = map_df.sort_values('price', ascending=False).iloc[0]

        st.success(
            f"🔥 Most expensive area: {top_area['location']} "
            f"(₹ {top_area['price']:.2f} Lakhs)"
        )

    st.subheader("📊 Market Insights")

    st.bar_chart(df['price'])

    loc_avg = df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(loc_avg)
# ============================
# ABOUT
# ============================
with tab3:
    st.write("Smart real estate prediction system") 