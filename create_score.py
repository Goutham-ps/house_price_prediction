import pandas as pd
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

print("🚀 Running score script...")

try:
    df = pd.read_excel("House_Data.xlsx")
    print("✅ Data loaded")

    df = df[['size', 'bhk', 'rooms', 'location', 'area_type', 'price']]

    # ----------------------------
    # FIX SIZE (IMPORTANT)
    # ----------------------------
    def convert_size(x):
        try:
            x = str(x)
            if '-' in x:
                a, b = x.split('-')
                return (float(a) + float(b)) / 2
            return float(''.join([c for c in x if c.isdigit() or c == '.']))
        except:
            return None

    df['size'] = df['size'].apply(convert_size)

    # ----------------------------
    # FIX OTHER COLUMNS
    # ----------------------------
    df['bhk'] = df['bhk'].astype(str).str.extract(r'(\d+)')
    df['bhk'] = pd.to_numeric(df['bhk'], errors='coerce')

    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    print("🔍 Before dropna:", df.shape)

    # SAFE DROP
    df = df.dropna(subset=['size', 'bhk', 'rooms', 'price'])

    print("🔍 After dropna:", df.shape)

    if len(df) == 0:
        print("❌ Still empty dataset!")
        exit()

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    df['bhk_per_size'] = df['bhk'] / df['size']
    df['total_space_per_room'] = df['size'] / df['rooms']

    # ----------------------------
    # ENCODING
    # ----------------------------
    df = pd.get_dummies(df, columns=['location', 'area_type'])

    model = joblib.load("house_model.pkl")
    columns = joblib.load("columns.pkl")

    X = df.drop("price", axis=1)
    y = df["price"]

    X = X.reindex(columns=columns, fill_value=0)

    # ----------------------------
    # SPLIT + SCORE
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    joblib.dump(score, "score.pkl")

    print("✅ score.pkl created successfully")
    print("📊 Score:", score)

except Exception as e:
    print("❌ ERROR:", e)