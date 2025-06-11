import streamlit as st
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import random

# Connect to MongoDB (raw data and analysis results)
def get_mongo_client():
    return MongoClient("mongodb://localhost:27017/")

def get_db():
    return get_mongo_client()["car_insurance"]

def get_collection(name):
    return get_db()[name]

# Load raw data
def load_data():
    df = pd.DataFrame(list(get_collection("raw_data").find()))
    df.drop(columns=['_id'], inplace=True, errors='ignore')

    le = LabelEncoder()
    categorical = df.select_dtypes(include='object').columns
    for col in categorical:
        df[col] = le.fit_transform(df[col].astype(str))

    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    return df

# Train model
def train_model(df):
    y = df['OUTCOME']
    X = df.drop(columns=['OUTCOME'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, X.columns.tolist(), mse, r2

# Turkish market price adjustment
def adjust_price(base_risk_score, user_data):
    # Convert risk score (0–1) into price (TRY)
    base_price = 10000 + (base_risk_score * 20000)  # Between ₺10k–₺30k
    factors = 1.0

    if user_data.get("AGE", 30) < 25:
        factors += 0.1
    if user_data.get("PAST_ACCIDENTS", 0) > 0:
        factors += 0.1 * user_data["PAST_ACCIDENTS"]
    if user_data.get("SPEEDING_VIOLATIONS", 0) > 0:
        factors += 0.05 * user_data["SPEEDING_VIOLATIONS"]
    if user_data.get("DUIS", 0) > 0:
        factors += 0.2 * user_data["DUIS"]

    return round(base_price * factors * random.uniform(0.98, 1.02), 2)


# Load MongoDB analysis results
def load_analysis_results():
    results = {}
    for col_name in get_db().list_collection_names():
        if col_name.startswith("analysis_"):
            results[col_name] = pd.DataFrame(list(get_collection(col_name).find())).drop(columns=['_id'], errors='ignore')
    return results

# Main Streamlit App
def main():
    st.set_page_config(page_title="Car Insurance System", layout="wide")
    st.title("🚗 Car Insurance Prediction System")

    df = load_data()
    model, features, mse, r2 = train_model(df)

    st.subheader("1. Enter User Information")
    col1, col2 = st.columns(2)
    user_data = {}
    with col1:
        user_data['AGE'] = st.slider("Age", 18, 90, 30)
        user_data['PAST_ACCIDENTS'] = st.number_input("Past Accidents", 0, 10, 0)
        user_data['ANNUAL_MILEAGE'] = st.number_input("Annual Mileage", 0, 100000, 10000)
    with col2:
        user_data['DUIS'] = st.number_input("DUIs", 0, 10, 0)
        user_data['CREDIT_SCORE'] = st.slider("Credit Score", 0.0, 1.0, 0.5)
        user_data['SPEEDING_VIOLATIONS'] = st.number_input("Speeding Violations", 0, 20, 0)

    if st.button("💰 Predict Insurance Price"):
        user_df = pd.DataFrame([user_data])
        for col in features:
            if col not in user_df.columns:
                user_df[col] = df[col].mean()
        user_df = user_df[features]
        base = model.predict(user_df)[0]
        price = adjust_price(base, user_data)
        st.success(f"Estimated Insurance Price (TRY): ₺{price:,.2f}")

    st.markdown("---")
    st.subheader("2. Veri Analizi (MongoDB Sonuçları)")
    analysis_data = load_analysis_results()
    if analysis_data:
        for title, table in analysis_data.items():
            with st.expander(title.replace("analysis_", "").upper()):
                st.dataframe(table)

                # Add chart if 2 columns and second is numeric
                if table.shape[1] == 2 and pd.api.types.is_numeric_dtype(table.iloc[:, 1]):
                    st.bar_chart(table.set_index(table.columns[0]))
    else:
        st.info("No analysis data found in MongoDB. Run Spark jobs first.")

if __name__ == "__main__":
    main()
