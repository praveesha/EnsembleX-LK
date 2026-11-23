
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="EnsembleX LK", page_icon="🚗", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ensemblex_lk_outputs")
MODEL_PATH = os.path.join(OUT, "ensemblex_lk_best_pipeline.joblib")
RESULTS_PATH = os.path.join(OUT, "model_comparison.csv")
IMPORTANCE_PATH = os.path.join(OUT, "permutation_importance.csv")
META_PATH = os.path.join(OUT, "model_metadata.json")

st.title("🚗 EnsembleX LK")
st.caption("Ensemble learning for Sri Lankan used-vehicle asking-price estimation")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the Colab notebook and include ensemblex_lk_outputs.")
    st.stop()

model = joblib.load(MODEL_PATH)
results = pd.read_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else pd.DataFrame()
importance = pd.read_csv(IMPORTANCE_PATH) if os.path.exists(IMPORTANCE_PATH) else pd.DataFrame()

page = st.sidebar.radio("Navigate", [
    "Overview", "Model Comparison", "Feature Insights", "Price Predictor"
])

if page == "Overview":
    st.subheader("Project overview")
    st.write("A supervised regression experiment comparing bagging, boosting, stacking and a heterogeneous voting ensemble.")
    if not results.empty:
        best = results.sort_values("RMSE").iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Lowest-RMSE model", best["Model"])
        c2.metric("Test RMSE", f'{best["RMSE"]:.3f}')
        c3.metric("Test MAE", f'{best["MAE"]:.3f}')
        st.plotly_chart(px.bar(results.sort_values("RMSE"), x="Model", y="RMSE",
                               title="Held-out test RMSE (lower is better)"),
                        use_container_width=True)
    st.info("Output is in the original dataset price units. Verify the source's unit convention before interpreting as currency.")
    st.warning("This model estimates advertised listing prices, not verified transaction values.")

elif page == "Model Comparison":
    st.subheader("Model performance")
    if results.empty:
        st.warning("No model_comparison.csv found.")
    else:
        metric = st.selectbox("Metric", ["RMSE", "MAE", "R2"])
        ascending = metric != "R2"
        st.dataframe(results.sort_values(metric, ascending=ascending), use_container_width=True)
        st.plotly_chart(px.bar(results.sort_values(metric, ascending=ascending),
                               x="Model", y=metric, title=f"{metric} comparison"),
                        use_container_width=True)

elif page == "Feature Insights":
    st.subheader("Permutation feature importance")
    if not importance.empty:
        top = importance.sort_values("Importance_Mean").tail(15)
        st.plotly_chart(px.bar(top, x="Importance_Mean", y="Feature",
                               orientation="h", error_x="Importance_Std",
                               title="Top 15 features"), use_container_width=True)
        st.caption("Permutation importance is model-agnostic. Correlated features can share or mask importance.")
    else:
        st.warning("Feature importance output not found.")

elif page == "Price Predictor":
    st.subheader("Estimate an advertised vehicle price")
    st.caption("The prediction is in the original dataset's price units.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        brand = c1.text_input("Brand", "TOYOTA")
        model_name = c2.text_input("Model", "AQUA")
        town = c3.text_input("Town", "Colombo")
        yom = c1.number_input("Year of manufacture", min_value=1950, max_value=2026, value=2018)
        engine = c2.number_input("Engine capacity (cc)", min_value=0, value=1500)
        mileage = c3.number_input("Mileage (km)", min_value=0, value=65000)
        gear = c1.selectbox("Transmission", ["Automatic", "Manual"])
        fuel = c2.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"])
        leasing = c3.selectbox("Leasing", ["No Leasing", "Leasing"])
        condition = c1.selectbox("Condition", ["USED", "BRAND NEW", "RECONDITION"])
        ac = c2.selectbox("Air condition", ["Available", "Not Available"])
        steering = c3.selectbox("Power steering", ["Available", "Not Available"])
        mirror = c1.selectbox("Power mirror", ["Available", "Not Available"])
        window = c2.selectbox("Power window", ["Available", "Not Available"])
        listing_year = c3.number_input("Listing year", min_value=2000, max_value=2035, value=2025)
        month = c1.selectbox("Listing month", list(range(1, 13)), index=0)
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        row = pd.DataFrame([{
            "Brand": brand, "Model": model_name, "YOM": yom,
            "Engine (cc)": engine, "Gear": gear, "Fuel Type": fuel,
            "Millage(KM)": mileage, "Town": town, "Leasing": leasing,
            "Condition": condition, "AIR CONDITION": ac,
            "POWER STEERING": steering, "POWER MIRROR": mirror,
            "POWER WINDOW": window, "Listing_Year": listing_year,
            "Vehicle_Age": listing_year - yom, "Listing_Month": month
        }])
        prediction = float(model.predict(row)[0])
        st.success(f"Estimated asking price: {prediction:,.2f} (original dataset units)")
        st.caption("This is a model estimate, not a professional valuation or guaranteed sale price.")
