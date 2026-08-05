import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

st.sidebar.title("About")

st.sidebar.info(
    """
    **Model:** Random Forest Regressor

    **Dataset:** Ames Housing Dataset

    **R² Score:** 0.849

    **Developer:** Aryan Ramteke
    """
)
st.title("🏡 House Price Prediction using Machine Learning")
st.markdown("Predict residential house prices using a trained Random Forest Regression model.")

overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Ground Living Area (sq ft)", value=1500)
garage_cars = st.number_input("Garage Cars", value=2)
garage_area = st.number_input("Garage Area", value=500)
total_bsmt_sf = st.number_input("Total Basement Area", value=900)
first_flr_sf = st.number_input("First Floor Area", value=1200)

if st.button("🔍 Predict House Price", use_container_width=True):

    new_house = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Gr Liv Area': [gr_liv_area],
        'Garage Cars': [garage_cars],
        'Garage Area': [garage_area],
        'Total Bsmt SF': [total_bsmt_sf],
        '1st Flr SF': [first_flr_sf]
    })

    prediction = model.predict(new_house)

    st.success(f"🏷️ Estimated House Price: ${prediction[0]:,.2f}")
    st.balloons()
    
    st.divider()
    st.caption(
    "Built by Aryan Ramteke | IIT Bombay | Random Forest Regression | Streamlit"
    )