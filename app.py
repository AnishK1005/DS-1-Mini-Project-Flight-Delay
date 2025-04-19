import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")
le_weather = joblib.load("label_encoder.pkl")
weather_classes = le_weather.classes_

def predict_delay_chance(dep_hour, distance, air_time, day_of_week, weather_str):
    try:
        weather_encoded = le_weather.transform([weather_str.upper()])[0]
    except ValueError:
        raise ValueError(f"Unknown weather condition: {weather_str}")
    
    input_data = pd.DataFrame([{
        'DAY_OF_WEEK': day_of_week,
        'DEP_HOUR': dep_hour,
        'DISTANCE': distance,
        'AIR_TIME': air_time,
        'Weather_Encoded': weather_encoded
    }])
    proba = model.predict_proba(input_data)[0][1]
    return round(proba * 100, 2)

st.set_page_config(page_title="Flight Delay Predictor", layout="centered")
st.title("Flight Delay Probability Checker")

with st.form("prediction_form"):
    st.subheader("Enter Flight Details:")
    dep_hour = st.slider("Departure Hour", 0, 23, 14)
    distance = st.number_input("Distance (miles)", 50, 5000, 1200)
    air_time = st.number_input("Air Time (minutes)", 30, 600, 180)
    day_of_week = st.selectbox("Day of the Week", options=[
        (1, "Monday"), (2, "Tuesday"), (3, "Wednesday"),
        (4, "Thursday"), (5, "Friday"), (6, "Saturday"), (7, "Sunday")
    ], format_func=lambda x: x[1])[0]
    weather_str = st.selectbox("Weather Condition", weather_classes)
    submitted = st.form_submit_button("Predict Delay Chance")

    if submitted:
        try:
            result = predict_delay_chance(dep_hour, distance, air_time, day_of_week, weather_str)
            if result < 20:
                message = "Low risk of delay. Your flight is likely to depart on time."
            elif result < 40:
                message = "Moderate to low risk. Some potential for delay, but generally on schedule."
            elif result < 60:
                message = "Moderate risk. Delay is possible, check for updates closer to departure."
            elif result < 80:
                message = "High risk of delay. Consider planning for potential changes."
            else:
                message = "Very high risk of delay. Expect disruptions and monitor flight status closely."
            st.success(f"Predicted Delay Chance: **{result}%**\n\n{message}")
        except ValueError as e:
            st.error(str(e))
