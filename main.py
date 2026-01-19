import streamlit as st
import pandas as pd
import joblib

#load trained model
model = joblib.load("njt_delay_model.pkl")

st.title("NJ Transit Delay Predictor")

st.write("Enter train details to predict the expected delay (in minutes).")

#user inputs
line = st.number_input("Line ID", min_value=0)
stop_sequence = st.number_input("Stop Number:", min_value=1)
from_station = st.number_input("From Station ID (station the train is coming from):", min_value=0)
to_station = st.number_input("To Station ID (station the train is going to):", min_value=0)

scheduled_hour = st.slider("Scheduled Hour (hour the train will run in military time):", 0, 23, 8)
scheduled_minute = st.slider("Scheduled Minute (exact minute within the hour):", 0, 59, 0)

day_of_week = st.selectbox(
    "Day of Week (0=Mon, 6=Sun):",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
)

month = st.slider("Month: (1=Jan, 12=Dec):", 1, 12, 3)
day = st.slider("Day of Month (calender day within the month):", 1, 31, 15)

#create input dataframe 
input_data = pd.DataFrame([{
    'line': line,
    'stop_sequence': stop_sequence,
    'from': from_station,
    'to': to_station,
    'hour': scheduled_hour,
    'day_of_week': day_of_week,
    'month': month,
    'day': day,
    'scheduled_hour': scheduled_hour,
    'scheduled_minute': scheduled_minute
}])

#predict
if st.button("Predict Delay"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Delay: {prediction:.2f} minutes")
