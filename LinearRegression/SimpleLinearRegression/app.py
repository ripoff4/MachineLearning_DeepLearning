import pickle
import streamlit as st
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Height vs Weight Calculation")

height = st.number_input('Enter Height (cm)', min_value=10, max_value=250)

if st.button("Submit Height"):
    df = pd.DataFrame({'Height(Inches)': height}, index=[0])
    scaled_df = scaler.transform(df)
    predicition = model.predict(scaled_df)
    st.write(f"The Weight of a man of Height {height} is {predicition}")
