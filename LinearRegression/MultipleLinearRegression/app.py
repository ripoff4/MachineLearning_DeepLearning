import streamlit as st
import pickle
import pandas as pd

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

st.title('Housing Price Prediction')

options = [1, 0]
furniting_options = ['furnished', 'semi-furnished', 'unfurnished']
area = st.number_input("Enter the area (in square feet):")
bedrooms = st.number_input("Enter no of bedrooms:", min_value=0, max_value=5)
bathrooms = st.number_input("Enter no of bathrooms:", min_value=0, max_value=5)
stories = st.number_input("Enter no of stories:", min_value=0, max_value=5)
mainroad = st.selectbox(label="select mainroad yes 1 or no 0 ",
                        options=options, index=0)
guestroom = st.selectbox(label="select guestroom yes 1 or no 0 ",
                         options=options, index=0)
basement = st.selectbox(label="select basement yes 1 or no 0 ",
                        options=options, index=0)
hotwaterheating = st.selectbox(
    label="select hotwater yes 1 or no 0 ", options=options, index=0)
airconditioning = st.selectbox(
    label="select AC yes 1 or no 0 ", options=options, index=0)
parking = st.number_input("Enter no of parkinglot:", min_value=0, max_value=5)
prefarea = st.selectbox(label="select prefacearea yes 1 or no 0 ",
                        options=options, index=0)
furnishingstatus = st.selectbox(
    label="select furniture status", options=furniting_options, index=0)

if st.button("Enter details"):
    df = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'stories': stories, 'mainroad': mainroad,
                       'guestroom': guestroom, 'basement': basement, 'hotwaterheating': hotwaterheating,
                       'airconditioning': airconditioning, 'parking': parking, 'prefarea': prefarea, 'furnishingstatus': furnishingstatus}, index=[0])
    furniture_status = onehot_encoder.transform(
        df[['furnishingstatus']]).toarray()
    furniture_df = pd.DataFrame(
        furniture_status, columns=onehot_encoder.get_feature_names_out(['furnishingstatus']))
    df = pd.concat(
        [df.drop(['furnishingstatus'], axis=1), furniture_df], axis=1)
    df[['area']] = scaler.transform(df[['area']])
    prediction = model.predict(df)
    st.write(f"The predicted house value is {prediction}")
