import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Classification of Flower Species")

petal_length = st.number_input("Enter Petal_Length")
petal_width = st.number_input("Enter Petal_Width")
sepal_length = st.number_input("Enter Sepal_Length")
sepal_width = st.number_input("Enter Sepal_Width")

if st.button("Submit Details"):

    new_data = pd.DataFrame({'sepal_length': sepal_length, 'sepal_width': sepal_width,
                             'petal_length': petal_length, 'petal_width': petal_width}, index=[0])

    if petal_length < 2:
        st.write("Iris Setosa")
    else:
        predicted_species = model.predict(new_data)
        st.write(f"{predicted_species}")
