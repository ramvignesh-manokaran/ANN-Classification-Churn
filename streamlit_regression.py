import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Load the pre-trained model and preprocessing objects
model = tf.keras.models.load_model('salary_regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title("Estimated Salary Prediction")

# Input features
st.header("Input Features")

geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider("Tenure (in months)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],  # Transform
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode the 'Geography' feature
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Concatenate the encoded 'Geography' with the input DataFrame
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input features
input_df_scaled = scaler.transform(input_df)

# Predict the churn probability
prediction = model.predict(input_df_scaled)
predicted_salary = prediction[0][0]  # This will give the churn probability

# Display the prediction result
st.write(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")