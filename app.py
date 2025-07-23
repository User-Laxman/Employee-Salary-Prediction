import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Loading the model, encoder, and scaler
model = joblib.load('random_forest_model.pkl')
encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Defining feature options based on the dataset
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Others']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 
                      'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 
                      'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Others']
relationship_options = ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 
                         'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 
                         'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 
                         'Peru', 'France', 'Greece', 'Ecuador', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Thailand', 
                         'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland', 'Holand-Netherlands', 'Others']

# Streamlit app layout
st.title("Adult Income Prediction")
st.write("Enter the details below to predict if an individual's income exceeds $50K.")

# Creating input fields
age = st.number_input("Age", min_value=17, max_value=75, value=30)
workclass = st.selectbox("Workclass", workclass_options)
fnlwgt = st.number_input("Final Weight", min_value=0, value=100000)
educational_num = st.number_input("Educational Number", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Marital Status", marital_status_options)
occupation = st.selectbox("Occupation", occupation_options)
relationship = st.selectbox("Relationship", relationship_options)
race = st.selectbox("Race", race_options)
gender = st.selectbox("Gender", gender_options)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=0, value=40)
native_country = st.selectbox("Native Country", native_country_options)

# Predicting when the user clicks the button
if st.button("Predict Income"):
    # Creating input data
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'educational-num': [educational_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    # Encoding categorical variables
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    for col in categorical_cols:
        input_data[col] = encoder.fit_transform(input_data[col].astype(str))

    # Scaling numerical features
    input_data_scaled = scaler.transform(input_data)

    # Making prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[0]

    # Displaying result
    result = "> $50K" if prediction[0] == 1 else "<= $50K"
    confidence = prediction_proba[prediction[0]] * 100
    st.success(f"Predicted Income: {result}")
    st.write(f"Confidence: {confidence:.2f}%")