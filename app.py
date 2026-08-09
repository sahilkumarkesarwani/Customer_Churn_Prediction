import streamlit as st
import pickle
import pandas as pd 

st.title("Customer Churn Prediction")

# st.write("This app Predict the churn of a customer ")

df = pickle.load(open('df.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


billing = st.selectbox('Enter paperless billing facility', df['PaperlessBilling'].unique())

tenure = st.number_input("Enter how many month before customer joined", min_value= 0)

mode_of_pay = st.selectbox("Payment option given to customer", df['PaymentMethod'].unique())

senior_citizen = st.selectbox("Is customer a senior citizen", ['No', 'Yes'])

multi_line = st.selectbox('Is customer avail multiple line', ['No', 'Yes'])

internet_service = st.selectbox('Internet Service Connection', df['InternetService'].unique())
if internet_service == 'No':
    online_security = 'No internet service'
    online_backup = 'No internet service'
    device_protection = 'No internet service'
    tech_support = 'No internet service'
    streaming = 'No internet service'
else: 

    online_security = st.selectbox('Is online security given to customer', df['OnlineSecurity'].unique())

    online_backup = st.selectbox('Is online backup given to customer', df['OnlineBackup'].unique())

    device_protection = st.selectbox('Is device protection given to customer', df['DeviceProtection'].unique())

    tech_support = st.selectbox('Is tech support given to customer', df['TechSupport'].unique())

    streaming = st.selectbox('Is facility to stream given to customer', df['StreamingMovies'].unique())

contract = st.selectbox('Conract trem with customer', df['Contract'].unique())

monthly_charges = st.number_input("Enter Monthly Charges", min_value= 0.0)

if st.button("Predict"):

    input_data = pd.DataFrame([{
            'PaymentMethod' : mode_of_pay,
            'tenure' : tenure, 
            'PaperlessBilling' : billing, 
            'SeniorCitizen' : senior_citizen, 
            'MultipleLines' :  multi_line, 
            'InternetService' : internet_service, 
            'OnlineSecurity' :online_security,
            'OnlineBackup' : online_backup, 
            'DeviceProtection' : device_protection, 
            'TechSupport' : tech_support, 
            'StreamingMovies' : streaming, 
            'Contract' : contract, 
            'MonthlyCharges' : monthly_charges
        }])

    prediction = model.predict(input_data)

    if prediction == 1:
        st.write("Customer will Churn")
    else :
        st.write('Customer will not churn')

    probability = model.predict_proba(input_data)

    st.subheader('Probability of Churn')
    st.write(probability * 100.00)
    

