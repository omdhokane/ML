import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("customer_churn_model.pkl", "rb"))


st.title("Customer Churn Prediction")

age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure")
usage = st.number_input("Usage Frequency")
support = st.number_input("Support Calls")
delay = st.number_input("Payment Delay")
spend = st.number_input("Total Spend")
last = st.number_input("Last Interaction")

gender = st.selectbox("Gender", ["Male", "Female"])
sub = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

if st.button("Predict"):
    data = pd.DataFrame([[age, tenure, usage, support, delay, spend, last,
                          gender, sub, contract]],
                        columns=['Age', 'Tenure', 'Usage Frequency',
                                 'Support Calls', 'Payment Delay',
                                 'Total Spend', 'Last Interaction',
                                 'Gender', 'Subscription Type',
                                 'Contract Length'])
   

    pred = model.predict(data)[0]

    if pred == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will stay ✅")
