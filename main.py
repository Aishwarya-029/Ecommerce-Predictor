# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# 1. App title
st.set_page_config(page_title="Ecommerce Customer Spending Prediction")
st.title("🛍️ Ecommerce Customer Spending Prediction")
st.write("Predict **Year Amount Spent** using customer usage data.")

# 2. Load Dataset

@st.cache_data
def load_data():
    data = pd.read_csv("Ecommerce Customers")
    return data

data = load_data()


# 3. Train Model
X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
ln = LinearRegression()
ln.fit(X_train, y_train)

predictions = ln.predict(X_test)
mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)
rmse = np.sqrt(mse)

# 4. User Input for Prediction

st.subheader("🧾 Enter Customer Details")

avg_session = st.number_input("Average Session Length",min_value=20.0,max_value=40.0,value=33.0)
time_on_app = st.number_input("Time on App (minutes)",min_value=8.0,max_value=16.0,value=12.0)
time_on_web = st.number_input("Time on Website (minutes)",min_value=30.0,max_value=40.0,value=37.0)
membership_length = st.number_input("Length of Membership (years)",min_value=0.0,max_value=10.0,value=4.0)

if st.button("Predict Yearly Spending 💰"):
    user_data = np.array([[avg_session, time_on_app, time_on_web,membership_length]])
    prediction = ln.predict(user_data)[0]
    st.success(f"Prediction Yearly Amount Spent: **${prediction:}**")



















