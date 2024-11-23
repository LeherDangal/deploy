

import streamlit as st
import pandas as pd
import os

# Function to load existing expenses
def load_expenses():
    if os.path.exists('expenses.csv'):
        return pd.read_csv('expenses.csv')
    else:
        return pd.DataFrame(columns=['Date', 'Amount', 'Category', 'Notes'])

# Function to save expenses
def save_expense(date, amount, category, notes):
    df = load_expenses()
    new_expense = pd.DataFrame({'Date': [date], 'Amount': [amount], 'Category': [category], 'Notes': [notes]})
    df = pd.concat([df, new_expense], ignore_index=True)
    df.to_csv('expenses.csv', index=False)

# Streamlit app
st.title("Personal Expense Tracker")

# Input form
with st.form(key='expense_form'):
    date = st.date_input("Date")
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    category = st.selectbox("Category", ["Food", "Rent", "Utilities", "Entertainment", "Other"])
    notes = st.text_input("Notes")
    submit_button = st.form_submit_button("Add Expense")

    if submit_button:
        save_expense(date, amount, category, notes)
        st.success("Expense added successfully!")

# Display stored expenses
st.subheader("Stored Expenses")
expenses = load_expenses()
st.dataframe(expenses)

import matplotlib.pyplot as plt
import plotly.express as px

# Function to visualize expenses
def visualize_expenses():
    df = load_expenses()
    if not df.empty:
        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Bar chart for expenses by category
        category_expenses = df.groupby('Category')['Amount'].sum().reset_index()
        fig = px.bar(category_expenses, x='Category', y='Amount', title='Expenses by Category')
        st.plotly_chart(fig)

        # Line chart for expenses over time
        time_expenses = df.groupby('Date')['Amount'].sum().reset_index()
        fig2 = px.line(time_expenses, x='Date', y='Amount', title='Expenses Over Time')
        st.plotly_chart(fig2)

# Call the visualization function
st.subheader("Expense Visualizations")
visualize_expenses()


from sklearn.linear_model import LinearRegression
import numpy as np

# Function to predict future expenses
def predict_expenses():
    df = load_expenses()
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

        # Prepare data for regression
        X = df[['Date_ordinal']]
        y = df['Amount']

        model = LinearRegression()
        model.fit(X, y)

        # Predict future expenses for the next 30 days
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
        future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_expenses = model.predict(future_dates_ordinal)

        # Display predictions
        prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Expense': future_expenses})
        st.subheader("Predicted Future Expenses")
        st.line_chart(prediction_df.set_index('Date'))
        st.write(prediction_df)

        # Budget input
budget = st.number_input("Set Monthly Budget", min_value=0.0, format="%.2f")

# Check if expenses exceed budget
total_expenses = expenses['Amount'].sum()
if total_expenses > budget:
    st.warning(f"You have exceeded your budget! Total expenses: {total_expenses:.2f}")
else:
    st.success(f"Total expenses: {total_expenses:.2f}. You are within your budget.")


    import base64

def download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

st.markdown(download_link(expenses, 'expenses.csv'), unsafe_allow_html=True)

