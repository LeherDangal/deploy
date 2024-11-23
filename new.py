import streamlit as st
import pandas as pd
import os
from datetime import timedelta
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import base64


# File path for storing expenses
EXPENSES_FILE = 'expenses.csv'


# Utility Functions
def load_expenses() -> pd.DataFrame:
    """
    Load expenses from a CSV file. Returns an empty DataFrame if the file doesn't exist.
    """
    if os.path.exists(EXPENSES_FILE):
        return pd.read_csv(EXPENSES_FILE)
    return pd.DataFrame(columns=['Date', 'Amount', 'Category', 'Notes'])


def save_expenses(df: pd.DataFrame):
    """
    Save the expenses DataFrame to a CSV file.
    """
    df.to_csv(EXPENSES_FILE, index=False)


def add_expense(date: pd.Timestamp, amount: float, category: str, notes: str):
    """
    Add a new expense to the dataset and save it to the CSV file.
    """
    df = load_expenses()
    new_entry = pd.DataFrame({'Date': [date], 'Amount': [amount], 'Category': [category], 'Notes': [notes]})
    updated_df = pd.concat([df, new_entry], ignore_index=True)
    save_expenses(updated_df)


def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Generate a CSV download link for the DataFrame.
    """
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'


# Visualization Functions
def visualize_expenses(df: pd.DataFrame):
    """
    Create and display visualizations for expenses.
    """
    if df.empty:
        st.info("No expenses to visualize yet.")
        return

    # Convert 'Date' to datetime for plotting
    df['Date'] = pd.to_datetime(df['Date'])

    # Bar Chart: Expenses by Category
    category_summary = df.groupby('Category')['Amount'].sum().reset_index()
    fig_category = px.bar(category_summary, x='Category', y='Amount', title='Expenses by Category', text='Amount')
    fig_category.update_layout(xaxis_title="Category", yaxis_title="Total Amount")
    st.plotly_chart(fig_category)

    # Line Chart: Expenses Over Time
    time_series = df.groupby('Date')['Amount'].sum().reset_index()
    fig_time = px.line(time_series, x='Date', y='Amount', title='Expenses Over Time', markers=True)
    fig_time.update_layout(xaxis_title="Date", yaxis_title="Total Amount")
    st.plotly_chart(fig_time)


# Prediction Function
def predict_expenses(df: pd.DataFrame):
    """
    Predict future expenses using a linear regression model.
    """
    if df.empty:
        st.info("Not enough data for prediction.")
        return

    # Prepare data for modeling
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

    X = df[['Date_ordinal']]
    y = df['Amount']

    model = LinearRegression()
    model.fit(X, y)

    # Predict future expenses for the next 30 days
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    predictions = model.predict(future_ordinal)

    # Display predictions
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Expense': predictions})
    st.subheader("Predicted Future Expenses")
    st.line_chart(prediction_df.set_index('Date'))
    st.write(prediction_df)


# Streamlit App
def main():
    st.title("Personal Expense Tracker")
    st.markdown(
        "Track your daily expenses, visualize spending trends, and predict future expenses. Manage your budget efficiently!"
    )

    # Form: Add Expense
    with st.form(key='expense_form'):
        st.subheader("Add a New Expense")
        date = st.date_input("Date")
        amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        category = st.selectbox("Category", ["Food", "Rent", "Utilities", "Entertainment", "Other"])
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add Expense")

        if submitted:
            if amount <= 0:
                st.error("Amount must be greater than zero.")
            else:
                add_expense(date, amount, category, notes)
                st.success("Expense added successfully!")

    # Load and Display Expenses
    expenses = load_expenses()
    st.subheader("Stored Expenses")
    st.dataframe(expenses)

    # Visualizations
    st.subheader("Visualize Your Expenses")
    visualize_expenses(expenses)

    # Predictions
    st.subheader("Expense Predictions")
    predict_expenses(expenses)

    # Budgeting
    st.subheader("Set a Monthly Budget")
    budget = st.number_input("Monthly Budget", min_value=0.01, format="%.2f")
    total_expenses = expenses['Amount'].sum()
    if total_expenses > budget:
        st.warning(f"Warning: You have exceeded your budget! Total expenses: {total_expenses:.2f}")
    else:
        st.success(f"You are within your budget! Total expenses: {total_expenses:.2f}")

    # Download Link
    st.subheader("Download Your Data")
    st.markdown(create_download_link(expenses, 'expenses.csv'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
