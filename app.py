import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dropbox
from io import BytesIO

# Dropbox API setup
ACCESS_TOKEN = 'sl.B-0EiyM4OCa0YqhyLGL9PdKmW3w4K2v0WpNb9-2k_6c8KzByDW4vkaid-cZXOs8ftDyQPXo5_dtMmDVBNouZzWOTh95fhOY_rXAhoUSMi7Z47aYldUNrWggLV9AoK2y77tfNCAj9T8kt'

# Dropbox file path
DROPBOX_FILE_PATH = 'expenseTracker_advanced/daily_expenses.csv'

# Dropbox connection
def connect_to_dropbox():
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    return dbx

# Upload file to Dropbox
def upload_file_to_dropbox(file_path):
    dbx = connect_to_dropbox()
    with open(file_path, 'rb') as f:
        dbx.files_upload(f.read(), DROPBOX_FILE_PATH, mode=dropbox.files.WriteMode.overwrite)

# Download file from Dropbox
def download_file_from_dropbox():
    dbx = connect_to_dropbox()
    try:
        metadata, res = dbx.files_download(path=DROPBOX_FILE_PATH)
        return res.content
    except dropbox.exceptions.ApiError as e:
        st.warning("No file found in Dropbox, starting with a new file.")
        return None

# Handling file paths properly for Streamlit Cloud
EXPENSE_FILE = "daily_expenses.csv"

api_key = "AIzaSyCzdCOyd-7os-SRgbEolxtwEEgYYkjKpsM"

# Initialize expense file if it doesn't exist
def initialize_file():
    # Check if the file exists locally
    if not os.path.exists(EXPENSE_FILE):
        # Try downloading from Dropbox
        file_data = download_file_from_dropbox()
        if file_data:
            with open(EXPENSE_FILE, 'wb') as f:
                f.write(file_data)
        else:
            # Create new file if not present in Dropbox
            df = pd.DataFrame(columns=['Date', 'Amount (INR)', 'Category', 'Description'])
            df.to_csv(EXPENSE_FILE, index=False)
            upload_file_to_dropbox(EXPENSE_FILE)

# Add new expense function
def add_expense(date, amount, category, description):
    new_expense = pd.DataFrame({
        'Date': [date], 
        'Amount (INR)': [amount], 
        'Category': [category], 
        'Description': [description]
    })
    new_expense.to_csv(EXPENSE_FILE, mode='a', header=False, index=False)

    # Upload updated file to Dropbox
    upload_file_to_dropbox(EXPENSE_FILE)

# Load expense data
def load_data():
    try:
        if os.path.exists(EXPENSE_FILE):
            return pd.read_csv(EXPENSE_FILE)
        else:
            return pd.DataFrame(columns=['Date', 'Amount (INR)', 'Category', 'Description'])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['Date', 'Amount (INR)', 'Category', 'Description'])

# Delete expense function
def delete_expense(date, amount, category):
    df = load_data()
    df = df[~((df['Date'] == date) & (df['Amount (INR)'] == amount) & (df['Category'] == category))]
    df.to_csv(EXPENSE_FILE, index=False)

    # Upload updated file to Dropbox
    upload_file_to_dropbox(EXPENSE_FILE)

# Plotting expense data (Bar and Line)
def plot_expense_data(df, chart_type='Bar'):
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Remove time
    if chart_type == 'Bar':
        fig = px.bar(df, x='Date', y='Amount (INR)', color='Category', title="Expenses Over Time")
    else:
        fig = px.line(df, x='Date', y='Amount (INR)', color='Category', title="Expenses Over Time")
    st.plotly_chart(fig)

# LLM-based RAG approach for CSV analysis
def retrieve_relevant_chunks(df, query, vectorizer, top_n=3):
    text_data = df.to_string(index=False)
    chunks = text_data.split('\n')

    # Vectorizing text data
    vectors = vectorizer.fit_transform(chunks).toarray()
    query_vec = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_vec, vectors).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

def generate_response(retrieved_chunks, user_query, api_key):
    prompt = f'''
    You are a chatbot that answers questions based on the following documents, where this data is about the daily expense it has the data like amount, date and category.:
    set default currency to indian rupee.
    {retrieved_chunks}

    User Question: "{user_query}"

    Provide a coherent and contextually relevant answer.
    '''
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
        response.raise_for_status()
        generated_content = response.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "No response available.")
        return generated_content.strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating response: {e}")
        return "Error in generating response."

# Streamlit App Layout
def main():
    st.set_page_config(page_title="Daily Expense Tracker", page_icon="https://static.vecteezy.com/system/resources/previews/012/766/555/original/expenses-icon-style-vector.jpg")

    st.title("Daily Expense Tracker")

    # Initialize file if it doesn't exist
    initialize_file()

    # Load expense data
    data = load_data()

    # Sidebar: Display the entire DataFrame (No Refresh Button)
    st.sidebar.header("All Expenses")
    if not data.empty:
        st.sidebar.dataframe(data)
    else:
        st.sidebar.write("No expenses recorded.")

    # Sidebar: Delete Expense
    st.sidebar.header("Delete Expense")
    if not data.empty:
        # Create unique keys for each expense (Date + Amount + Category)
        data['Expense_ID'] = data.apply(lambda row: f"{row['Date']} | {row['Amount (INR)']} | {row['Category']}", axis=1)

        # Select the expense to delete
        expense_to_delete = st.sidebar.selectbox("Select an expense to delete:", data['Expense_ID'])

        if st.sidebar.button("Delete Expense"):
            date, amount, category = expense_to_delete.split(' | ')
            delete_expense(date, float(amount), category)
            st.sidebar.success(f"Expense on {date} for {amount} INR in category '{category}' deleted successfully.")
    else:
        st.sidebar.write("No expenses available to delete.")

    # Layout: Two columns for Add Expense and Summary Insights
    col1, col2 = st.columns([1, 1])  # Make columns equally sized

    # Add new expense section in the first column
    with col1:
        st.header("Add New Expense")
        with st.form("expense_form"):
            date = st.date_input("Select Date")
            amount = st.number_input("Amount (INR)", min_value=0.0, format="%.2f")
            category = st.selectbox("Category", ["Grocery", "Snacks", "Outside Eatings", "Online Orders", "Others"])
            description = st.text_area("Description (Optional)")
            submitted = st.form_submit_button("Add Expense")

        if submitted:
            if amount > 0:
                add_expense(date, amount, category, description)
                st.success("Expense added successfully!")
            else:
                st.error("Amount should be greater than zero.")

    # Summary Insights in the second column
    with col2:
        st.header("Summary Insights")
        if not data.empty:
            total_expenses = data['Amount (INR)'].sum()
            top_category = data.groupby('Category')['Amount (INR)'].sum().idxmax()
            most_expensive_date = data.groupby('Date')['Amount (INR)'].sum().idxmax()

            st.metric(label="Total Expenses (INR)", value=f"{total_expenses:,.2f}")
            st.metric(label="Highest Spending Category", value=top_category)
            st.metric(label="Highest Spending Day", value=most_expensive_date)
        else:
            st.write("No data available.")

    # View analysis section
    st.header("View Expense Analysis")
    chart_type = st.radio("Choose chart type:", ["Bar", "Line"])
    if st.button("View Analysis"):
        if not data.empty:
            plot_expense_data(data, chart_type=chart_type)
        else:
            st.warning("No expenses to analyze yet!")

    # LLM Query Section
    st.header("Ask Questions About Your Expenses")
    
    # Add the input and send button inside a form
    with st.form("query_form"):
        user_query = st.text_input("Enter your query related to expenses:")
        submitted_query = st.form_submit_button("Send")  # Add a "Send" button
    
    # Check if the query is submitted and process it
    if submitted_query and user_query:
        vectorizer = TfidfVectorizer(stop_words='english')
        retrieved_chunks = retrieve_relevant_chunks(data, user_query, vectorizer)
        
        with st.spinner('Generating response...'):
            response = generate_response(retrieved_chunks, user_query, api_key)
        
        st.write("Response:", response)


if __name__ == "__main__":
    main()
