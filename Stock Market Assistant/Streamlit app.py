import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
from scipy.stats import linregress  # Import linregress function from scipy.stats
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# genai.configure(api_key = os.environ.get("GOOGLE_API_KEY"))


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert Stock news analyst. Understand the "Context" which are news articles and provide a detailed analysis to the "Question" given by the user. Make sure to provide all the details, if the answer is not present in the context, just say that the "Answer is not there in the provided context". Format the output in Markdown style.\n\n
    Context: {context}\n\n
    Question: {questions}\n\n
    
    Answer:  """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "questions"])
    chain = load_qa_chain(model, chain_type = "stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "questions": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply:\n\n", response["output_text"])
    
def load_sentiment_data(file_path):
    return pd.read_csv(file_path)

def plot_stock_data(stock_data, sentiment_data, selected_stock):
    fig_size= (10,5)
    plt.figure(figsize=fig_size)  # Set the same figure size for both plots
    plt.style.use('cyberpunk')
    
    selected_stock_data = stock_data[stock_data['Stock'] == selected_stock]
    plt.plot(selected_stock_data['Date'], selected_stock_data['Close'], marker='o', linestyle='-', label='Closing Price')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Closing Prices for {selected_stock}')
    plt.legend()
    
    # Display fewer dates on the x-axis
    xticks = plt.gca().get_xticks()
    plt.gca().set_xticks(xticks[::int(len(xticks)/5)])
    
    plt.xticks(rotation=45)
    st.pyplot(plt)


def convert_time_frame_to_date(selected_time_frame):
    today = pd.Timestamp.today().normalize()  # Get today's date as a Timestamp object
    if selected_time_frame == "Last Week":
        start_date = today - timedelta(days=7)
    elif selected_time_frame == "Last 2 Weeks":
        start_date = today - timedelta(days=14)
    elif selected_time_frame == "Last 3 Weeks":
        start_date = today - timedelta(days=21)
    elif selected_time_frame == "Last Month":
        # Calculate the start date of the last month
        start_date = today - relativedelta(months=1)
        start_date = start_date.replace(day=1)  # Set the day to the first day of the month
    else:
        start_date = None
    
    return start_date.to_pydatetime() if start_date else None


def plot_sentiment_data(sentiment_data, selected_stock, selected_time_frame):
    fig_size = (10,5)
    plt.figure(figsize=fig_size)  # Set the figure size
    plt.style.use('cyberpunk')  # Use the cyberpunk style
    
    start_date = convert_time_frame_to_date(selected_time_frame)
    
    if start_date:
        selected_sentiment_data = sentiment_data[(sentiment_data['Stock'] == selected_stock) & (pd.to_datetime(sentiment_data['Date']) >= start_date)]
        
        if not selected_sentiment_data.empty:
            # Calculate the counts of each sentiment category
            sentiment_counts = selected_sentiment_data['new_sentiment'].value_counts(normalize=True)
            
            # Plot the percentages of each sentiment category
            plt.bar(sentiment_counts.index, sentiment_counts.values * 100, color=['red', 'white', 'green'])
            
            plt.xlabel('Sentiment')
            plt.ylabel('Percentage')
            plt.title(f'Predicted Sentiment for {selected_stock}')
            
            # Add legends for each sentiment category
            plt.legend(['Positive', 'Neutral', 'Negative'])
        else:
            plt.text(0.5, 0.5, "No data available for the selected time frame.", horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, "Invalid time frame selected.", horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.xticks(rotation=45)
    st.pyplot(plt)

def display_summary(sentiment_data, selected_stock, selected_time_frame):
    if selected_stock in sentiment_data['Stock'].unique():
        selected_stock_data = sentiment_data[(sentiment_data['Stock'] == selected_stock) & (sentiment_data['Date'] >= selected_time_frame)]
        if not selected_stock_data.empty:
            st.subheader(f"Summary for {selected_stock} for {selected_time_frame}")
            for summary in selected_stock_data['Summary']:
                st.text(summary)
        else:
            st.write("No summary data available for the selected time frame.")
    else:
        st.write(f"No sentiment data available for {selected_stock}.")



def load_summary_data(file_path):
    with open(file_path, 'r') as file:
        summary_data = json.load(file)
    return summary_data

def convert_time_frame_to_date(selected_time_frame):
    today = datetime.today().replace(day=1)  # Start of the current month
    if selected_time_frame == "Last Month":
        start_date = today - timedelta(days=1)  # End of the previous month
        start_date = start_date.replace(day=1)  # Start of the previous month
    else:
        start_date = None
    return start_date

def filter_summaries(summary_data, selected_stock, selected_time_frame):
    filtered_summaries = []
    start_date = convert_time_frame_to_date(selected_time_frame)
    if start_date:
        for summary in summary_data:
            summary_date = datetime.strptime(summary['date'], '%Y-%m-%d')
            if summary['company'] == selected_stock and start_date <= summary_date:
                filtered_summaries.append(summary)
    return filtered_summaries

def display_summaries(filtered_summaries):
    if filtered_summaries:
        st.subheader("Summaries:")
        for summary in filtered_summaries:
            st.text(f"Date: {summary['date']}")
            st.text(f"Title: {summary['title']}")
            st.text(f"Summary: {summary['summary']}")
            st.markdown("---")
    else:
        st.write("No summaries available for the selected stock and time frame.")

def load_summary_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            summary_data = json.load(file)
        return summary_data
    except UnicodeDecodeError as e:
        # If there's an encoding error, print the error message and return None
        print(f"UnicodeDecodeError: {e}")
        return None

def main():
    summary_data = load_summary_data("D:/MSAI Lectures and Documents/Deep Learning with NLP/Streamlit Dev/one_month_news_summary.json")
    if summary_data:
        print("Summary data loaded successfully!")
    # Now you can use the summary data in your application
    else:
        print("Failed to load summary data. Check the file path and encoding.")
    # Load the sentiment data
    sentiment_data = load_sentiment_data("D:/MSAI Lectures and Documents/Deep Learning with NLP/Streamlit Dev/predictions_test_set.csv")
    st.set_page_config(page_title="Market Sentiment - Summarizer App", page_icon=":money_with_wings:", layout="wide")
    st.header("Market Sentiment - Summarizer App :money_with_wings:")
    
     # Set the figure size and aspect ratio here
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    with st.sidebar:
        st.subheader("Chat Option:")
        user_question = st.text_input("Ask a question:")

    # Chat functionality here
    if user_question:
        response = user_input(user_question)
        st.write(response)

    with st.sidebar:
        st.subheader("Select one of the stocks from the following options:")
        selected_stock = st.radio("Stock", options=["AAPL", "NVDA", "META", "GOOGL", "MSFT"])

        st.subheader("Select the time frame to view the summarized insights:")
        selected_time_frame = st.radio("Time Frame", options=["Last Week", "Last 2 Weeks", "Last 3 Weeks", "Last Month"])
        filtered_summaries = filter_summaries(summary_data, selected_stock, selected_time_frame)
    # Load and display the graphs
    col1, col2 = st.columns(2)

    with col1:
        # Load and plot sentiment data
        sentiment_data = load_sentiment_data("predictions_test_set.csv")
        plot_sentiment_data(sentiment_data, selected_stock, selected_time_frame)

    with col2:
        # Load and plot stock data
        stock_data = pd.read_csv("stock_data.csv")
        plot_stock_data(stock_data, sentiment_data, selected_stock)

    # Display summaries below the graphs
    
    display_summaries(filtered_summaries)

if __name__ == '__main__':
    main()