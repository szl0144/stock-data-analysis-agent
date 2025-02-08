# BUSINESS SCIENCE
# Exploratory Data Analysis (EDA) Copilot App
# -----------------------

# This app helps you search for data and produces exploratory analysis reports.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI

import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import asyncio

from pathlib import Path
import sys

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team.agents import DataLoaderToolsAgent
from ai_data_science_team.ds_agents import EDAToolsAgent

# * APP INPUTS ----

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']

TITLE = "Your Exploratory Data Analysis (EDA) Copilot"

# * STREAMLIT APP SETUP ----

st.set_page_config(page_title=TITLE, page_icon="📊")
st.title("📊 " + TITLE)

st.markdown("""
Welcome to the EDA Copilot. This AI agent is designed to help you find and load data 
and return exploratory data analysis reports that can be used to understand the data 
prior to other analysis (e.g. modeling, feature engineering, etc).
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What data is available in my current directory?
        - Load the customer churn data.
        - What tools are available for exploratory data analysis?
        - Generate a missing data report.
        """
    )

# * STREAMLIT EXCEL/CSV UPLOAD (REPLACING DATABASE WITH SESSION STORAGE) ----

st.sidebar.header("EDA Copilot: Data Upload/Selection", divider=True)

# Add a checkbox for using demo data
st.sidebar.header("Upload Data (CSV or Excel)")
use_demo_data = st.sidebar.checkbox("Use demo data", value=False)

# Initialize session state "DATA_RAW" if not present
if "DATA_RAW" not in st.session_state:
    st.session_state["DATA_RAW"] = None

if use_demo_data:
    # Load the demo data from 'data/churn_data.csv'
    demo_file_path = Path("data/churn_data.csv")
    if demo_file_path.exists():
        df = pd.read_csv(demo_file_path)
        file_name = "churn_data"

        # Store DataFrame in session state
        st.session_state["DATA_RAW"] = df.copy()

        # Display demo data preview
        st.write(f"## Preview of {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"])
    else:
        st.error(f"Demo data file not found at {demo_file_path}. Please ensure it exists.")

else:
    # Allow user to upload CSV or Excel file
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Read the uploaded file into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            
        # Store DataFrame in session state
        st.session_state["DATA_RAW"] = df.copy()
        file_name = Path(uploaded_file.name).stem

        # Show uploaded data preview
        st.write(f"## Preview of {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"])
    else:
        st.info("Please upload a CSV or Excel file or Use Demo Data to proceed.")


# * OpenAI API Key

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key", 
    type="password", 
    help="Your OpenAI API key is required for the app to function."
)

# Test OpenAI API Key
if st.session_state["OPENAI_API_KEY"]:
    # Set the API key for OpenAI
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
    
    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()

# * OpenAI Model Selection

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    MODEL_LIST,
    index=0
)

OPENAI_LLM = ChatOpenAI(
    model=model_option,
    api_key=st.session_state["OPENAI_API_KEY"]
)

llm = OPENAI_LLM

# * STREAMLIT MESSAGE HANDLING

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Initialize plot storage in session state
if "plots" not in st.session_state:
    st.session_state.plots = []

# Initialize dataframe storage in session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Function to display chat messages including Plotly charts and dataframes
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index])
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg.content)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()



# html_code = """
# <!DOCTYPE html>
# <html>
# <head>
#   <meta charset="utf-8">
#   <title>Iframe Full-Screen Toggle</title>
#   <style>
#     body, html {
#       margin: 0;
#       padding: 0;
#       height: 100%;
#     }
#     #iframe-container {
#       position: relative;
#       width: 100%;
#       height: 600px;
#     }
#     #myIframe {
#       width: 100%;
#       height: 100%;
#       border: none;
#     }
#     #fullscreen-btn {
#       position: absolute;
#       top: 10px;
#       right: 10px;
#       z-index: 1000;
#       padding: 8px 12px;
#       background-color: #007bff;
#       color: white;
#       border: none;
#       border-radius: 4px;
#       cursor: pointer;
#     }
#   </style>
# </head>
# <body>
#   <div id="iframe-container">
#     <button id="fullscreen-btn" onclick="toggleFullscreen()">Full Screen</button>
#     <iframe id="myIframe" src="https://www.wikipedia.org" allowfullscreen></iframe>
#   </div>
#   <script>
#     function toggleFullscreen() {
#       var container = document.getElementById("iframe-container");
#       if (!document.fullscreenElement) {
#         container.requestFullscreen().catch(err => {
#           alert("Error attempting to enable full-screen mode: " + err.message);
#         });
#         document.getElementById("fullscreen-btn").innerText = "Exit Full Screen";
#       } else {
#         document.exitFullscreen();
#         document.getElementById("fullscreen-btn").innerText = "Full Screen";
#       }
#     }
    
#     // Optional: Listen for fullscreen change events to update button text if the user exits full-screen via ESC.
#     document.addEventListener('fullscreenchange', () => {
#       if (!document.fullscreenElement) {
#         document.getElementById("fullscreen-btn").innerText = "Full Screen";
#       }
#     });
#   </script>
# </body>
# </html>
# """

# # Render the HTML component in Streamlit. Adjust the height if necessary.
# components.html(html_code, height=620)

    

