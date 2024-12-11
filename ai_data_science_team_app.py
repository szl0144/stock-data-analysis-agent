# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI DATA SCIENCE TEAM
# ***

# Goal: Create an AI data science team automation that can clean, analyze, model, predict, and interpret data.


#region LIBRARIES

# * LIBRARIES

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from openai import OpenAI

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import sys

# DETECT PROJECT ROOT DIRECTORY
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]  
sys.path.append(str(project_root))

# from ai_data_science_team import forecast_team_logic_based

import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

#endregion 

#region STREAMLIT_SETUP

# * STREAMLIT APP SETUP ----

CHAT_LLM_OPTIONS = ["gpt-4o-mini", "gpt-4o"]

# * Page Setup

TITLE = "Your AI Data Science Team"

st.set_page_config(page_title=TITLE)
st.title(TITLE)

# * Instructions

INSTRUCTIONS = """
I'm a handy Data Science AI agent team that allows users to upload a CSV or Excel file and the file gets loaded in a temporary SQLite database as a table named "uploaded_data". You can ask me questions about the uploaded_data, ask me to perform aggregations on the uploaded_data, and make time series forecasts. I will report the results. 
"""

EXAMPLE_QUESTIONS = """
- What tables exist in the database?
- What are the first 10 rows in the uploaded_data table?
- Aggregate sales by month for each food item. Make a forecast for the next 12 months.
- Collect the data for FOODS_3_090. Aggregate sales by day. Make a forecast for the next 365 days.
- Aggregate sales by day. Make a forecast for the next 365 days.
- Aggregate sales by day for each food item. Make a forecast for the next 365 days. Do not include a legend in the forecast.
"""

st.markdown(INSTRUCTIONS)

with st.expander("Example Questions", expanded=False):
    st.write(EXAMPLE_QUESTIONS)
     
#endregion        


#region DATA_UPLOAD

# * Data Upload

st.sidebar.header(TITLE, divider=True)

# Add a checkbox for using demo data
st.sidebar.header("Upload Data (CSV or Excel)")
use_demo_data = st.sidebar.checkbox("Use demo data", value=False)

st.session_state["PATH_DB"] = None

if use_demo_data:
    # Load the demo data 
    demo_file_path = Path("data/churn_data.csv")
    if demo_file_path.exists():
        df = pd.read_csv(demo_file_path)
        st.session_state["df"] = df

        # Display demo data preview
        st.write("## Preview of Demo Data:")
        st.dataframe(df)
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

        st.session_state["df"] = df

        # Show uploaded data preview
        st.write("## Preview of Uploaded Data:")
        st.dataframe(df)
    else:
        st.info("Please upload a CSV or Excel file or use Demo Data to proceed.")

#endregion

# * OpenAI API Key

#region OPENAI_INPUTS

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("API Key", type="password", help="Your OpenAI API key is required for the app to function.")

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
    CHAT_LLM_OPTIONS,
    index=0
)

OPENAI_LLM = ChatOpenAI(
    model = model_option,
    api_key=st.session_state["OPENAI_API_KEY"]
)

llm = OPENAI_LLM

#endregion

#region STREAMLIT_MESSAGE_HISTORY

# * STREAMLIT MESSAGE HISTORY

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

#endregion

#region STREAMLIT_QUESTION_ANSWER

# * STREAMLIT QUESTION ANSWER

if "df" in st.session_state and st.session_state["df"] is not None and (question := st.chat_input("Enter your question here:", key="query_input")):
    
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()
    
    with st.spinner("Thinking..."):
        
        st.chat_message("human").write(question)
        msgs.add_user_message(question)
        
        ai_message = "TODO"
        
        st.write(ai_message)
        msgs.add_ai_message(ai_message)
    
#endregion
            