

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph

import os

from typing import TypedDict, Annotated, Sequence
import operator

import pandas as pd
import sqlalchemy as sql
import plotly.io as pio

import os
import yaml


from ai_data_science_team.tools import PythonOutputParser


# Setup

# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

MODEL    = "gpt-4o-mini"
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), "logs/")


# Load Data

df = pd.read_csv("data/churn_data.csv")

# Create LLM

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    verbose=True
)



# Data Cleaning Agent

def data_cleaning_agent(df):
    
    if df is pd.core.frame.DataFrame:
        df_dict = df.to_dict()
    else:
        df_dict = df
    
    @tool(response_format="content")
    def show_summary_tool(df) -> pd.DataFrame:
        """
        Returns a summary of the DataFrame including shape, columns, and number of missing values.
        """
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return "DataFrame is empty or not available."
        summary = f"Shape: {df.shape}\n"
        summary += f"Columns: {list(df.columns)}\n"
        summary += "Missing values per column:\n"
        summary += str(df.isnull().sum()) + "\n"
        
        return summary
    
    @tool(response_format="content_and_artifact")
    def drop_missing_columns_tool(df) -> pd.DataFrame:
        """
        If column has more than 50% missing values, drop it.
        """
        df = pd.DataFrame(df)
        original_shape = df.shape
        df_new = df.dropna(axis=1, thresh=df.shape[0] // 2)
        
        message=f"Dropped missing rows. Original shape: {original_shape}, New shape: {df_new.shape}"
        
        return (message, df_new)

    
    tools = [show_summary_tool, drop_missing_columns_tool]
    
    llm_with_tools = llm.bind_tools(tools)
    
    
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        data_raw = df_dict
        data_cleaned: dict
        
    def react_router(state: GraphState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    
    
    
    
    
    
        
    
    
    
    

