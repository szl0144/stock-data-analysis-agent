

from langchain_openai import ChatOpenAI

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolCall

from langgraph.graph import START, END, StateGraph

import os

from typing import TypedDict, Annotated, Sequence
import operator

import pandas as pd

import os
import yaml

from IPython.display import Image
from pprint import pprint

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


def data_cleaning_agent():
    
    # Define tools
    @tool(response_format="content")
    def show_summary_tool(df: dict) -> str:
        """
        Returns a summary of the DataFrame including shape, columns, and number of missing values.
        """
        df_pd = pd.DataFrame(df)
        if df_pd.empty:
            return "DataFrame is empty or not available."
        summary = f"Shape: {df_pd.shape}\n"
        summary += f"Columns: {list(df_pd.columns)}\n"
        summary += "Missing values per column:\n"
        summary += str(df_pd.isnull().sum()) + "\n"
        return summary
    
    @tool(response_format="content_and_artifact")
    def drop_missing_columns_tool(df: dict):
        """
        If a column has more than 50% missing values, drop it.
        Returns a message and the cleaned DataFrame as a dict.
        """
        df_pd = pd.DataFrame(df)
        original_shape = df_pd.shape
        # Threshold: columns that must have at least half non-null values
        thresh = df_pd.shape[0] / 2
        df_new = df_pd.dropna(axis=1, thresh=thresh)
        message = f"Dropped columns with >50% missing values. Original shape: {original_shape}, New shape: {df_new.shape}"
        return (message, df_new)

    # Bind tools to LLM
    tools = [show_summary_tool, drop_missing_columns_tool]
    llm_with_tools = llm.bind_tools(tools)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        data_raw: dict
        data_cleaned: dict
        next: str
    
    def react_router(state: GraphState):
        """
        The router decides whether to call a tool or finish.
        Logic:
        - If we have no summary yet, first call the summary tool.
        - Once we have a summary, check if we need to drop columns.
        - If drop action is needed, call drop_missing_columns_tool.
        - Otherwise, end the process.
        """
        messages = state["messages"]
        
        pprint(messages)
        
        if len(messages) == 0:
            return {"next": "call_tool"}
        last_message = messages[-1]
        if last_message.tool_calls:
            return {"next": "call_tool"}
        return {"next": END}
        
        # # If no tools have been called yet, call show_summary_tool first
        # if not any(isinstance(msg, ToolCall) for msg in messages):
        #     # We have no summary yet, so let's call the summary tool
        #     return "call_tool"
        
        # # If the last message was a tool call, check its result:
        # last_message = messages[-1]
        # if isinstance(last_message, ToolCall) and last_message.tool_name == "show_summary_tool":
        #     content = last_message.output.get("content", "")
        #     df_pd = pd.DataFrame(state["data_raw"])
        #     # Check columns with >50% missing
        #     missing_ratio = df_pd.isnull().mean()
        #     if any(missing_ratio > 0.5):
        #         # Need to drop columns
        #         return "call_tool"
        #     else:
        #         # No need for further cleaning
        #         return END
        
        # # If the last message was a drop tool call:
        # if isinstance(last_message, ToolCall) and last_message.tool_name == "drop_missing_columns_tool":
        #     # We have cleaned the data. The cleaning is done.
        #     return END
        
        # # Default fallback
        # return END
    
    def call_tool(state: GraphState):
        messages = state["messages"]
        
        if state.get("data_cleaned") is not None:
            df = state["data_cleaned"]
        else:
            df = state["data_raw"]
        
        response = llm_with_tools().invoke(messages, df)
        
        pprint(response)
        
        if response.tool_name == "show_summary_tool":
            return {
                "messages": messages + [response],
            }
        else:
            # Parse tuple into message and data
            return {
                "messages": messages + [response[0]],
                "data_cleaned": response[1].to_dict()
            }
            
    workflow = StateGraph(GraphState)
    
    workflow.add_node("react_router", react_router)
    workflow.add_node("call_tool", call_tool)
    
    workflow.add_edge(START, "react_router")
    workflow.add_conditional_edges("react_router", lambda x: x["next"], ["call_tool", END])
    workflow.add_edge("call_tool", "react_router")
    
    app = workflow.compile()
    
    return app
    

agent_data_cleaning = data_cleaning_agent()

Image(agent_data_cleaning.get_graph().draw_mermaid_png())

response = agent_data_cleaning.invoke({"data_raw": df.to_dict()})
    
    
    
        
    
    
    
    

