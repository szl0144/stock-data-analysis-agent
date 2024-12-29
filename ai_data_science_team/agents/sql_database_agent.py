

from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import io
import pandas as pd
import sqlalchemy as sql

from ai_data_science_team.tools.parsers import PythonOutputParser, SQLOutputParser  
from ai_data_science_team.tools.metadata import get_database_metadata

# Setup
AGENT_NAME = "sql_database_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


def make_sql_database_agent(model, connection, log=False, log_path=None, overwrite = True, human_in_the_loop=False):
    
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection
    
    llm = model
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        # data_raw: dict
        data_sql: dict
        all_sql_database_summary: str
        # data_cleaner_function: str
        # data_cleaner_function_path: str
        # data_cleaner_function_name: str
        # data_cleaner_error: str
        max_retries: int
        retry_count: int
    
    def recommend_sql_steps(state: GraphState):
        
        print("---SQL DATABASE AGENT---")
        print("    * RECOMMEND CLEANING STEPS")
        
        
        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a SQL Database Expert. Given the following information about the SQL database, 
            recommend a series of numbered steps to take to collect the data and process it according to user instructions. 
            The steps should be tailored to the SQL database characteristics and should be helpful 
            for a sql database agent that will write the SQL code.
            
            IMPORTANT INSTRUCTIONS:
            1. Take into account the user instructions and the previously recommended steps.
              - If no user instructions are provided, just return the steps needed to understand the database.
            2. Never modify the existing data in the database, create new tables, or modify the database schema.
            3. Avoid unsafe code that could cause data loss or corruption.
            4. Take into account the database dialect and the tables and columns in the database.
            
            
            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of the database metadata and the SQL tables:
            {all_sql_database_summary}

            Return the steps as a numbered point list (no code, just the steps).
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include steps to modify existing tables, create new tables or modify the database schema.
            3. Make sure not to alter the existing data in the database.
            4. Make sure not to include unsafe code that could cause data loss or corruption.
            
            """,
            input_variables=["user_instructions", "recommended_steps", "all_sql_database_summary"]
        )
        
        # Create a connection if needed
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        # Get the database metadata
        all_sql_database_summary = get_database_metadata(conn, n_values=10)
        
        steps_agent = recommend_steps_prompt | llm
        
        recommended_steps = steps_agent({
            "user_instructions": state["user_instructions"],
            "recommended_steps": state["recommended_steps"],
            "all_sql_database_summary": all_sql_database_summary
        })
        
        return {
            "recommended_steps": "\n\n# Recommended SQL Database Steps:\n" + recommended_steps.content.strip(),
            "all_sql_database_summary": all_sql_database_summary
        }
        
        
        
        
    