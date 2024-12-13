# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# Agents
# ai_data_science_team/agents.py

# Libraries
from typing import TypedDict, Annotated, Sequence
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

import os
import io
import pandas as pd

from ai_data_science_team.templates.agent_templates import execute_agent_code_on_data, fix_agent_code, explain_agent_code
from ai_data_science_team.tools.parsers import PythonOutputParser

# Setup

LOG_PATH = os.path.join(os.getcwd(), "logs/")


# * Data Cleaning Agent

def data_cleaning_agent(model, log=False, log_path=None):
    """
    Creates a data cleaning agent that can be run on a dataset. The agent can be used to clean a dataset in a variety of
    ways, such as removing columns with more than 40% missing values, imputing missing
    values with the mean of the column if the column is numeric, or imputing missing
    values with the mode of the column if the column is categorical.
    The agent takes in a dataset and some user instructions, and outputs a python
    function that can be used to clean the dataset. The agent also logs the code
    generated and any errors that occur.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to
        "logs/".
        
    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import data_cleaning_agent
    
    llm = ChatOpenAI(model = "gpt-4o-mini")

    data_cleaning_agent = data_cleaning_agent(llm)
    
    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")
    
    response = data_cleaning_agent.invoke({
        "user_instructions": "Don't remove outliers when cleaning the data.",
        "data_raw": df.to_dict(),
        "max_retries":3, 
        "retry_count":0
    })
    
    pd.DataFrame(response['data_cleaned'])
    ```

    Returns
    -------
    app : langchain.graphs.StateGraph
        The data cleaning agent as a state graph.
    """
    llm = model
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        data_cleaner_function: str
        data_cleaner_error: str
        data_cleaned: dict
        max_retries: int
        retry_count: int
    
    
    def create_data_cleaner_code(state: GraphState):
        print("---DATA CLEANING AGENT----")
        print("    * CREATE DATA CLEANER CODE")
        
        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Your job is to create a data_cleaner() function to that can be run on the data provided.
            
            Things that should be considered in the data summary function:
            
            * Removing columns if more than 40 percent of the data is missing
            * Imputing missing values with the mean of the column if the column is numeric
            * Imputing missing values with the mode of the column if the column is categorical
            * Converting columns to the correct data type
            * Removing duplicate rows
            * Removing rows with missing values
            * Removing rows with extreme outliers (3X the interquartile range)
            
            Make sure to take into account any additional user instructions that may negate some of these steps or add new steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
            
            User instructions:
            {user_instructions}
            
            Return Python code in ```python ``` format with a single function definition, data_cleaner(data_raw), that incldues all imports inside the function.
            
            You can use Pandas, Numpy, and Scikit Learn libraries to clean the data.

            Sample Data (first 100 rows):
            {data_head}
            
            Data Description:
            {data_description}
            
            Data Info:
            {data_info}
            
            Return code to provide the data cleaning function:
            
            def data_cleaner(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_cleaner
            
            Best Practices and Error Preventions:
            
            Always ensure that when assigning the output of fit_transform() from SimpleImputer to a Pandas DataFrame column, you call .ravel() or flatten the array, because fit_transform() returns a 2D array while a DataFrame column is 1D.
            
            """,
            input_variables=["user_instructions","data_head", "data_description", "data_info"]
        )

        data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()
        
        data_raw = state.get("data_raw")
        
        df = pd.DataFrame.from_dict(data_raw)
        
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()
        
        response = data_cleaning_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "data_head": df.head().to_string(), 
            "data_description": df.describe().to_string(), 
            "data_info": info_text
        })
        
        # For logging: store the code generated:
        if log:
            with open(log_path + 'data_cleaner.py', 'w') as file:
                file.write(response)
   
        return {"data_cleaner_function" : response}
    
    def execute_data_cleaner_code(state):
        return execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name="data_cleaner",
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict(),
            error_message_prefix="An error occurred during data cleaning: "
        )
        
    def fix_data_cleaner_code(state: GraphState):
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Your job is to create a data_cleaner() function that can be run on the data provided. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for data_cleaner().
        
        Return Python code in ```python``` format with a single function definition, data_cleaner(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=llm,  
            prompt_template=data_cleaner_prompt,
            log=True,
            log_path="logs/",
            log_file_name="data_cleaner.py"
        )
    
    def explain_data_cleaner_code(state: GraphState):        
        return explain_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            result_key="messages",
            error_key="data_cleaner_error",
            llm=llm,  
            explanation_prompt_template="""
            Explain the data cleaning steps that the data cleaning agent performed in this function. 
            Keep the summary succinct and to the point.\n\n# Data Cleaning Agent:\n\n{code}
            """,
            success_prefix="# Data Cleaning Agent:\n\n ",
            error_message="The Data Cleaning Agent encountered an error during data cleaning. Data could not be explained."
        )
        
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    workflow.add_node("explain_data_cleaner_code", explain_data_cleaner_code)
    
    workflow.set_entry_point("create_data_cleaner_code")
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    
    workflow.add_conditional_edges(
        "execute_data_cleaner_code", 
        lambda state: "fix_code" 
            if (state.get("data_cleaner_error") is not None
                and state.get("retry_count") is not None
                and state.get("max_retries") is not None
                and state.get("retry_count") < state.get("max_retries")) 
            else "explain_code",
        {"fix_code": "fix_data_cleaner_code", "explain_code": "explain_data_cleaner_code"},
    )
    
    workflow.add_edge("fix_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("explain_data_cleaner_code", END)
    
    app = workflow.compile()
    
    return app

# # * Data Summary Agent

# def data_summary_agent(model, log=True, log_path=None):
    
#     # Setup Log Directory
#     if log:
#         if log_path is None:
#             log_path = LOG_PATH
#         if not os.path.exists(log_path):
#             os.makedirs(log_path)
    
#     llm = model
    
#     data_summary_prompt = PromptTemplate(
#         template="""
#         You are a Data Summary Agent. Your job is to summarize a dataset.
        
#         Things that should be considered in the data summary function:
        
#         * How many missing values
#         * How many unique values
#         * How many rows
#         * How many columns
#         * What data types are present
#         * What the data looks like
#         * What column types are present
#         * What is the distribution of the data
#         * What is the correlation between the data
        
#         Make sure to take into account any additional user instructions that may negate some of these steps or add new steps.
        
#         User instructions:
#         {user_instructions}
        
#         Return Python code in ```python ``` format with a single function definition, data_sumary(data), that incldues all imports inside the function.
        
#         You can use Pandas, Numpy, and Scikit Learn libraries to summarize the data.

#         Sample Data (first 100 rows):
#         {data_head}
        
#         Data Description:
#         {data_description}
        
#         Data Info:
#         {data_info}
        
#         Return code to provide the data cleaning function:
        
#         def data_summary(data):
#             import pandas as pd
#             import numpy as np
#             ...
#             return {
#                 'data_summary': ..., 
#                 'data_correlation': ...
#                 [INSERT MORE KEYS HERE],
#             }
        
#         """,
#         input_variables=["user_instructions","data_head", "data_description", "data_info"]
#     )

#     data_summary_agent = data_summary_prompt | llm | PythonOutputParser()
    
    
    
#     return 1
