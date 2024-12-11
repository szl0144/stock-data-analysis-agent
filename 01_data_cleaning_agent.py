

from langchain_openai import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from langgraph.graph import START, END, StateGraph

import os
import io

from typing import TypedDict, Annotated, Sequence
import operator

import pandas as pd

import os
import yaml

from IPython.display import Image
from pprint import pprint

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

#region DATA_CLEANING_AGENT

# * Data Cleaning Agent

def data_cleaning_agent(model, log=True, log_path=None):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model   
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    
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
        input_variables=["data_head", "data_description", "data_info"]
    )

    data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()
    

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        data_raw: dict
        data_cleaning_function: str
        data_cleaner_error: str
        data_cleaned: dict
        max_retries: int
        retry_count: int
    
    
    def create_data_cleaner_code(state: GraphState):
        print("---DATA CLEANING AGENT----")
        print("    * CREATE DATA CLEANER CODE")
        
        data_raw = state.get("data_raw")
        
        df = pd.DataFrame.from_dict(data_raw)
        
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()
        
        response = data_cleaning_agent.invoke({
            "data_head": df.head().to_string(), 
            "data_description": df.describe().to_string(), 
            "data_info": info_text
        })
        
        pprint(response)
        
        # For logging: store the code generated:
        if log:
            with open(log_path + 'data_cleaner.py', 'w') as file:
                file.write(response)
   
        return {"data_cleaning_function" : response}
    
    def execute_data_cleaner_code(state: GraphState):     
        print("    * EXECUTE DATA CLEANER FUNCTION")
        # Execute the code
        
        data_raw = state.get("data_raw")
        
        df = pd.DataFrame.from_dict(data_raw)
        
        data_cleaning_function = state.get("data_cleaning_function")
        
        # Execute the code
        local_vars = {}
        global_vars = {}
        exec(data_cleaning_function, global_vars, local_vars)
        
        data_cleaner = local_vars.get('data_cleaner')
        
        data_cleaner_error = None
        df_cleaned = None
        try:
            df_cleaned = data_cleaner(df).to_dict()
        except Exception as e:
            print(e)
            data_cleaner_error = f"An error occurred during data cleaning: {str(e)}"
        
        return {
            "data_cleaned": df_cleaned,
            "data_cleaner_error": data_cleaner_error
        }
        
    def fix_data_cleaner_code(state: GraphState):
        print("    * FIX DATA CLEANER CODE")
        
        data_cleaning_function = state.get("data_cleaning_function")
        data_cleaner_error = state.get("data_cleaner_error")
        
        print(data_cleaner_error)
        
        response = (llm | PythonOutputParser()).invoke(
            f"""
            You are a Data Cleaning Agent. Your job is to create a data_cleaner() function to that can be run on the data provided. The function is currently broken and needs to be fixed.
            
            Make sure to only return the function definition for data_cleaner().
            
            Return Python code in ```python ``` format with a single function definition, data_cleaner(data_raw), that incldues all imports inside the function.
            
            This is the broken code (please fix): \n\n 
            
            { data_cleaning_function}
            
            Last Known Error: 
            
            {data_cleaner_error}
            """
        ) 
        
        print(response)
        
        # For logging: store the code generated:
        if log:
            with open(log_path + 'data_cleaner.py', 'w') as file:
                file.write(response)
        
        return {
            "data_cleaning_function" : response, "data_cleaner_error": None,
            "retry_count": state.get("retry_count") + 1
        }
    
    def explain_data_cleaner_code(state: GraphState):
        print("    * EXPLAIN DATA CLEANER CODE")
        
        if state.get("data_cleaner_error") is None:
        
            data_cleaning_function = state.get("data_cleaning_function")
            
            response = llm.invoke("Explain the data cleaning steps that the data cleaning agent performed in this function. Keep the summary succinct and to the point. \n\n # Data Cleaning Agent: \n\n" + data_cleaning_function)
            
            
            message = AIMessage(content=f"# Data Cleaning Agent: \n\n The Data Cleaning Agent preformed data cleaning with the following code explanation for decisions made: \n\n{response.content}")
            
            return {"messages": [message]}
        
        else:
            message = AIMessage(content="The Data Cleaning Agent encountered an error during data cleaning. Data could not be cleaned.")
            return {"messages": [message]}
        
    
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
    

agent_data_cleaning = data_cleaning_agent(model = llm)

Image(agent_data_cleaning.get_graph().draw_mermaid_png())


# Test

response = agent_data_cleaning.invoke({
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})
    

response.keys()

pd.DataFrame(response['data_cleaned'])

pprint(response['messages'][0].content)

#endregion 
    
        
    
    
    

