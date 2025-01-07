# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Visualization Agent



# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import io
import pandas as pd

from ai_data_science_team.templates import(
    node_func_execute_agent_code_on_data, 
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_explain_agent_code, 
    create_coding_agent_graph
)
from ai_data_science_team.tools.parsers import PythonOutputParser
from ai_data_science_team.tools.regex import relocate_imports_inside_function, add_comments_to_top, format_agent_name
from ai_data_science_team.tools.metadata import get_dataframe_summary
from ai_data_science_team.tools.logging import log_ai_function

# Setup
AGENT_NAME = "data_visualization_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Agent

def make_data_visualization_agent(
    model, 
    n_samples=30,
    log=False, 
    log_path=None, 
    file_name="data_visualization.py",
    overwrite = True, 
    human_in_the_loop=False, 
    bypass_recommended_steps=False, 
    bypass_explain_code=False
):
    
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
        user_instructions_processed: str
        recommended_steps: str
        data_raw: dict
        plotly_graph: dict
        all_datasets_summary: str
        data_visualization_function: str
        data_visualization_function_path: str
        data_visualization_function_name: str
        data_visualization_error: str
        max_retries: int
        retry_count: int
        
    def chart_instructor(state: GraphState):
        
        print(format_agent_name(AGENT_NAME))
        print("    * CREATE CHART GENERATOR INSTRUCTIONS")
        
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 
    
            You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart from the data that will be passed to a chart generator agent.
            
            USER QUESTION / INSTRUCTIONS: 
            {user_instructions}
            
            Previously Recommended Instructions (if any):
            {recommended_steps}
            
            DATA: 
            {all_datasets_summary}
            
            Formulate chart generator instructions by informing the chart generator of what type of plotly plot to use (e.g. bar, line, scatter, etc) to best represent the data. 
            
            Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.
            
            Instruct the chart generator to use the following theme colors, sizes, etc:
            
            - Start with the "plotly_white" template
            - Use a white background
            - Use this color for bars and lines:
                'blue': '#3381ff',
            - Base Font Size: 8.8 (Used for x and y axes tickfont, any annotations, hovertips)
            - Title Font Size: 13.2
            - Line Size: 0.65 (specify these within the xaxis and yaxis dictionaries)
            - Add smoothers or trendlines to scatter plots unless not desired by the user
            - Do not use color_discrete_map (this will result in an error)
            - Hover tip size: 8.8
            
            Return your instructions in the following format:
            CHART GENERATOR INSTRUCTIONS: 
            FILL IN THE INSTRUCTIONS HERE
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            """,
            input_variables=["user_instructions", "recommended_steps", "all_datasets_summary"]
            
        )
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples, skip_stats=True)
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        chart_instructor = recommend_steps_prompt | llm | StrOutputParser()
        
        recommended_steps = chart_instructor.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_datasets_summary": all_datasets_summary_str
        })
        
        return {
            "recommended_steps": "\n\n# Recommended Data Cleaning Steps:\n" + recommended_steps.content.strip(),
            "all_datasets_summary": all_datasets_summary_str
        }
        
    def chart_generator(state: GraphState):
        
        print("    * CREATE DATA VISUALIZATION CODE")

        
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))
            
            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)

            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples, skip_stats=True)
            
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
        
        
        
        
        return
            
            
        
    