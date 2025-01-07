# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Visualization Agent



# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples, skip_stats=False)
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        chart_instructor = recommend_steps_prompt | llm 
        
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

            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples, skip_stats=False)
            
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
            
            chart_generator_instructions = state.get("user_instructions")
            
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            chart_generator_instructions = state.get("recommended_steps")
        
        prompt_template = PromptTemplate(
            template="""
            You are a chart generator agent that is an expert in generating plotly charts. You must use plotly or plotly.express to produce plots.
    
            Your job is to produce python code to generate visualizations.
            
            You will take instructions from a Chart Instructor and generate a plotly chart from the data provided.
            
            CHART INSTRUCTIONS: 
            {chart_generator_instructions}
            
            DATA: 
            {all_datasets_summary}
            
            RETURN:
            
            Return Python code in ```python ``` format with a single function definition, data_visualization(data_raw), that includes all imports inside the function.
            
            Return the plotly chart as a dictionary.
            
            Return code to provide the data visualization function:
            
            def data_visualization(data_raw):
                import pandas as pd
                import numpy as np
                import json
                import plotly.graph_objects as go
                import plotly.io as pio
                
                ...
                
                fig_json = pio.to_json(fig)
                fig_dict = json.loads(fig_json)
                
                return fig_dict
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            
            """,
            input_variables=["chart_generator_instructions", "all_datasets_summary"]
        )
        
        data_visualization_agent = prompt_template | llm | PythonOutputParser()
        
        response = data_visualization_agent.invoke({
            "chart_generator_instructions": chart_generator_instructions,
            "all_datasets_summary": all_datasets_summary_str
        })
        
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)
        
        # For logging: store the code generated:
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )
        
        return {
            "data_visualization_function": response,
            "data_visualization_function_path": file_path,
            "data_visualization_function_name": file_name_2,
            "all_datasets_summary": all_datasets_summary_str
        }
            
    def human_review(state: GraphState) -> Command[Literal["chart_instructor", "chart_generator"]]:
        return node_func_human_review(
            state=state,
            prompt_text="Is the following data visualization instructions correct? (Answer 'yes' or provide modifications)\n{steps}",
            yes_goto="chart_generator",
            no_goto="chart_instructor",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_steps"            
        )
    
        
    def execute_data_visualization_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="plotly_graph",
            error_key="data_visualization_error",
            code_snippet_key="data_visualization_function",
            agent_function_name="data_visualization",
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            # post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during data visualization: "
        )
    
    def fix_data_visualization_code(state: GraphState):
        prompt = """
        You are a Data Visualization Agent. Your job is to create a data_visualization() function that can be run on the data provided. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for data_visualization().
        
        Return Python code in ```python``` format with a single function definition, data_visualization(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_visualization_function",
            error_key="data_visualization_error",
            llm=llm,  
            prompt_template=prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_visualization_function_path"),
        )
    
    def explain_data_visualization_code(state: GraphState):        
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="data_visualization_function",
            result_key="messages",
            error_key="data_visualization_error",
            llm=llm,  
            role=AGENT_NAME,
            explanation_prompt_template="""
            Explain the data visualization steps that the data visualization agent performed in this function. 
            Keep the summary succinct and to the point.\n\n# Data Visualization Agent:\n\n{code}
            """,
            success_prefix="# Data Visualization Agent:\n\n ",
            error_message="The Data Visualization Agent encountered an error during data visualization. No explanation could be provided."
        )
        
    # Define the graph
    node_functions = {
        "chart_instructor": chart_instructor,
        "human_review": human_review,
        "chart_generator": chart_generator,
        "execute_data_visualization_code": execute_data_visualization_code,
        "fix_data_visualization_code": fix_data_visualization_code,
        "explain_data_visualization_code": explain_data_visualization_code
    }
    
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="chart_instructor",
        create_code_node_name="chart_generator",
        execute_code_node_name="execute_data_visualization_code",
        fix_code_node_name="fix_data_visualization_code",
        explain_code_node_name="explain_data_visualization_code",
        error_key="data_visualization_error",
        human_in_the_loop=human_in_the_loop,  # or False
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
    )
        
    return app
    