# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# Agents

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


import os
import io
import pandas as pd

from pprint import pprint

from ai_data_science_team.templates.agent_templates import execute_agent_code_on_data, fix_agent_code, explain_agent_code
from ai_data_science_team.tools.parsers import PythonOutputParser
from ai_data_science_team.tools.regex import relocate_imports_inside_function

# Setup

LOG_PATH = os.path.join(os.getcwd(), "logs/")

# * Data Cleaning Agent

def make_data_cleaning_agent(model, log=False, log_path=None, human_in_the_loop=False):
    """
    Creates a data cleaning agent that can be run on a dataset. The agent can be used to clean a dataset in a variety of
    ways, such as removing columns with more than 40% missing values, imputing missing
    values with the mean of the column if the column is numeric, or imputing missing
    values with the mode of the column if the column is categorical.
    The agent takes in a dataset and some user instructions, and outputs a python
    function that can be used to clean the dataset. The agent also logs the code
    generated and any errors that occur.
    
    The agent is instructed to to perform the following data cleaning steps:
    
    - Removing columns if more than 40 percent of the data is missing
    - Imputing missing values with the mean of the column if the column is numeric
    - Imputing missing values with the mode of the column if the column is categorical
    - Converting columns to the correct data type
    - Removing duplicate rows
    - Removing rows with missing values
    - Removing rows with extreme outliers (3X the interquartile range)
    - User instructions can modify, add, or remove any of the above steps

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
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the data cleaning instructions. Defaults to False.
        
    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import data_cleaning_agent
    
    llm = ChatOpenAI(model = "gpt-4o-mini")

    data_cleaning_agent = make_data_cleaning_agent(llm)
    
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
        recommended_steps: str
        data_raw: dict
        data_cleaner_function: str
        data_cleaner_error: str
        data_cleaned: dict
        max_retries: int
        retry_count: int

    
    def recommend_cleaning_steps(state: GraphState):
        """
        Recommend a series of data cleaning steps based on the input data. 
        These recommended steps will be appended to the user_instructions.
        """
        print("---DATA CLEANING AGENT----")
        print("    * RECOMMEND CLEANING STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Expert. Given the following information about the data, 
            recommend a series of numbered steps to take to clean and preprocess it. 
            The steps should be tailored to the data characteristics and should be helpful 
            for a data cleaning agent that will be implemented.
            
            Things that should be considered in the data cleaning steps:
            
            * Removing columns if more than 40 percent of the data is missing
            * Imputing missing values with the mean of the column if the column is numeric
            * Imputing missing values with the mode of the column if the column is categorical
            * Converting columns to the correct data type
            * Removing duplicate rows
            * Removing rows with missing values
            * Removing rows with extreme outliers (3X the interquartile range)
            
            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
            
            User instructions:
            {user_instructions}
            
            Previously Recommended Steps (if any):
            {recommended_steps}
            
            Data Sample (first 5 rows):
            {data_head}
            
            Data Description:
            {data_description}
            
            Data Info:
            {data_info}

            Return the steps as a bullet point list (no code, just the steps).
            """,
            input_variables=["user_instructions", "data_head","data_description","data_info"]
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "data_head": df.head().to_string(),
            "data_description": df.describe().to_string(),
            "data_info": info_text
        }) 
        
        # pprint(recommended_steps.content)
        
        return {"recommended_steps": "\n\n# Recommended Steps:\n" + recommended_steps.content.strip()}
    
    def human_review(state: GraphState) -> Command[Literal["recommend_cleaning_steps", "create_data_cleaner_code"]]:
        print("    * HUMAN REVIEW")
        
        user_input = interrupt(
            value=f"Is the following data cleaning instructions correct? (Answer 'yes' or provide modifications to make to make them correct)\n{state.get('recommended_steps')}",
        )
        
        # print(user_input)
        
        if user_input.strip().lower() == "yes":
            goto = "create_data_cleaner_code"
            update = {}
        else:
            goto = "recommend_cleaning_steps"
            modifications = "Modifications: \n" + user_input
            if state.get("user_instructions") is None:
                update = {
                    "user_instructions": modifications,
                    # "recommended_steps": None
                }
            else:
                update = {
                    "user_instructions": state.get("user_instructions") + modifications,
                    # "recommended_steps": None
                }
        
        return Command(goto=goto, update=update)
    
    def create_data_cleaner_code(state: GraphState):
        print("    * CREATE DATA CLEANER CODE")
        
        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Your job is to create a data_cleaner() function that can be run on the data provided using the following recommended steps.
            
            Recommended Steps:
            {recommended_steps}
            
            You can use Pandas, Numpy, and Scikit Learn libraries to clean the data.
            
            Use this information about the data to help determine how to clean the data:

            Sample Data (first 100 rows):
            {data_head}
            
            Data Description:
            {data_description}
            
            Data Info:
            {data_info}
            
            Return Python code in ```python ``` format with a single function definition, data_cleaner(data_raw), that incldues all imports inside the function. 
            
            Return code to provide the data cleaning function:
            
            def data_cleaner(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_cleaned
            
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
            "recommended_steps": state.get("recommended_steps"),
            "data_head": df.head().to_string(), 
            "data_description": df.describe().to_string(), 
            "data_info": info_text
        })
        
        response = relocate_imports_inside_function(response)
        
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
            log=log,
            log_path=log_path,
            log_file_name="data_cleaner.py"
        )
    
    def explain_data_cleaner_code(state: GraphState):        
        return explain_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            result_key="messages",
            error_key="data_cleaner_error",
            llm=llm,  
            role="data_cleaning_agent",
            explanation_prompt_template="""
            Explain the data cleaning steps that the data cleaning agent performed in this function. 
            Keep the summary succinct and to the point.\n\n# Data Cleaning Agent:\n\n{code}
            """,
            success_prefix="# Data Cleaning Agent:\n\n ",
            error_message="The Data Cleaning Agent encountered an error during data cleaning. Data could not be explained."
        )
        
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("recommend_cleaning_steps", recommend_cleaning_steps)
    
    if human_in_the_loop:
        workflow.add_node("human_review", human_review)
    
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    workflow.add_node("explain_data_cleaner_code", explain_data_cleaner_code)
    
    workflow.set_entry_point("recommend_cleaning_steps")
    
    if human_in_the_loop:
        workflow.add_edge("recommend_cleaning_steps", "human_review")
    else:
        workflow.add_edge("recommend_cleaning_steps", "create_data_cleaner_code")
        
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
    
    if human_in_the_loop:
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    return app

# * Feature Engineering Agent

def make_feature_engineering_agent(model, log=False, log_path=None):
    """
    Creates a feature engineering agent that can be run on a dataset. The agent applies various feature engineering
    techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,
    and generating polynomial features. The agent takes in a dataset and user instructions and outputs a Python
    function for feature engineering. It also logs the code generated and any errors that occur.
    
    The agent is instructed to apply the following feature engineering techniques:
    
    - Remove string or categorical features with unique values equal to the size of the dataset
    - Remove constant features with the same value in all rows
    - High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
    - Encoding categorical variables using OneHotEncoding
    - Numeric features should be left untransformed
    - Create datetime-based features if datetime columns are present
    - If a target variable is provided:
        - If a categorical target variable is provided, encode it using LabelEncoding
        - All other target variables should be converted to numeric and unscaled
    - Convert any boolean True/False values to 1/0
    - Return a single data frame containing the transformed features and target variable, if one is provided.
    - Any specific instructions provided by the user

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to "logs/".

    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import feature_engineering_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    feature_engineering_agent = make_feature_engineering_agent(llm)

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    response = feature_engineering_agent.invoke({
        "user_instructions": None,
        "target_variable": "Churn",
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['data_engineered'])
    ```

    Returns
    -------
    app : langchain.graphs.StateGraph
        The feature engineering agent as a state graph.
    """
    llm = model

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        target_variable: str
        feature_engineer_function: str
        feature_engineer_error: str
        data_engineered: dict
        max_retries: int
        retry_count: int

    def create_feature_engineering_code(state: GraphState):
        print("---FEATURE ENGINEERING AGENT----")
        print("    * CREATE FEATURE ENGINEERING CODE")

        feature_engineering_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Agent. Your job is to create a feature_engineer() function that generates and applies new features for the given data.
            
            The function should consider:
            - Remove string or categorical features with unique values equal to the size of the dataset
            - Remove constant features with the same value in all rows
            - High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
            - Encoding categorical variables using OneHotEncoding
            - Numeric features should be left untransformed
            - Create datetime-based features if datetime columns are present
            - If a target variable is provided:
                - If a categorical target variable is provided, encode it using LabelEncoding
                - All other target variables should be converted to numeric and unscaled
            - Convert any boolean True/False values to 1/0
            - Return a single data frame containing the transformed features and target variable, if one is provided.
            - Any specific instructions provided by the user
            
            User instructions:
            {user_instructions}
            
            Return Python code in ```python``` format with a single function definition, feature_engineer(data_raw), including all imports inside the function.
            
            Target Variable: {target_variable}
            
            Sample Data (first 100 rows):
            {data_head}
            
            Data Description:
            {data_description}
            
            Data Info:
            {data_info}
            
            Return code to provide the feature engineering function:
            
            def feature_engineer(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_engineered
            
            Best Practices and Error Preventions:
            - Handle missing values in numeric and categorical features before transformations.
            - Avoid creating highly correlated features unless explicitly instructed.
            
            Avoid the following errors:
            
            - name 'OneHotEncoder' is not defined
            
            - Shape of passed values is (7043, 48), indices imply (7043, 47)
            
            - name 'numeric_features' is not defined
            
            - name 'categorical_features' is not defined


            """,
            input_variables=["user_instructions", "target_variable", "data_head", "data_description", "data_info"]
        )

        feature_engineering_agent = feature_engineering_prompt | llm | PythonOutputParser()

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        response = feature_engineering_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "target_variable": state.get("target_variable"),
            "data_head": df.head().to_string(),
            "data_description": df.describe(include='all').to_string(),
            "data_info": info_text
        })
        
        response = relocate_imports_inside_function(response)

        # For logging: store the code generated
        if log:
            with open(log_path + 'feature_engineer.py', 'w') as file:
                file.write(response)

        return {"feature_engineer_function": response}

    def execute_feature_engineering_code(state):
        return execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_engineered",
            error_key="feature_engineer_error",
            code_snippet_key="feature_engineer_function",
            agent_function_name="feature_engineer",
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict(),
            error_message_prefix="An error occurred during feature engineering: "
        )

    def fix_feature_engineering_code(state: GraphState):
        feature_engineer_prompt = """
        You are a Feature Engineering Agent. Your job is to fix the feature_engineer() function that currently contains errors.
        
        Provide only the corrected function definition.
        
        Broken code:
        {code_snippet}

        Last Known Error:
        {error}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="feature_engineer_function",
            error_key="feature_engineer_error",
            llm=llm,
            prompt_template=feature_engineer_prompt,
            log=log,
            log_path=log_path,
            log_file_name="feature_engineer.py"
        )

    def explain_feature_engineering_code(state: GraphState):
        return explain_agent_code(
            state=state,
            code_snippet_key="feature_engineer_function",
            result_key="messages",
            error_key="feature_engineer_error",
            llm=llm,
            role="feature_engineering_agent",
            explanation_prompt_template="""
            Explain the feature engineering steps performed by this function. Keep the explanation clear and concise.\n\n# Feature Engineering Agent:\n\n{code}
            """,
            success_prefix="# Feature Engineering Agent:\n\n ",
            error_message="The Feature Engineering Agent encountered an error during feature engineering. Data could not be explained."
        )

    workflow = StateGraph(GraphState)

    workflow.add_node("create_feature_engineering_code", create_feature_engineering_code)
    workflow.add_node("execute_feature_engineering_code", execute_feature_engineering_code)
    workflow.add_node("fix_feature_engineering_code", fix_feature_engineering_code)
    workflow.add_node("explain_feature_engineering_code", explain_feature_engineering_code)

    workflow.set_entry_point("create_feature_engineering_code")
    workflow.add_edge("create_feature_engineering_code", "execute_feature_engineering_code")

    workflow.add_conditional_edges(
        "execute_feature_engineering_code",
        lambda state: "fix_code" 
            if (state.get("feature_engineer_error") is not None
                and state.get("retry_count") is not None
                and state.get("max_retries") is not None
                and state.get("retry_count") < state.get("max_retries")) 
            else "explain_code",
        {"fix_code": "fix_feature_engineering_code", "explain_code": "explain_feature_engineering_code"},
    )

    workflow.add_edge("fix_feature_engineering_code", "execute_feature_engineering_code")
    workflow.add_edge("explain_feature_engineering_code", END)

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
