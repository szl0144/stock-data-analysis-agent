from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

import pandas as pd

from typing import Any, Callable, Dict, Type, Optional

from ai_data_science_team.tools.parsers import PythonOutputParser
from ai_data_science_team.tools.regex import relocate_imports_inside_function, add_comments_to_top

def create_coding_agent_graph(
    GraphState: Type,
    node_functions: Dict[str, Callable],
    recommended_steps_node_name: str,
    create_code_node_name: str,
    execute_code_node_name: str,
    fix_code_node_name: str,
    explain_code_node_name: str,
    error_key: str,
    max_retries_key: str = "max_retries",
    retry_count_key: str = "retry_count",
    human_in_the_loop: bool = False,
    human_review_node_name: str = "human_review",
    checkpointer: Optional[Callable] = None
):
    """
    Creates a generic agent graph using the provided node functions and node names.
    
    Parameters
    ----------
    GraphState : Type
        The TypedDict or class used as state for the workflow.
    node_functions : dict
        A dictionary mapping node names to their respective functions.
        Example: {
            "recommend_cleaning_steps": recommend_cleaning_steps,
            "human_review": human_review,
            "create_data_cleaner_code": create_data_cleaner_code,
            "execute_data_cleaner_code": execute_data_cleaner_code,
            "fix_data_cleaner_code": fix_data_cleaner_code,
            "explain_data_cleaner_code": explain_data_cleaner_code
        }
    recommended_steps_node_name : str
        The node name that recommends steps.
    create_code_node_name : str
        The node name that creates the code.
    execute_code_node_name : str
        The node name that executes the generated code.
    fix_code_node_name : str
        The node name that fixes code if errors occur.
    explain_code_node_name : str
        The node name that explains the final code.
    error_key : str
        The state key used to check for errors.
    max_retries_key : str, optional
        The state key used for the maximum number of retries.
    retry_count_key : str, optional
        The state key for the current retry count.
    human_in_the_loop : bool, optional
        Whether to include a human review step.
    human_review_node_name : str, optional
        The node name for human review if human_in_the_loop is True.
    checkpointer : callable, optional
        A checkpointer callable if desired.
        
    Returns
    -------
    app : langchain.graphs.StateGraph
        The compiled workflow application.
    """

    workflow = StateGraph(GraphState)
    
    # Add the recommended steps node
    workflow.add_node(recommended_steps_node_name, node_functions[recommended_steps_node_name])
    
    # Optionally add the human review node
    if human_in_the_loop:
        workflow.add_node(human_review_node_name, node_functions[human_review_node_name])
        
    # Add main nodes
    workflow.add_node(create_code_node_name, node_functions[create_code_node_name])
    workflow.add_node(execute_code_node_name, node_functions[execute_code_node_name])
    workflow.add_node(fix_code_node_name, node_functions[fix_code_node_name])
    workflow.add_node(explain_code_node_name, node_functions[explain_code_node_name])
    
    # Set the entry point
    workflow.set_entry_point(recommended_steps_node_name)
    
    # Add edges depending on human_in_the_loop
    if human_in_the_loop:
        workflow.add_edge(recommended_steps_node_name, human_review_node_name)
    else:
        workflow.add_edge(recommended_steps_node_name, create_code_node_name)
    
    # Connect create_code_node to execution node
    workflow.add_edge(create_code_node_name, execute_code_node_name)
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        execute_code_node_name,
        lambda state: "fix_code" if (
            state.get(error_key) is not None and
            state.get(retry_count_key) is not None and
            state.get(max_retries_key) is not None and
            state.get(retry_count_key) < state.get(max_retries_key)
        ) else "explain_code",
        {"fix_code": fix_code_node_name, "explain_code": explain_code_node_name},
    )
    
    # From fix_code_node_name back to execution node
    workflow.add_edge(fix_code_node_name, execute_code_node_name)
    
    # explain_code_node_name leads to end
    workflow.add_edge(explain_code_node_name, END)
    
    # Compile workflow, optionally with checkpointer
    if human_in_the_loop and checkpointer is not None:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    return app


def node_func_human_review(
    state: Any, 
    prompt_text: str, 
    yes_goto: str, 
    no_goto: str,
    user_instructions_key: str = "user_instructions",
    recommended_steps_key: str = "recommended_steps",
) -> Command[str]:
    """
    A generic function to handle human review steps.
    
    Parameters
    ----------
    state : GraphState
        The current GraphState.
    prompt_text : str
        The text to display to the user before their input.
    yes_goto : str
        The node to go to if the user confirms (answers "yes").
    no_goto : str
        The node to go to if the user suggests modifications.
    user_instructions_key : str, optional
        The key in the state to store user instructions.
    recommended_steps_key : str, optional
        The key in the state to store recommended steps.    
    
    Returns
    -------
    Command[str]
        A Command object directing the next state and updates to the state.    
    """
    print("    * HUMAN REVIEW")

    # Display instructions and get user response
    user_input = interrupt(value=prompt_text.format(steps=state.get(recommended_steps_key, '')))

    # Decide next steps based on user input
    if user_input.strip().lower() == "yes":
        goto = yes_goto
        update = {}
    else:
        goto = no_goto
        modifications = "Modifications: \n" + user_input
        if state.get(user_instructions_key) is None:
            update = {user_instructions_key: modifications}
        else:
            update = {user_instructions_key: state.get(user_instructions_key) + modifications}

    return Command(goto=goto, update=update)


def node_func_execute_agent_code_on_data(
    state: Any, 
    data_key: str, 
    code_snippet_key: str, 
    result_key: str,
    error_key: str,
    agent_function_name: str,
    pre_processing: Optional[Callable[[Any], Any]] = None, 
    post_processing: Optional[Callable[[Any], Any]] = None,
    error_message_prefix: str = "An error occurred during agent execution: "
) -> Dict[str, Any]:
    """
    Execute a generic agent code defined in a code snippet retrieved from the state on input data and return the result.
    
    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` method to retrieve values.
    data_key : str
        The key in the state used to retrieve the input data.
    code_snippet_key : str
        The key in the state used to retrieve the Python code snippet defining the agent function.
    result_key : str
        The key in the state used to store the result of the agent function.
    error_key : str
        The key in the state used to store the error message if any.
    agent_function_name : str
        The name of the function (e.g., 'data_cleaner') expected to be defined in the code snippet.
    pre_processing : Callable[[Any], Any], optional
        A function to preprocess the data before passing it to the agent function.
        This might be used to convert raw data into a DataFrame or otherwise transform it.
        If not provided, a default approach will be used if data is a dict.
    post_processing : Callable[[Any], Any], optional
        A function to postprocess the output of the agent function before returning it.
    error_message_prefix : str, optional
        A prefix or full message to use in the error output if an exception occurs.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing the result and/or error messages. Keys are arbitrary, 
        but typically include something like "result" or "error".
    """
    
    print("    * EXECUTING AGENT CODE")
    
    # Retrieve raw data and code snippet from the state
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    
    # Preprocessing: If no pre-processing function is given, attempt a default handling
    if pre_processing is None:
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data)
        elif isinstance(data, list):
            df = [pd.DataFrame.from_dict(item) for item in data]
        else:
            raise ValueError("Data is not a dictionary or list and no pre_processing function was provided.")
    else:
        df = pre_processing(data)
    
    # Execute the code snippet to define the agent function
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Retrieve the agent function from the executed code
    agent_function = local_vars.get(agent_function_name, None)
    if agent_function is None or not callable(agent_function):
        raise ValueError(f"Agent function '{agent_function_name}' not found or not callable in the provided code.")
    
    # Execute the agent function
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        
        # Test an error
        # if state.get("retry_count") == 0:
        #     10/0
        
        # Apply post-processing if provided
        if post_processing is not None:
            result = post_processing(result)
    except Exception as e:
        print(e)
        agent_error = f"{error_message_prefix}{str(e)}"
    
    # Return results
    output = {result_key: result, error_key: agent_error}
    return output

def node_func_fix_agent_code(
    state: Any, 
    code_snippet_key: str, 
    error_key: str, 
    llm: Any, 
    prompt_template: str,
    agent_name: str,
    retry_count_key: str = "retry_count",
    log: bool = False,
    file_path: str = "logs/agent_function.py",
) -> dict:
    """
    Generic function to fix a given piece of agent code using an LLM and a prompt template.
    
    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` method to retrieve values.
    code_snippet_key : str
        The key in the state used to retrieve the broken code snippet.
    error_key : str
        The key in the state used to retrieve the related error message.
    llm : Any
        The language model or pipeline capable of receiving prompts and returning responses.
        It should support a call like `(llm | PythonOutputParser()).invoke(prompt)`.
    prompt_template : str
        A string template for the prompt that will be sent to the LLM. It should contain
        placeholders `{code_snippet}` and `{error}` which will be formatted with the actual values.
    agent_name : str
        The name of the agent being fixed. This is used to add comments to the top of the code.
    retry_count_key : str, optional
        The key in the state that tracks how many times we've retried fixing the code.
    log : bool, optional
        Whether to log the returned code to a file.
    file_path : str, optional
        The path to the file where the code will be logged.
    
    Returns
    -------
    dict
        A dictionary containing updated code, cleared error, and incremented retry count.
    """
    print("    * FIX AGENT CODE")
    print("      retry_count:" + str(state.get(retry_count_key)))
    
    # Retrieve the code snippet and the error from the state
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)

    # Format the prompt with the code snippet and the error
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message
    )
    
    # Execute the prompt with the LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    response = relocate_imports_inside_function(response)
    response = add_comments_to_top(response, agent_name="data_wrangler")
    
    # Log the response if requested
    if log:
        with open(file_path, 'w') as file:
            file.write(response)
            print(f"      File saved to: {file_path}")
    
    # Return updated results
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }

def node_func_explain_agent_code(
    state: Any, 
    code_snippet_key: str,
    result_key: str,
    error_key: str,
    llm: Any, 
    role: str,
    explanation_prompt_template: str,
    success_prefix: str = "# Agent Explanation:\n\n",
    error_message: str = "The agent encountered an error during execution and cannot be explained."
) -> Dict[str, Any]:
    """
    Generic function to explain what a given agent code snippet does.
    
    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` to retrieve values.
    code_snippet_key : str
        The key in `state` where the agent code snippet is stored.
    result_key : str
        The key in `state` where the LLM's explanation is stored. Typically this is "messages".
    error_key : str
        The key in `state` where any error messages related to the code snippet are stored.
    llm : Any
        The language model used to explain the code. Should support `.invoke(prompt)`.
    role : str
        The role of the agent explaining the code snippet. Examples: "Data Scientist", "Data Engineer", etc.
    explanation_prompt_template : str
        A prompt template that can be used to explain the code. It should contain a placeholder 
        for inserting the agent code snippet. For example:
        
        "Explain the steps performed by this agent code in a succinct manner:\n\n{code}"
        
    success_prefix : str, optional
        A prefix to add before the LLM's explanation, helping format the final message.
    error_message : str, optional
        Message to return if the agent code snippet cannot be explained due to an error.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing one key "messages", which is a list of messages (e.g., AIMessage) 
        describing the explanation or the error.
    """
    print("    * EXPLAIN AGENT CODE")
    
    # Check if there's an error associated with the code
    agent_error = state.get(error_key)
    if agent_error is None:
        # Retrieve the code snippet
        code_snippet = state.get(code_snippet_key)
        
        # Format the prompt by inserting the code snippet
        prompt = explanation_prompt_template.format(code=code_snippet)
        
        # Invoke the LLM to get an explanation
        response = llm.invoke(prompt)
        
        # Prepare the success message
        message = AIMessage(content=f"{success_prefix}{response.content}", role=role)
        return {"messages": [message]}
    else:
        # Return an error message if there was a problem with the code
        message = AIMessage(content=error_message)
        return {result_key: [message]}
