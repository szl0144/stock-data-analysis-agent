


import pandas as pd
from typing import Any, Callable, Dict, Optional

from ai_data_science_team.tools import PythonOutputParser

def execute_agent_code_on_data(
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
    data_raw = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    
    # Preprocessing: If no pre-processing function is given, attempt a default handling
    if pre_processing is None:
        if isinstance(data_raw, dict):
            df = pd.DataFrame.from_dict(data_raw)
        else:
            # If data isn't a dict, we must decide how to handle it.
            # For a generic function, raise an error or implement another fallback logic.
            raise ValueError("Data is not a dictionary and no pre_processing function was provided.")
    else:
        df = pre_processing(data_raw)
    
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
        # Apply post-processing if provided
        if post_processing is not None:
            result = post_processing(result)
    except Exception as e:
        print(e)
        agent_error = f"{error_message_prefix}{str(e)}"
    
    # Return results
    output = {result_key: result, error_key: agent_error}
    return output

def fix_agent_code(
    state: Any, 
    code_snippet_key: str, 
    error_key: str, 
    llm: Any, 
    prompt_template: str,
    retry_count_key: str = "retry_count",
    log: bool = False,
    log_path: str = "logs/",
    log_file_name: str = "fixed_code.py"
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
    retry_count_key : str, optional
        The key in the state that tracks how many times we've retried fixing the code.
    log : bool, optional
        Whether to log the returned code to a file.
    log_path : str, optional
        Directory path for the log file.
    log_file_name : str, optional
        File name for logging the fixed code.
    
    Returns
    -------
    dict
        A dictionary containing updated code, cleared error, and incremented retry count.
    """
    # Retrieve the code snippet and the error from the state
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)

    print("    * FIX AGENT CODE")
    print(error_message)

    # Format the prompt with the code snippet and the error
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message
    )
    
    # Execute the prompt with the LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    print(response)
    
    # Log the response if requested
    if log:
        with open(log_path + log_file_name, 'w') as file:
            file.write(response)
    
    # Return updated results
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }
