
from typing import Any, Optional, Annotated, Sequence, List, Dict, Tuple

from langchain.tools import tool

from langgraph.prebuilt import InjectedState  


@tool(response_format='content_and_artifact')
def describe_dataset(
    data_raw: Annotated[dict, InjectedState("data_raw")]
) -> Tuple[str, Dict]:
    """
    Tool: describe_dataset
    Description:
        Converts injected raw data (a dict) into a pandas DataFrame and computes summary
        statistics using the DataFrame's describe() method.
        
    Returns:
    -------
    Tuple[str, Dict]:
        content: A textual summary of the DataFrame's descriptive statistics.
        artifact: A dictionary (from DataFrame.describe()) for further inspection.
    """
    print("    * Tool: describe_dataset")
    import pandas as pd
    df = pd.DataFrame(data_raw)
    description_df = df.describe(include='all')
    content = "Summary statistics computed using pandas describe()."
    artifact = description_df.to_dict()
    return content, artifact


@tool(response_format='content_and_artifact')
def visualize_missing(
    data_raw: Annotated[dict, InjectedState("data_raw")]
) -> Tuple[str, Dict]:
    """
    Tool: visualize_missing
    Description:
        Converts injected raw data (a dict) into a DataFrame, generates a missing data visualization
        (using missingno), and returns a base64-encoded PNG image.
        
    Returns:
    -------
    Tuple[str, Dict]:
        content: A message describing the generated plot.
        artifact: A dict with key 'plot_image' containing the base64 encoded image.
    """
    print("    * Tool: visualize_missing")
    try:
        import missingno as msno  # Ensure missingno is installed
    except ImportError:
        raise ImportError("Please install the 'missingno' package to use this tool. pip install missingno")
    
    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(data_raw)
    
    plt.figure(figsize=(8, 6))
    msno.matrix(df)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    content = "Missing data visualization generated."
    artifact = {"plot_image": encoded}
    return content, artifact


@tool(response_format='content_and_artifact')
def correlation_funnel(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    target: str,
    index: int = -1,
    method: str = "pearson",
    n_bins: int = 4,
    thresh_infreq: float = 0.01,
    name_infreq: str = "-OTHER",
) -> Tuple[str, Dict]:
    """
    Tool: correlation_funnel
    Description:
        Converts injected raw data (a dict) into a DataFrame, applies binarization, computes the
        correlation funnel with respect to the specified target level, and (optionally) generates a static plot.
    
    Parameters:
    ----------
    target : str
        The base target column name (e.g., 'Member_Status'). The tool will look for columns that begin
        with this string followed by '__' (e.g., 'Member_Status__Gold', 'Member_Status__Silver', etc.).
    index : int, default -1
        The index of the target level to select from the list of matching columns. For example, -1 selects
        the last matching column.
    method : str
        The correlation method ('pearson', 'kendall', or 'spearman'). Default is 'pearson'.
    n_bins : int
        The number of bins to use for binarization. Default is 4.
    thresh_infreq : float
        The threshold for infrequent levels. Default is 0.01.
    name_infreq : str
        The name to use for infrequent levels. Default is '-OTHER'.
    """
    print("    * Tool: correlation_funnel")
    try:
        import pytimetk as tk
    except ImportError:
        raise ImportError("Please install the 'pytimetk' package to use this tool. pip install pytimetk")
    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import json
    import plotly.graph_objects as go
    import plotly.io as pio
    
    # Convert the raw injected state into a DataFrame.
    df = pd.DataFrame(data_raw)
    
    # Apply the binarization method.
    df_binarized = df.binarize(
        n_bins=n_bins, 
        thresh_infreq=thresh_infreq, 
        name_infreq=name_infreq, 
        one_hot=True
    )
    
    # Determine the full target column name.
    # Look for all columns that start with "target__"
    matching_columns = [col for col in df_binarized.columns if col.startswith(f"{target}__")]
    if not matching_columns:
        # If no matching columns are found, warn and use the provided target as-is.
        full_target = target
    else:
        # Use the provided index (e.g., -1 for the last item)
        try:
            full_target = matching_columns[index]
        except IndexError:
            raise IndexError(f"Index {index} is out of bounds for target levels: {matching_columns}")
    
    # Compute correlation funnel using the full target column name.
    df_correlated = df_binarized.correlate(target=full_target, method=method)
    
    # Attempt to generate a static plot.
    try:
        # Here we assume that your DataFrame has a method plot_correlation_funnel.
        fig = df_correlated.plot_correlation_funnel(engine='plotnine', height=600)
        buf = BytesIO()
        # Use the appropriate save method for your figure object.
        fig.save(buf, format="png")
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        encoded = {"error": str(e)}
    
    # Attempt to generate plotly plot.
    try:
        fig = df_correlated.plot_correlation_funnel(engine='plotly')
        fig_json = pio.to_json(fig)
        fig_dict = json.loads(fig_json)
    except Exception as e:
        fig_dict = {"error": str(e)}

    content = (f"Correlation funnel computed using method '{method}' for target level '{full_target}'. "
               f"Base target was '{target}' with index {index}.")
    artifact = {
        "correlation_data": df_correlated.to_dict(orient="list"),
        "plot_image": encoded,
        "plotly_figure": fig_dict,
    }
    return content, artifact

