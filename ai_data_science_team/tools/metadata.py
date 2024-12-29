import io
import pandas as pd
import sqlalchemy as sql
from typing import Union, List, Dict

def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]]
) -> List[str]:
    """
    Generate a summary for one or more DataFrames. Accepts a single DataFrame, a list of DataFrames,
    or a dictionary mapping names to DataFrames.

    Parameters
    ----------
    dataframes : pandas.DataFrame or list of pandas.DataFrame or dict of (str -> pandas.DataFrame)
        - Single DataFrame: produce a single summary (returned within a one-element list).
        - List of DataFrames: produce a summary for each DataFrame, using index-based names.
        - Dictionary of DataFrames: produce a summary for each DataFrame, using dictionary keys as names.
        
    Example:
    --------
    ``` python
    import pandas as pd
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    dataframes = {
        "iris": data.frame,
        "iris_target": data.target,
    }
    summaries = get_dataframe_summary(dataframes)
    print(summaries[0])
    ```

    Returns
    -------
    list of str
        A list of summaries, one for each provided DataFrame. Each summary includes:
        - Shape of the DataFrame (rows, columns)
        - Column data types
        - Missing value percentage
        - Unique value counts
        - First 30 rows
        - Descriptive statistics
        - DataFrame info output
    """

    summaries = []

    # --- Dictionary Case ---
    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(_summarize_dataframe(df, dataset_name))

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(_summarize_dataframe(dataframes, "Single_Dataset"))

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(_summarize_dataframe(df, dataset_name))

    else:
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries


def _summarize_dataframe(df: pd.DataFrame, dataset_name: str) -> str:
    """Generate a summary string for a single DataFrame."""
    # 1. Convert dictionary-type cells to strings
    #    This prevents unhashable dict errors during df.nunique().
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))
    
    # 2. Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    # 3. Calculate missing value stats
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])

    # 4. Get column data types
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    # 5. Get unique value counts
    unique_counts = df.nunique()  # Will no longer fail on unhashable dict
    unique_counts_summary = "\n".join([f"{col}: {count}" for col, count in unique_counts.items()])

    summary_text = f"""
    Dataset Name: {dataset_name}
    ----------------------------
    Shape: {df.shape[0]} rows x {df.shape[1]} columns

    Column Data Types:
    {column_types}

    Missing Value Percentage:
    {missing_summary}

    Unique Value Counts:
    {unique_counts_summary}

    Data (first 30 rows):
    {df.head(30).to_string()}

    Data Description:
    {df.describe().to_string()}

    Data Info:
    {info_text}
    """
    return summary_text.strip()


def get_database_metadata(connection: Union[sql.engine.base.Connection, sql.engine.base.Engine], n_values: int=10):
    """
    Collects metadata and sample data from a database.

    Parameters:
    -----------
    connection (sqlalchemy.engine.base.Connection or sqlalchemy.engine.base.Engine): 
        An active SQLAlchemy connection or engine.
    n_values (int): 
        Number of sample values to retrieve for each column.

    Returns:
    --------
    str: Formatted text with database metadata.
    """
    # If a connection is passed, use it; if an engine is passed, connect to it
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection
    output = []

    try:
        # Engine metadata
        sql_engine = conn.engine
        output.append(f"Database Dialect: {sql_engine.dialect.name}")
        output.append(f"Driver: {sql_engine.driver}")
        output.append(f"Connection URL: {sql_engine.url}")
        
        # Inspect the database
        inspector = sql.inspect(sql_engine)
        output.append(f"Tables: {inspector.get_table_names()}")
        output.append(f"Schemas: {inspector.get_schema_names()}")
        
        # For each table, get the columns and their metadata
        for table_name in inspector.get_table_names():
            output.append(f"\nTable: {table_name}")
            for column in inspector.get_columns(table_name):
                output.append(f"  Column: {column['name']} Type: {column['type']}")
                # Fetch sample values for the column
                query = f"SELECT {column['name']} FROM {table_name} LIMIT {n_values}"
                data = pd.read_sql(query, sql_engine)
                output.append(f"    First {n_values} Values: {data.values.flatten().tolist()}")
    finally:
        # Close connection if it was created inside this function
        if is_engine:
            conn.close()
    
    # Join all collected information into a single string
    return "\n".join(output)
