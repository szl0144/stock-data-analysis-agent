import io
import pandas as pd
import sqlalchemy as sql
from sqlalchemy import inspect
from typing import Union, List, Dict

def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 30,
    skip_stats: bool = False,
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
    n_sample : int, default 30
        Number of rows to display in the "Data (first 30 rows)" section.
    skip_stats : bool, default False
        If True, skip the descriptive statistics and DataFrame info sections.
        
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
            summaries.append(_summarize_dataframe(df, dataset_name, n_sample, skip_stats))

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(_summarize_dataframe(dataframes, "Single_Dataset", n_sample, skip_stats))

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(_summarize_dataframe(df, dataset_name, n_sample, skip_stats))

    else:
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries


def _summarize_dataframe(df: pd.DataFrame, dataset_name: str, n_sample=30, skip_stats=False) -> str:
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

    # 6. Generate the summary text
    if not skip_stats:
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

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}

        Data Description:
        {df.describe().to_string()}

        Data Info:
        {info_text}
        """
    else:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}
        """
        
    return summary_text.strip()



def get_database_metadata(connection, n_samples=10) -> dict:
    """
    Collects metadata and sample data from a database, with safe identifier quoting and
    basic dialect-aware row limiting. Prevents issues with spaces/reserved words in identifiers.
    
    Parameters
    ----------
    connection : Union[sql.engine.base.Connection, sql.engine.base.Engine]
        An active SQLAlchemy connection or engine.
    n_samples : int
        Number of sample values to retrieve for each column.

    Returns
    -------
    dict
        A dictionary with database metadata, including some sample data from each column.
    """
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    metadata = {
        "dialect": None,
        "driver": None,
        "connection_url": None,
        "schemas": [],
    }

    try:
        sql_engine = conn.engine
        dialect_name = sql_engine.dialect.name.lower()

        metadata["dialect"] = sql_engine.dialect.name
        metadata["driver"] = sql_engine.driver
        metadata["connection_url"] = str(sql_engine.url)

        inspector = inspect(sql_engine)
        preparer = inspector.bind.dialect.identifier_preparer

        # For each schema
        for schema_name in inspector.get_schema_names():
            schema_obj = {
                "schema_name": schema_name,
                "tables": []
            }

            tables = inspector.get_table_names(schema=schema_name)
            for table_name in tables:
                table_info = {
                    "table_name": table_name,
                    "columns": [],
                    "primary_key": [],
                    "foreign_keys": [],
                    "indexes": []
                }
                # Get columns
                columns = inspector.get_columns(table_name, schema=schema_name)
                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"])
                    table_name_quoted = f"{preparer.quote_identifier(schema_name)}.{preparer.quote_identifier(table_name)}"
                    col_name_quoted = preparer.quote_identifier(col_name)

                    # Build query for sample data
                    query = build_query(col_name_quoted, table_name_quoted, n_samples, dialect_name)

                    # Retrieve sample data
                    try:
                        df = pd.read_sql(query, conn)
                        samples = df[col_name].head(n_samples).tolist()
                    except Exception as e:
                        samples = [f"Error retrieving data: {str(e)}"]

                    table_info["columns"].append({
                        "name": col_name,
                        "type": col_type,
                        "sample_values": samples
                    })

                # Primary keys
                pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name)
                table_info["primary_key"] = pk_constraint.get("constrained_columns", [])

                # Foreign keys
                fks = inspector.get_foreign_keys(table_name, schema=schema_name)
                table_info["foreign_keys"] = [
                    {
                        "local_cols": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_cols": fk["referred_columns"]
                    }
                    for fk in fks
                ]

                # Indexes
                idxs = inspector.get_indexes(table_name, schema=schema_name)
                table_info["indexes"] = idxs

                schema_obj["tables"].append(table_info)
            
            metadata["schemas"].append(schema_obj)
    
    finally:
        if is_engine:
            conn.close()

    return metadata

def build_query(col_name_quoted: str, table_name_quoted: str, n: int, dialect_name: str) -> str:
    # Example: expand your build_query to handle random sampling if possible
    if "postgres" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RANDOM() LIMIT {n}"
    if "mysql" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RAND() LIMIT {n}"
    if "sqlite" in dialect_name:
        return f"SELECT {col_name_quoted} FROM {table_name_quoted} ORDER BY RANDOM() LIMIT {n}"
    if "mssql" in dialect_name:
        return f"SELECT TOP {n} {col_name_quoted} FROM {table_name_quoted} ORDER BY NEWID()"
    # Oracle or fallback
    return f"SELECT {col_name_quoted} FROM {table_name_quoted} WHERE ROWNUM <= {n}"

