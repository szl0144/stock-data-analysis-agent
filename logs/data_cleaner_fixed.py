def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove columns with more than 40% missing values
    threshold = 0.4 * len(data_raw)
    data_cleaned = data_raw.dropna(thresh=len(data_raw) - threshold, axis=1)

    # Step 2: Impute missing values
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
    categorical_columns = data_cleaned.select_dtypes(include=[object]).columns

    # Impute numeric columns with mean
    if len(numeric_columns) > 0:  # Check if there are numeric columns to impute
        imputer_num = SimpleImputer(strategy='mean')
        data_cleaned[numeric_columns] = imputer_num.fit_transform(data_cleaned[numeric_columns])

    # Impute categorical columns with mode
    if len(categorical_columns) > 0:  # Check if there are categorical columns to impute
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data_cleaned[categorical_columns] = imputer_cat.fit_transform(data_cleaned[categorical_columns])

    # Step 3: Convert columns to the correct data type
    # Convert TotalCharges to numeric, handling errors
    if 'TotalCharges' in data_cleaned.columns:  # Check if 'TotalCharges' exists
        data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')

    # Step 4: Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Step 5: Remove rows with any missing values
    data_cleaned = data_cleaned.dropna()

    return data_cleaned