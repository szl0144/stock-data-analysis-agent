def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove columns with more than 40% missing values
    threshold = 0.4 * len(data_raw)
    data_cleaned = data_raw.dropna(axis=1, thresh=threshold)

    # Step 2: Separate numeric and categorical columns
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = data_cleaned.select_dtypes(include=[object]).columns

    # Step 3: Impute missing values
    # Impute numeric columns with mean
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        data_cleaned[numeric_cols] = num_imputer.fit_transform(data_cleaned[numeric_cols])

    # Impute categorical columns with mode
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data_cleaned[categorical_cols] = cat_imputer.fit_transform(data_cleaned[categorical_cols])

    # Step 4: Convert columns to the correct data type
    data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')

    # Step 5: Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Step 6: Remove rows with remaining missing values
    data_cleaned = data_cleaned.dropna()

    # Note: Step for removing outliers is skipped based on user instructions

    return data_cleaned