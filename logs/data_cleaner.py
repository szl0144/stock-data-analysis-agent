def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove columns with more than 40% missing values
    threshold = 0.4 * len(data_raw)
    data_cleaned = data_raw.dropna(thresh=threshold, axis=1)

    # Step 2: Impute missing values
    for column in data_cleaned.columns:
        if data_cleaned[column].dtype == 'object':
            # Impute categorical columns with mode
            imputer = SimpleImputer(strategy='most_frequent')
            data_cleaned[column] = imputer.fit_transform(data_cleaned[[column]]).ravel()
        else:
            # Impute numeric columns with mean
            imputer = SimpleImputer(strategy='mean')
            data_cleaned[column] = imputer.fit_transform(data_cleaned[[column]]).ravel()

    # Step 3: Convert columns to correct data types
    # Convert 'TotalCharges' to numeric, forcing errors to NaN (in case of strings)
    data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')
    
    # Step 4: Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Step 5: Remove rows with any missing values
    data_cleaned = data_cleaned.dropna()

    # Return the cleaned DataFrame
    return data_cleaned