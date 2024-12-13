def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove columns with more than 40% missing values
    threshold = 0.4 * len(data_raw)
    data_cleaned = data_raw.loc[:, data_raw.isnull().sum() <= threshold]

    # Step 2: Impute missing values
    # For numeric columns, use mean imputation
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
    imputer_numeric = SimpleImputer(strategy='mean')
    data_cleaned[numeric_cols] = imputer_numeric.fit_transform(data_cleaned[numeric_cols])

    # For categorical columns, use mode imputation
    categorical_cols = data_cleaned.select_dtypes(include=[object]).columns
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    data_cleaned[categorical_cols] = imputer_categorical.fit_transform(data_cleaned[categorical_cols])

    # Step 3: Convert columns to the correct data types
    # Convert 'TotalCharges' to numeric, errors='coerce' will convert non-convertible values to NaN
    data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')

    # Step 4: Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Step 5: Remove rows with missing values
    data_cleaned = data_cleaned.dropna()

    # Note: According to user instructions, we will NOT remove outliers
    # If it were to be done, it would involve calculating the IQR and filtering based on that.

    return data_cleaned