def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Step 1: Remove columns with more than 40% missing values
    threshold = 0.4 * len(data_raw)
    data_cleaned = data_raw.dropna(thresh=threshold, axis=1)

    # Step 2: Impute missing values
    for column in data_cleaned.columns:
        if data_cleaned[column].dtype == 'object':  # Categorical column
            imputer = SimpleImputer(strategy='most_frequent')
            data_cleaned[column] = imputer.fit_transform(data_cleaned[[column]]).ravel()
        else:  # Numeric column
            imputer = SimpleImputer(strategy='mean')
            data_cleaned[column] = imputer.fit_transform(data_cleaned[[column]]).ravel()

    # Step 3: Convert columns to the correct data type
    # Assuming 'SeniorCitizen' is binary and should be int
    if 'SeniorCitizen' in data_cleaned.columns:
        data_cleaned['SeniorCitizen'] = data_cleaned['SeniorCitizen'].astype(int)

    # Convert 'TotalCharges' to numeric, errors='coerce' will turn invalid parsing into NaN
    if 'TotalCharges' in data_cleaned.columns:
        data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')

    # Step 4: Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Step 5: Remove rows with missing values
    data_cleaned = data_cleaned.dropna()

    # Step 6: Remove rows with extreme outliers
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        # Calculate the IQR
        Q1 = data_cleaned[column].quantile(0.25)
        Q3 = data_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers
        data_cleaned = data_cleaned[(data_cleaned[column] >= lower_bound) & (data_cleaned[column] <= upper_bound)]

    return data_cleaned