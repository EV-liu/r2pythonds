import pandas as pd

def enforce_schema(df, schema_class):
    """Ensure the DataFrame matches the given schema."""
    schema_fields = schema_class.__annotations__  # Get field names and types from the dataclass
    for column, dtype in schema_fields.items():
        if column in df.columns:
            # Convert column to the expected type
            if dtype == pd.Categorical:
                df[column] = pd.Categorical(df[column])  # Convert to categorical
            elif dtype == int:
                df[column] = df[column].fillna(0).astype('int')
            elif dtype == float:
                df[column] = df[column].fillna(0.0).astype('float')
            elif dtype == str:
                df[column] = df[column].fillna("").astype('str')
        else:
            # Add missing columns with default values
            if dtype == pd.Categorical:
                df[column] = pd.Categorical([])
            elif dtype == int:
                df[column] = 0
            elif dtype == float:
                df[column] = 0.0
            elif dtype == str:
                df[column] = ""
    return df