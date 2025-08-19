import pandas as pd
import numpy as np
import warnings
from typing import Union, List

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
                df[column] = df[column].fillna("").astype('str').str.strip()
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

def enforce_numeric_datatype(df: pd.DataFrame, columns: Union[str, List[str]]):
    """
    Ensure specified columns in the DataFrame are numeric.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (Union[str, List[str]]): Column name or list of column names to enforce numeric type.
    
    Returns:
    pd.DataFrame: DataFrame with specified columns converted to numeric type.
    """ 
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        else:
            warnings.warn(f"Column '{column}' not found in DataFrame.")
    
    return df

def detect_changes_between_dataframes(
    df_old: pd.DataFrame,
    df_actual: pd.DataFrame,
    check_columns: list,
    unique_key: list[str],
    detect_column_changes=False
):
    """
    This function reads dataframes, detects the changes between them and flags the change details, returning a dataframe.
    ----------
    :param df_old: A dataframe with the old values
    :param df_actual: A dataframe with the actual value. This one will be compared to the old_df
    :param check_columns: list of column(s) which you want to be used to check for changes in data
    :param unique_key: list of column(s) which you want to be used in order to group data. Must be a list of strings, even for a single column.
    :param detect_column_changes: detect new column as change
    :return: Returns a dataframe with desired check_columns and 2 new columns change_type (deleted, new or edited), changes (contains a dict with the changed fields and their old/new values)
    """
    if not df_old.duplicated(subset=unique_key).sum() == 0:
        print("Duplicated in df_old records:")
        print(df_old[df_old.duplicated(subset=unique_key, keep=False)].to_string())
        raise ValueError('The unique_key columns are not unique in the old dataframe')
    if not df_actual.duplicated(subset=unique_key).sum() == 0:
        print("Duplicated in df_actual records:")
        print(df_actual[df_actual.duplicated(subset=unique_key, keep=False)].to_string())
        raise ValueError('The unique_key columns are not unique in the actual dataframe')
    
    # Detect column changes and add missing columns to both dataframes
    if detect_column_changes:
        deleted_columns = [column for column in df_old.columns.values if column not in df_actual.columns.values]
        added_columns = [column for column in df_actual.columns.values if column not in df_old.columns.values]
        df_old[added_columns] = [pd.NA] * len(added_columns)
        # set values of columns to object because one of the dataframes only contains NA values
        df_old = df_old.astype(dtype={key: 'object' for key in added_columns})
        df_actual = df_actual.astype(dtype={key: 'object' for key in added_columns})
        df_actual[deleted_columns] = [pd.NA] * len(deleted_columns)
        df_actual = df_actual.astype(dtype={key: 'object' for key in deleted_columns})
        df_old = df_old.astype(dtype={key: 'object' for key in deleted_columns})

    # Check the datatype of columns in both dataframes
    columns_to_check = check_columns + unique_key
    for column in columns_to_check:
        # int64 and float64 are an exception: a combination of these two types works fine
        if not (df_old[column].dtype in ['int64', 'float64'] and df_actual[column].dtype in ['int64', 'float64']) \
                and not df_old[column].dtype == df_actual[column].dtype:
            raise ValueError(f"The types of the column '{column}' do not correspond between df_old ("
                             f"{df_old[column].dtype}) and df_actual ({df_actual[column].dtype}).")
    
    # Detect changes in column values
    merged_df = pd.merge(df_old, df_actual, on=unique_key, how='outer', suffixes=('_old', '_actual'), indicator=True)

    def detect_row_changes(row):
        changes = {}
        for column in check_columns:
            old_value = row.get(f"{column}_old", pd.NA)
            new_value = row.get(f"{column}_actual", pd.NA)
            if not (pd.isna(old_value) and pd.isna(new_value)):  # Skip if both are NA
                if pd.isna(old_value) or pd.isna(new_value) or old_value != new_value:
                    changes[column] = {'old': old_value, 'new': new_value}
        return changes

    merged_df['changes'] = merged_df.apply(detect_row_changes, axis=1)

    # Determine change type
    def determine_change_type(row):
        if row['_merge'] == 'left_only':
            return 'deleted'
        elif row['_merge'] == 'right_only':
            return 'new'
        elif row['changes']:
            return 'edited'
        return 'unchanged'

    merged_df['change_type'] = merged_df.apply(determine_change_type, axis=1)

    # Filter relevant columns for the result
    check_columns_actual = [f"{col}_actual" for col in check_columns if col not in unique_key]
    result_columns = list(set(unique_key + check_columns_actual + ['change_type', 'changes']))
    result_df = merged_df[result_columns]
    return result_df


