
import pandas as pd
import os
import sys
import math
import pandas as pd
import random

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)
from Helpers import detect_changes_between_dataframes


def extract_csv(file_name):
        """Load a CSV file from the input folder, case-insensitively if needed."""
        # Exact match first
        input_path = os.path.join(os.path.dirname(__file__), 'input')
        full_path = os.path.join(input_path, file_name)
        if os.path.exists(full_path):
            return pd.read_csv(full_path)
        # Case-insensitive lookup
        fname_lower = file_name.lower()
        for f in os.listdir(input_path):
            if f.lower() == fname_lower:
                return pd.read_csv(os.path.join(input_path, f))
        raise FileNotFoundError(f"Input file not found: {file_name} in {input_path}")
    
def load_csv(file_name, df):
        """Save result dataframes as CSV files in the output folder."""
        output_path = os.path.join(os.path.dirname(__file__), 'output')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        df.to_csv(os.path.join(output_path, file_name), index=False)

def concat_df_horizontal(df_long, df_short, repeat_count=1):
        """Concatenate two dataframes horizontally, aligning by index. The shorter dataframe is repeated to match the length of the longer one.
         - df_long: The longer dataframe (e.g., df_to_rep).
         - df_short: The shorter dataframe (e.g., df_samplename).
         - repeat_count: The number of times to repeat the shorter dataframe to ensure it matches the length of the longer one.
         Returns a new dataframe that is the horizontal concatenation of the two input dataframes same length as the longer one."""
        df_long_reset = df_long.reset_index(drop=True)
        df_short_reset = df_short.reset_index(drop=True)
        df_short_repeated = pd.concat([df_short_reset] * repeat_count, ignore_index=True)
        df_short_repeated = df_short_repeated.iloc[:len(df_long_reset)].reset_index(drop=True)

        return pd.concat([df_long_reset, df_short_repeated], axis=1)

# Step 1: Load the original dataset and filter it to create the base and replicate datasets.
df_ori = extract_csv('TG.csv')
print(df_ori.shape)

df_to_rep = df_ori[df_ori['name'].str.contains("IS TG", na=False)]
df_to_rep = df_to_rep[["sample_id", "area", "name"]]
df_to_rep["IS_name"] = df_to_rep["name"]
df_to_rep = df_to_rep.drop(columns=["name"])
df_to_rep["IS_area"] = df_to_rep["area"]
df_to_rep = df_to_rep.drop(columns=["area"])
df_to_rep["Sample_ID"] = df_to_rep["sample_id"]
df_to_rep = df_to_rep.drop(columns=["sample_id"])
print(df_to_rep.shape)

df_base = df_ori[df_ori['name'].str.contains("IS TG", na=False) == False]
df_base = df_base[df_base['name'].str.contains("IS", na=False) == False]
print(df_base.shape)

# Step 2: Load the samplename dataset and concatenate it with the replicate dataset, repeating as necessary to match the length of the base dataset.
df_samplename =  extract_csv('samplename.csv')
df_samplename["Sample_Name"] = df_samplename["Sample Name"]
df_samplename = df_samplename.drop(columns=["Sample Name"])
print(df_samplename.shape)

df_combined = concat_df_horizontal(df_samplename, df_to_rep, repeat_count=1)
print(df_combined.shape)

# Step 3: Concatenate the base dataset with the repeated replicate dataset, ensuring that the final combined dataset has the same number of rows as the base dataset.
repeat_count = math.ceil(len(df_base) / len(df_combined))
df_combined_repeated = concat_df_horizontal(df_base, df_combined, repeat_count=repeat_count)

# Step 4: Verify that the final combined dataset has correct repeated rows and the same number of rows as the base dataset
repeat_length = len(df_combined)
df_check1 = df_combined_repeated.iloc[:repeat_length]
i = random.randint(1,repeat_count-1)
df_check_random = df_combined_repeated.iloc[(i-1)*repeat_length:i*repeat_length]

check_df = detect_changes_between_dataframes(
    df_check1,
    df_check_random,
    check_columns=['IS_name', "IS_area", 'Sample_Name', "Sample_ID"],
    unique_key=['sample_id'],
    detect_column_changes=True
)

check_df = check_df[['sample_id', 'change_type', 'changes']]
check_result = check_df[check_df['change_type'] != 'unchanged']
print(f"Check if repeat correctly: {check_result.shape[0]} rows changed, should be 0 rows changed.")

# Step 5: Save the final combined dataset as a CSV file in the output folder.
load_csv('TG_replicate.csv', df_combined_repeated)

