
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
    
def extract_txt(file_name):
        """Load a TXT file from the input folder, case-insensitively if needed."""
        # Exact match first
        input_path = os.path.join(os.path.dirname(__file__), 'input')
        full_path = os.path.join(input_path, file_name)
        if os.path.exists(full_path):
            return pd.read_csv(full_path, sep = "\t", encoding="utf-8-sig")
        # Case-insensitive lookup
        fname_lower = file_name.lower()
        for f in os.listdir(input_path):
            if f.lower() == fname_lower:
                return pd.read_csv(os.path.join(input_path, f))
        raise FileNotFoundError(f"Input file not found: {file_name} in {input_path}")
    
def load_csv(file_name, df):
    """Save dataframe as CSV in the output folder."""
    output_path = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, file_name), index=False, sep="\t", encoding="utf-8-sig")


def concat_df_horizontal(df_long, df_short):
        """Concatenate two dataframes horizontally, aligning by index. The shorter dataframe is repeated to match the length of the longer one.
         - df_long: The longer dataframe (e.g., df_to_rep).
         - df_short: The shorter dataframe (e.g., df_samplename).
         - repeat_count: The number of times to repeat the shorter dataframe to ensure it matches the length of the longer one.
         Returns a new dataframe that is the horizontal concatenation of the two input dataframes same length as the longer one."""
        df_long_reset = df_long.reset_index(drop=True)
        df_short_reset = df_short.reset_index(drop=True)
        repeat_count = math.ceil(len(df_long_reset) / len(df_short_reset))
        df_short_repeated = pd.concat([df_short_reset] * repeat_count, ignore_index=True)
        df_short_repeated = df_short_repeated.iloc[:len(df_long_reset)].reset_index(drop=True)

        return pd.concat([df_long_reset, df_short_repeated], axis=1)

def get_aliquot_type(x):
    x = str(x)

    if "BLANK" in x:
        return "BLANK"
    if "LQC" in x:
        return "LQC"
    if "SQC" in x:
        return "SQC"

    # if there is nothing after the number before the next "_", call it sample
    return "sample"

# # Step 1: Load the original dataset and filter it to create the base and replicate datasets.
# df_ori = extract_txt("SM_64.csv")

# # Step 2: Load the samplename dataset and concatenate it with the replicate dataset, repeating as necessary to match the length of the base dataset.
# df_samplename =  extract_csv('samplename.csv')

# df_samplename["Sample Name"] = df_samplename["Sample_Name_64"]
# df_samplename = df_samplename[df_samplename["Sample_ID"]<65][["Sample Name"]]
# print(df_samplename.shape)

# # Step 3: Combine the sample name and df_ori
# df_combined = concat_df_horizontal(df_ori, df_samplename)
# print(df_combined.shape)
# df_combined.drop(columns=["aliquot"], inplace=True)
# df_combined.rename(columns={"Sample Name": "aliquot"}, inplace=True)

# # Step 4: change the type based on aliquot name
# df_combined["type"] = df_combined["aliquot"].apply(get_aliquot_type)

# # Step 5: Concatenate the base dataset with the repeated replicate dataset, ensuring that the final combined dataset has the same number of rows as the base dataset.
# quotient, remainder = divmod(len(df_ori), len(df_combined))
# if remainder != 0:
#       raise ValueError(f"The length of df_ori ({len(df_ori)}) is not a multiple of the length of df_combined ({len(df_combined)}). Cannot repeat df_combined to match the length of df_ori.")
# else:
#       repeat_count = quotient

# # Step 6: Load the mz1 dataset and concatenate it with the combined dataset.
# load_csv('SM_mz1.txt', df_combined)

def build_is_assigned_file(input_file, samplename_col, output_file,
                           samplename_file="samplename.csv",
                           keep_classes=None):
    

    """
    Universal function.

    Parameters
    ----------
    input_file : str
        Input raw csv file, e.g. 'TG_65.csv'
    samplename_col : str
        Column in samplename.csv to use as Sample Name, e.g. 'Sample_Name_65'
    output_file : str
        Output csv filename, e.g. 'TG_mz.csv'
    samplename_file : str
        Samplename csv file, default 'samplename.csv'
    is_pattern : str
        Pattern used to identify IS rows, default 'IS TG'
    """
    desired_row_count = int(samplename_col.split('_')[-1])  # Extract the number from the column name, e.g., 'Sample_Name_65' -> 65

# Step 1: Load original data and check
    df_ori = extract_txt(input_file)
    df_samplename =  extract_csv('samplename.csv')
    if samplename_col not in df_samplename.columns:
        raise ValueError(f"Column '{samplename_col}' not found in {samplename_file}")

    if "Sample_ID" not in df_samplename.columns:
        raise ValueError(f"Column 'Sample_ID' not found in {samplename_file}")

# Step 2: check the class and only keep the desired cleasses if specified

    df_ori["class"] = df_ori["target_class"].str.split('-').str[0]

    unique_classes = df_ori["class"].dropna().unique()
    print(f"Number of unique classes: {len(unique_classes)}")
    print("Unique classes:")
    for cls in unique_classes:
        print(cls)
    
    if keep_classes is not None:
        unique_classes = [cls for cls in unique_classes if cls in keep_classes]
        print(f"Filtered classes to keep: {unique_classes}")
    
    # remove rows from df_ori that are not in the keep_classes list
    if keep_classes is not None:
        df_ori = df_ori[df_ori["class"].isin(unique_classes)]

# Step 3: Combine the sample name and df_ori and rename the columns
    df_samplename["Sample Name"] = df_samplename[samplename_col]
    df_samplename = df_samplename[df_samplename["Sample_ID"] < desired_row_count+1][["Sample Name"]].copy()

    df_combined = concat_df_horizontal(df_ori, df_samplename)
    df_combined.drop(columns=["aliquot"], inplace=True)
    df_combined.rename(columns={"Sample Name": "aliquot"}, inplace=True)

# Step 4: change the type based on aliquot name
    df_combined["type"] = df_combined["aliquot"].apply(get_aliquot_type)
    # print the number of unique aliquot
    print("Number of unique aliquots:", df_combined["aliquot"].nunique())

# Step 5: Remove specific aliquots from the combined dataframe
    df_combined = df_combined[~df_combined["aliquot"].isin(["2312ALU_0001LQC_H1", "2312ALU_0001LQC_H2", "2312ALU_0001LQC_H3", "2312ALU_0001BLANK_B2"])]
    print("Number of unique aliquots:", df_combined["aliquot"].nunique())
    print("Final combined shape:", df_combined.shape)

# Step 6: Save the final combined dataframe to the output file
    load_csv(output_file, df_combined)
    return df_combined

# # run the function with the specified parameters
# df_final = build_is_assigned_file(
#     input_file="tgs.txt",
#     samplename_col="Sample_Name_65",
#     output_file="TG_mz.txt",
#     keep_classes=["TG"],
# )

from pathlib import Path

# # Load and combine all output files
input_path = os.path.join(os.path.dirname(__file__), 'output')
df_result = pd.DataFrame()
all_dfs = []
count = 0

for file in Path(input_path).glob("*.txt"):
    if file.name != "final_mz.txt":  # Exclude the final_mz.csv if it exists
        df_temp = pd.read_csv(str(file), sep="\t")
        count = count +len(df_temp)
        all_dfs.append(df_temp)
        print(f"unique classes in dataframe: {df_temp['class'].unique()}")
  
if all_dfs:
    df_result = pd.concat(all_dfs, ignore_index=True)
    first_injection_time = df_result.groupby("aliquot", sort=False)["injection_time"].first()
    df_result["injection_time"] = df_result["aliquot"].map(first_injection_time)

    load_csv("final_mz.txt", df_result)


