import pandas as pd
import os
import sys


basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)
from Helpers import detect_changes_between_dataframes


def extract_csv(file_name):
    """Load a CSV file from the input folder, case-insensitively if needed."""
    input_path = os.path.join(os.path.dirname(__file__), 'input')
    full_path = os.path.join(input_path, file_name)

    if os.path.exists(full_path):
        return pd.read_csv(full_path)

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


def concat_df_horizontal(df_long, df_short, repeat_count=1):
    """Concatenate two dataframes horizontally after repeating the shorter one."""
    df_long_reset = df_long.reset_index(drop=True)
    df_short_reset = df_short.reset_index(drop=True)

    df_short_repeated = pd.concat([df_short_reset] * repeat_count, ignore_index=True)
    df_short_repeated = df_short_repeated.iloc[:len(df_long_reset)].reset_index(drop=True)

    return pd.concat([df_long_reset, df_short_repeated], axis=1)


def process_one_class(df_class, df_samplename, class_name, is_pattern_map=None, desired_row_count=None):
    """
    Process one class and assign its corresponding IS rows.

    Parameters
    ----------
    df_class : pd.DataFrame
        Data for one lipid class only
    df_samplename : pd.DataFrame
        Samplename dataframe with column 'Sample Name'
    class_name : str
        Current class, e.g. 'TG', 'PC', 'PE'
    is_pattern_map : dict or None
        Optional mapping, e.g. {'TG': 'IS TG', 'PC': 'IS PC'}
        If None, use default rule: f"IS {class_name}"
    """

    # Decide IS pattern for this class
    if is_pattern_map is not None:
        if class_name not in is_pattern_map:
            print(f"Skip class {class_name}: no IS pattern defined.")
            return pd.DataFrame()
        is_pattern = is_pattern_map[class_name]
    else:
        is_pattern = f"IS {class_name}"

    print(f"Processing class {class_name} with IS pattern: {is_pattern}")

    # IS rows for this class
    df_to_rep = df_class[df_class['Compound Name'].str.contains(is_pattern, na=False, regex=False)].copy()
    df_to_rep = df_to_rep[["sample_id", "Area", "Compound Name"]]
    df_to_rep.rename(columns={
        "Compound Name": "IS Name",
        "Area": "IS Area",
        "sample_id": "Sample_ID"
    }, inplace=True)

    # Base analyte rows for this class
    df_base = df_class[~df_class['Compound Name'].str.contains(is_pattern, na=False, regex=False)].copy()
    df_base = df_base[~df_base['Compound Name'].str.contains("IS", na=False, regex=False)].copy()

    if df_to_rep.empty:
        print(f"Skip class {class_name}: no IS rows found for pattern '{is_pattern}'.")
        return pd.DataFrame()

    if df_base.empty:
        print(f"Skip class {class_name}: no base rows found.")
        return pd.DataFrame()

    df_combined = concat_df_horizontal(df_samplename, df_to_rep, repeat_count=1)

    # Check if the length of df_base is a multiple of the length desired_row_count and if length of df_combined is equal to desired_row_count
    if desired_row_count is not None:
        if len(df_combined) != desired_row_count:
            raise ValueError(f"Class {class_name}: len(df_combined)={len(df_combined)} does not match desired_row_count={desired_row_count}.")
        if len(df_base) % desired_row_count != 0:
            raise ValueError(f"Class {class_name}: len(df_base)={len(df_base)} is not a multiple of desired_row_count={desired_row_count}.")
        
    quotient, remainder = divmod(len(df_base), len(df_combined))

    df_result = concat_df_horizontal(df_base, df_combined, repeat_count=quotient)

    df_result["Retention Time"] = 1
    df_result["IS Retention Time"] = 1
    df_result["Class"] = class_name

    df_result.drop(columns=["sample_id", "Sample_ID"], inplace=True, errors="ignore")
    return df_result

def build_is_assigned_file(input_file, samplename_col, output_file,
                           samplename_file="samplename.csv",
                           is_pattern_map=None,
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

    # Load original data
    df_ori = extract_csv(input_file)
    df_ori = df_ori.drop(columns=["rt_start", "rt_end", "data_points", "parent_mz", "product_mz"], errors="ignore")

    df_ori.rename(columns={
        "run_start_time": "Aquisition Date & Time",
        "name": "Compound Name",
        "area": "Area"
    }, inplace=True)

    if "class" not in df_ori.columns:
        raise ValueError(f"Column 'class' not found in {input_file}")

    df_ori["Class"] = df_ori["class"].str.split('-').str[0]

    unique_classes = df_ori["Class"].dropna().unique()
    print(f"Number of unique classes: {len(unique_classes)}")
    print("Unique classes:")
    for cls in unique_classes:
        print(cls)
    
    if keep_classes is not None:
        unique_classes = [cls for cls in unique_classes if cls in keep_classes]
        print(f"Filtered classes to keep: {unique_classes}")

    # Load samplename file
    df_samplename = extract_csv(samplename_file)

    if samplename_col not in df_samplename.columns:
        raise ValueError(f"Column '{samplename_col}' not found in {samplename_file}")

    if "Sample_ID" not in df_samplename.columns:
        raise ValueError(f"Column 'Sample_ID' not found in {samplename_file}")

    df_samplename["Sample Name"] = df_samplename[samplename_col]
    df_samplename = df_samplename[df_samplename["Sample_ID"] < desired_row_count+1][["Sample Name"]].copy()

    # Process each class
    all_results = []

    for cls in unique_classes:
        df_class = df_ori[df_ori["Class"] == cls].copy()
        result = process_one_class(df_class, df_samplename, cls, is_pattern_map=is_pattern_map, desired_row_count=desired_row_count)

        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise ValueError("No class produced any result.")

    df_final = pd.concat(all_results, ignore_index=True)
    print("Final combined shape:", df_final.shape)

    load_csv(output_file, df_final)
    return df_final



# df_final = build_is_assigned_file(
#     input_file="SM_64.csv",
#     samplename_col="Sample_Name_64",
#     output_file="SM_mz.csv",
#     keep_classes=["SM"],
# )

# modift TG_mz.csv, find the rows that have missing value in run_start_time column and print
# first load the TG_mz.csv file and check the columns
# TG_mz_path = os.path.join(os.path.dirname(__file__), 'output', 'TG_mz.csv')
# df_tg_mz = pd.read_csv(TG_mz_path, sep="\t")
# # find the rows that have missing value in run_start_time column and print
# missing_run_start_time = df_tg_mz[df_tg_mz['Aquisition Date & Time'].isna()]
# print("Rows with missing 'Aquisition Date & Time':")
# print(missing_run_start_time)
# # delete the rows that have missing value in run_start_time column and load the modified dataframe to TG_mz.csv
# df_tg_mz_cleaned = df_tg_mz.dropna(subset=['Aquisition Date & Time'])
# load_csv('TG_mz.csv', df_tg_mz_cleaned) 

# # extract the rows for TG42:0-FA16:0 
# replicate_col = df_tg_mz[df_tg_mz['Compound Name'] == 'TG42:0-FA16:0']
# # add a new column from 1 to 130 to the replicate_col dataframe and name it 'Replicate Number'
# print("Shape of replicate_col:", replicate_col.shape)
# print(replicate_col.head())
# # keep the last half rows
# replicate_col = replicate_col.iloc[len(replicate_col)//2:]

# # delete the rows in df_tg_mz that are in replicate_col
# df_tg_mz = df_tg_mz[~df_tg_mz.index.isin(replicate_col.index)]  
# # load the replicate_col dataframe to TG_mz_replicate.csv
# load_csv('TG_mz.csv', df_tg_mz)

from pathlib import Path

# # Load and combine all output files
input_path = os.path.join(os.path.dirname(__file__), 'output')
df_result = pd.DataFrame()
all_dfs = []
count = 0

for file in Path(input_path).glob("*.csv"):
    if file.name != "final_mz.csv":  # Exclude the final_mz.csv if it exists
        df_temp = pd.read_csv(str(file), sep="\t")
        count = count +len(df_temp)
        all_dfs.append(df_temp)
        print(f"unique classes in dataframe: {df_temp['Class'].unique()}")
  


if all_dfs:
    df_result = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows in combined dataframe: {len(df_result)}")
    print(f"Total rows processed: {count}")
    print(f"unique classes in combined dataframe: {df_result['Class'].unique()}")
    print("check the NA in class column:", df_result[df_result['Class'].isna()])
    
    df_result = df_result.rename(columns={
    "Aquisition Date & Time": "Acquisition Date & Time",
    "Compound Name": "Component Name"})

    df_result = df_result[[
        "Sample Name",
        "Acquisition Date & Time",
        "Component Name",
        "IS Name",
        "Area",
        "IS Area",
        "Retention Time",
        "IS Retention Time",
        "class"]]
    
    df_result["Acquisition Date & Time"] = pd.to_datetime(
        df_result["Acquisition Date & Time"],
        errors="coerce"
    ).dt.strftime("%d/%m/%Y %H:%M:%S")
    
    # remove rows that have "2312ALU_0001LQC_H1", "2312ALU_0001LQC_H2", "2312ALU_0001LQC_H3" 2312ALU_0001BLANK_B2"in the Sample Name column
    df_result = df_result[~df_result["Sample Name"].isin(["2312ALU_0001LQC_H1", "2312ALU_0001LQC_H2", "2312ALU_0001LQC_H3", "2312ALU_0001BLANK_B2"])]
    print(f"Total rows after removing specific Sample Names: {len(df_result)}")

    # use N/A to replace 0.0 in Area and IS Area columns 
    df_result["Area"] = df_result["Area"].replace(0.0, "N/A")
    df_result["IS Area"] = df_result["IS Area"].replace(0.0, "N/A") 

    # order the rows by Sample Name, then by Component Name
    df_result = df_result.sort_values(by=["Sample Name", "Component Name"]).reset_index(drop=True)  

    for col in ["Component Name", "IS Name"]:
        df_result[col] = (
            df_result[col]
            .astype(str)
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace(":", ".", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    
    counts = df_result.groupby("Component Name").size()
    for comp, n in counts.items():
        if n != 64:
            print(f"{comp}: {n}")

    
    # wrong_counts = counts[counts != 64]
    # print(wrong_counts)

    load_csv('final_mz1.txt', df_result)
else:
    print("No CSV files found in output folder.")


