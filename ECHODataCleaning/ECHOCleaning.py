# First step for any ECHO project: Data Cleaning. 
import pandas as pd
import os
import sys
from pathlib import Path

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import concat_df_horizontal, detect_changes_between_dataframes
from DataProcessor.ProcessorClass import ProcessorClass

# Constant for class cleanup mapping, this will change based on each project and how many classes/samples exported
CLASS_CLEANUP_MAPPING = {
    "SM": {
            "input_file": "SM_64.csv",
            "samplename_col": "Sample_Name_64",
            "output_file": "SM_mz.csv",
            "keep_classes": ["SM"],
        },
    "TG": {
            "input_file": "TG_65.csv",
            "samplename_col": "Sample_Name_65",
            "output_file": "TG_mz.csv",
            "keep_classes": ["TG"],
        },
    "PCLPC": {
            "input_file": "PCLPC_68.csv",
            "samplename_col": "Sample_Name_68",
            "output_file": "PCLPC_mz.csv",
            "keep_classes": ["PC", "LPC"],
        },
    "PELPE": {
            "input_file": "PELPE_65.csv",
            "samplename_col": "Sample_Name_65",
            "output_file": "PELPE_mz.csv",
            "keep_classes": ["PE","LPE "], 
    },
    "PIPGPS": {
            "input_file": "PIPGPS_65.csv",
            "samplename_col": "Sample_Name_65",
            "output_file": "PIPGPS_mz.csv",
            "keep_classes": ["PI","PG","PS"], 
    },
    "CEDG": {
            "input_file": "CEDG_68.csv",
            "samplename_col": "Sample_Name_68",
            "output_file": "CEDG_mz.csv",
            "keep_classes": ["CE","DG"], 
    },
}

class ECHOCleaning(ProcessorClass):
    """This class includes methods for loading excel files, mergeing multiple files, droping duplicates and filtering."""

    def __init__(self):
        super().__init__(__file__)
        self.output_copy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Preprocessing', 'input')
        self.dataframes = {}

    def preprocess(self, final_output_name='final_mz1.txt'):
        """Preprocess the data loaded, cleanup and merge."""
        ## Load and combine all output files
        input_path = os.path.join(os.path.dirname(__file__), 'output')
        df_result = pd.DataFrame()
        all_dfs = []
        count = 0

        for file in Path(input_path).glob("*.csv"):
            if file.name != "final_mz.csv":  # Exclude the final_mz.csv if it exists
                df_temp = self.extract_csv(str(file))
                count = count +len(df_temp)
                all_dfs.append(df_temp)

        if all_dfs:
            df_result = pd.concat(all_dfs, ignore_index=True)
            print(f"====================================Step 6: Merging all class-specific data into final_mz.csv. Total rows in combined dataframe: {len(df_result)}========================")
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
            self.load_csv(final_output_name, df_result)
        else:
            print("No CSV files found in output folder.")
  
    def process_one_class(self, df_class, df_samplename, class_name, is_pattern_map=None, desired_row_count=None):
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

        print(f"====================================Step 2: Start processing class {class_name} with IS pattern: {is_pattern}========================")

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
    
    def build_is_assigned_file(self, input_file, samplename_col, output_file,
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
        is_pattern_map : dict
            Dictionary mapping class names to IS patterns, default None
        keep_classes : list
            List of classes to keep, default None
        """
        desired_row_count = int(samplename_col.split('_')[-1])  # Extract the number from the column name, e.g., 'Sample_Name_65' -> 65

        # Load original data
        df_ori = self.extract_csv(input_file)
        if df_ori.empty:
            raise ValueError(f"Input file {input_file} is empty or not found.")
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
        print(f"====================================Step 1: Counting the number of unique classes for file {input_file}========================")
        print("Unique classes:")
        for cls in unique_classes:
            print(cls)
        
        if keep_classes is not None:
            unique_classes = [cls for cls in unique_classes if cls in keep_classes]
            print(f"Filtered classes to keep: {unique_classes}")

        # Load samplename file
        df_samplename = self.extract_csv(samplename_file)

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
            result = self.process_one_class(df_class, df_samplename, cls, is_pattern_map=is_pattern_map, desired_row_count=desired_row_count)

            if not result.empty:
                all_results.append(result)

        if not all_results:
            raise ValueError("No class produced any result.")

        df_final = pd.concat(all_results, ignore_index=True)
        print(f"====================================Step 3: Loading csv for {output_file}. Final combined shape: {df_final.shape}========================")

        self.load_csv(output_file, df_final)
        return df_final
    
    def Validate_ADT_and_duplicates_per_class_file(self, filename):
        """Validate the ADT per class file."""
        print(f"====================================Step 4: Validating Aquisition Date & Time for {filename}========================")

        file_path = os.path.join(os.path.dirname(__file__), 'output', filename)
        df = self.extract_csv(file_path)
        if df.empty:
            raise ValueError(f"Input file {file_path} is empty or not found.")
        
        # find the rows that have missing value in run_start_time column and print
        missing_run_start_time = df[df['Aquisition Date & Time'].isna()]
        print("Rows with missing 'Aquisition Date & Time':")
        print(missing_run_start_time)

        print(f"====================================Step 5: Validating duplicates for {filename}============")
        # find the rows that have duplicate values in Sample Name and Compound Name columns and print
        duplicate_rows = df[df.duplicated(subset=['Sample Name', 'Compound Name'], keep=False)]
        print("Duplicate rows based on 'Sample Name'+'Compound Name':")
        print(duplicate_rows)

        issues = []
        if not missing_run_start_time.empty:
            issues.append(
                f"{len(missing_run_start_time)} rows have missing 'Aquisition Date & Time'"
            )

        if not duplicate_rows.empty:
            issues.append(
                f"{len(duplicate_rows)} duplicate rows found based on 'Sample Name' and 'Compound Name'"
            )

        return filename, issues
    

if __name__ == "__main__":
    # Create an instance of the Cleaning class
    cleaning_generator = ECHOCleaning()
    # Create one csv file for each class, e.g. TG_65.csv, PC_65.csv, PE_65.csv, etc.
    for class_name, params in CLASS_CLEANUP_MAPPING.items():
        cleaning_generator.build_is_assigned_file(
            input_file=params["input_file"],
            samplename_col=params["samplename_col"],
            output_file=params["output_file"],
            samplename_file="samplename.csv",
            is_pattern_map=None if not params.get("is_pattern_map") else params["is_pattern_map"],
            keep_classes=params["keep_classes"]
        )
    # Validate for each class-specific file before merging into final_mz.csv
    validation_issues = {}
    for class_name, params in CLASS_CLEANUP_MAPPING.items():
        filename, fileissues = cleaning_generator.Validate_ADT_and_duplicates_per_class_file(params["output_file"])
        if fileissues:
            validation_issues[filename] = fileissues
    if validation_issues:
        print("Validation issues found:")
        for filename, issues in validation_issues.items():
            print(f"{filename}: {', '.join(issues)}")
        raise ValueError("Please fix the above issues before proceeding to final merging.")
    # Merge the class-specific data into final result and clean up accordingly
    cleaning_generator.preprocess(final_output_name='final_mz1.txt') # could keep multiple versions of final_mz.txt for comparison, e.g. final_mz1.txt, final_mz2.txt, etc.

