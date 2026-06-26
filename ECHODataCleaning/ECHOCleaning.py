# First step for any ECHO project: Data Cleaning. 
import pandas as pd
import os
import sys
from pathlib import Path
import math

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from DataProcessor.ProcessorClass import ProcessorClass

# Constant for class cleanup mapping, this will change based on each project and how many classes/samples exported
CLASS_CLEANUP_MAPPING = {
    "SM": {
            "input_file": "SM_64.csv",
            "samplename_col": "Sample_Name_64",
            "output_file": "SM_mz.csv",
            "keep_classes": ["SM"],
        },
    # "TG": {
    #         "input_file": "TG_65.csv",
    #         "samplename_col": "Sample_Name_65",
    #         "output_file": "TG_mz.csv",
    #         "keep_classes": ["TG"],
    #     },
    "PCLPC": {
            "input_file": "PC_LPC_68.csv",
            "samplename_col": "Sample_Name_68",
            "output_file": "PCLPC_mz.csv",
            "keep_classes": ["PC", "LPC"],
        },
    "PELPE": {
            "input_file": "PE_LPE_65.csv",
            "samplename_col": "Sample_Name_65",
            "output_file": "PELPE_mz.csv",
            "keep_classes": ["PE","LPE "], 
    },
    "PIPGPS": {
            "input_file": "PI_PG_PS_65.csv",
            "samplename_col": "Sample_Name_65",
            "output_file": "PIPGPS_mz.csv",
            "keep_classes": ["PI","PG","PS"], 
    },
    "CEDG": {
            "input_file": "CE_DG_68.csv",
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
        print(f"====================================Step 3: Merging all intermediate files to {final_output_name}========================")

        for file in Path(input_path).glob("*.csv"):
            if file.name != final_output_name:  # Exclude the final output file if it exists
                df_temp = self.extract_txt(str(file), encoding="utf-8-sig")
                count = count +len(df_temp)
                all_dfs.append(df_temp)
                print(f"Reading intermediate file: {file.name}, shape: {df_temp.shape}")
                print(f"Unique classes in {file.name} dataframe: {df_temp['class'].unique()}")
        
        if all_dfs:
            df_result = pd.concat(all_dfs, ignore_index=True)
            first_injection_time = df_result.groupby("aliquot", sort=False)["injection_time"].first()
            df_result["injection_time"] = df_result["aliquot"].map(first_injection_time)

            self.load_txt(final_output_name, df_result)
  
    def build_is_assigned_file(self, input_file, samplename_col, output_file,
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

        # Step 1: Load original data and check; input files with .csv extension while they should be treated as .txt and read with sep = "\t"
        df_ori = self.extract_txt(input_file, encoding="utf-8-sig")
        df_samplename =  self.extract_csv('samplename.csv')
        if samplename_col not in df_samplename.columns:
            raise ValueError(f"Column '{samplename_col}' not found in {samplename_file}")

        if "Sample_ID" not in df_samplename.columns:
            raise ValueError(f"Column 'Sample_ID' not found in {samplename_file}")

        # Step 2: check the class and only keep the desired cleasses if specified
        print(f"====================================Step 1: Counting the number of unique classes for file {input_file}========================")
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

        df_combined = self.concat_df_horizontal(df_ori, df_samplename)
        df_combined.drop(columns=["aliquot"], inplace=True)
        df_combined.rename(columns={"Sample Name": "aliquot"}, inplace=True)

        # Step 4: change the type based on aliquot name
        df_combined["type"] = df_combined["aliquot"].apply(self.get_aliquot_type)
        # print the number of unique aliquot
        print("Number of unique aliquots:", df_combined["aliquot"].nunique())

        # Step 5: Remove specific aliquots from the combined dataframe
        df_combined = df_combined[~df_combined["aliquot"].isin(["2312ALU_0001LQC_H1", "2312ALU_0001LQC_H2", "2312ALU_0001LQC_H3", "2312ALU_0001BLANK_B2"])]
        print("Number of unique aliquots:", df_combined["aliquot"].nunique())
        print("Final combined shape:", df_combined.shape)

        # Step 6: Save the final combined dataframe to the output file
        print(f"====================================Step 2: Saving intermediate result per class to {output_file}========================")

        self.load_txt(output_file, df_combined)
        return df_combined

    def concat_df_horizontal(self, df_long, df_short):
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

    def get_aliquot_type(self,x):
        x = str(x)

        if "BLANK" in x:
            return "BLANK"
        if "LQC" in x:
            return "LQC"
        if "SQC" in x:
            return "SQC"

        # if there is nothing after the number before the next "_", call it sample
        return "sample"


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
            keep_classes=params["keep_classes"]
        )

    # Merge the class-specific data into final result and clean up accordingly
    cleaning_generator.preprocess(final_output_name='final_mz1.txt') # could keep multiple versions of final_mz.txt for comparison, e.g. final_mz1.txt, final_mz2.txt, etc.

