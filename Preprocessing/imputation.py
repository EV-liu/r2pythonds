import pandas as pd
import numpy as np
import os
import sys
import types, numpy as _np
from datetime import datetime

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import impute_left_censored, enforce_numeric_datatype, detect_changes_between_dataframes
from Schema.sample_schema import SampleSchema

R_IMPUTED = "R_imputed.csv"

class Imputation:
    """This class includes methods for imputing missing values in the dataframes."""

    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(__file__), 'input')
        self.output_path = os.path.join(os.path.dirname(__file__), 'output')
        self.output_copy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PCA', 'input')
        self.dataframes = {}
        # Add a timestamp attribute for file naming
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def impute_missing_values(self):
        """Impute missing values in the dataframe."""
        # load input file name and dataframe
        file_names = [f for f in os.listdir(self.input_path) if os.path.isfile(os.path.join(self.input_path, f)) and f.lower().endswith('.csv')]

        for file_name in file_names:
            if file_name.startswith('merged_DataCleaning'):
                impute_file = file_name
            elif file_name.startswith('sample'):
                sampleId_file = file_name

        if impute_file and sampleId_file:
            df = self.extract_csv(impute_file)

            # step 1 exclude injectionID for imputation steps
            injection_ids = df[['injectionID', 'SampleID', 'Group']]
            numeric_df = df.drop(columns=['injectionID', 'SampleID', 'Group'], errors='ignore')

            # step 2 check the missing rate
            # Coerce all remaining columns to numeric (strings -> numbers, invalid -> NaN)
            numeric_df = enforce_numeric_datatype(numeric_df, (set(numeric_df.columns)))
            missing_rate = numeric_df.isna().mean()
            
            # step 3 Remove targets with more than 20% missing data
            cols_to_keep = missing_rate[missing_rate < 0.2].index
            df_filtered = numeric_df[cols_to_keep]

            # step 4 log-transformation for filtered data
            df_filtered = df_filtered.select_dtypes(include=[np.number])
            # Replace non-positive values with NaN so log2 won't fail; they will be imputed
            df_filtered = df_filtered.mask(df_filtered <= 0)
            data_log = np.log2(df_filtered)
            
            # step 5 left-censored data and missing data imputation
            imputed_df = impute_left_censored(data_log)
            
            # step 6 join back injectionID and non-numeric columns
            imputed_df = pd.concat([injection_ids, imputed_df], axis=1)

            # VERIFY IMPUTED VALUES WITH R
            self.compare_imputed_results(injection_ids, data_log, imputed_df)
            import time
            time.sleep(2)
            # Update timestamp for the second output
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df_R_IMPUTED = self.extract_csv(R_IMPUTED)
            df_R_IMPUTED = df_R_IMPUTED.drop(columns=['Group'], errors='ignore').merge(injection_ids, on='SampleID', how='left')
            self.compare_imputed_results(injection_ids, data_log, df_R_IMPUTED)
            # COMPARE THE IMPUTED RESULTS MANUALLY BEFORE MOVING ON

            # step 7 join also sample_info
            sample_info = self.extract_csv(sampleId_file)
            merged_df = pd.concat([injection_ids, imputed_df], axis=1)
            
            merged_df = merged_df.merge(sample_info, on='injectionID', how='left')

            self.load_csv(f"merged_Imputed_output.csv", merged_df)

    def extract_csv(self, file_name):
        """Load a CSV file from the input folder."""
        full_path = os.path.join(self.input_path, file_name)
        df = pd.read_csv(full_path)
        return df
    
    def load_csv(self, file_name, df):
        """Save result dataframes as CSV files in the output folder."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        df.to_csv(os.path.join(self.output_path, file_name), index=False)

        # Copy the final output to the output_copy_path
        if not os.path.exists(self.output_copy_path):
            os.makedirs(self.output_copy_path, exist_ok=True)
        df.to_csv(os.path.join(self.output_copy_path, file_name), index=False)

    
    def compare_imputed_results(self, df_injection, df1, df2):
        """Compare two dataframes to detect changes in imputed results."""
        # CHECK THE IMPUTED VALUES
        df1 = pd.concat([df_injection, df1], axis=1)
        df2 = df2

        columns_list = df2.columns.tolist()

        check_df = detect_changes_between_dataframes(
            df1,
            df2,
            check_columns=columns_list,
            unique_key=['injectionID', 'SampleID', 'Group'],
            detect_column_changes=True
        )

        check_df = check_df[['injectionID', 'SampleID', 'Group', 'change_type', 'changes']]

        self.load_csv(f"imputation_check_{self.timestamp}.csv", check_df)


if __name__ == "__main__":
    # Create an instance of the Imputation class
    imputation_generator = Imputation()
    result = imputation_generator.impute_missing_values()