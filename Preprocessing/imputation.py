import pandas as pd
import numpy as np
import os
import sys
import types, numpy as _np

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import impute_left_censored

class Imputation:
    """This class includes methods for imputing missing values in the dataframes."""

    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(__file__), 'input')
        self.output_path = os.path.join(os.path.dirname(__file__), 'output')
        self.output_copy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Preprocessing', 'input')
        self.dataframes = {}

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
            injection_ids = df['injectionID']
            df = df.drop(columns=['injectionID'], errors='ignore')

            # step 2 check the missing rate
            missing_rate = df.isna().mean()

            # step 3 Remove targets with more than 20% missing data
            df_filtered = df.loc[:, missing_rate < 0.2]
            print("is numpy module:", isinstance(np, types.ModuleType) and np is _np)
            # step 4 log-transformation for filtered data
            data_log = np.log2(df_filtered)

            # step 5 left-censored data and missing data imputation
            imputed_df = impute_left_censored(data_log)

            # step 6 join back injectionID
            imputed_df['injectionID'] = injection_ids

            # step 7 join also sample_info
            sample_info = self.extract_csv(sampleId_file)
            merged_df = imputed_df.merge(sample_info, on='injectionID', how='left')

            print(merged_df)

    def extract_csv(self, file_name):
        """Load a CSV file from the input folder."""
        full_path = os.path.join(self.input_path, file_name)
        df = pd.read_csv(full_path)
        return df


if __name__ == "__main__":
    # Create an instance of the Imputation class
    imputation_generator = Imputation()
    result = imputation_generator.impute_missing_values()