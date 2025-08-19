import pandas as pd
import os
import sys

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import enforce_schema, detect_changes_between_dataframes
from Schema import SampleSchema, HighPHSchema, LowPHSchema

# Customized unwanted columns in input files
UNWANTEDCOLUMNS = ['Batch']
# Everytime running new preprocess, please load files to input directory and update the PROCESSFILES list
PROCESSFILES = ['sample.xlsx', 'highPH.xlsx', 'lowPH.xlsx']

class Cleaning:
    """This class includes methods for loading excel files, mergeing multiple files, droping duplicates and filtering."""

    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(__file__), 'input')
        self.output_path = os.path.join(os.path.dirname(__file__), 'output')
        self.output_copy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Preprocessing', 'input')
        self.dataframes = {}

    def preprocess(self):
        """Preprocess the data loaded, cleanup and merge."""
        for file_name in PROCESSFILES:
            self.dataframes[file_name] = self.extract_excel(file_name)

        df_sample = self.dataframes['sample.xlsx']
        df_highPH = self.dataframes['highPH.xlsx']
        df_lowPH = self.dataframes['lowPH.xlsx']

        # filter unwanted columns of highPH and lowPH
        df_highPH = df_highPH.drop(columns=UNWANTEDCOLUMNS, errors='ignore')
        df_lowPH = df_lowPH.drop(columns=UNWANTEDCOLUMNS, errors='ignore')
        
        # rename columns of highPH and lowPH to match sample
        df_highPH.columns = df_highPH.columns.str.slice(4)
        df_lowPH.columns = df_lowPH.columns.str.slice(4)
        
        # give the first column a name
        df_highPH.columns.values[0] = 'injectionID'
        df_lowPH.columns.values[0] = 'injectionID'

        # Normalize injectionID in both DataFrames before merging
        df_highPH['injectionID'] = df_highPH['injectionID'].astype(str).str.extract(r'_(\d+)_')[0].str.lstrip('0')
        df_lowPH['injectionID'] = df_lowPH['injectionID'].astype(str).str.extract(r'_(\d+)_')[0].str.lstrip('0')

        # Remove duplicate columns from highPH that are also in lowPH
        unique_columns = ['injectionID'] + list(set(df_highPH.columns) - set(df_lowPH.columns) - {'injectionID'})
        df_highPH = df_highPH[unique_columns]
        
        # Ensure schema for dataframes, question should this injectionID be str or int?
        df_sample = enforce_schema(df_sample, SampleSchema)
        df_highPH = enforce_schema(df_highPH, HighPHSchema)
        df_lowPH = enforce_schema(df_lowPH, LowPHSchema)
        
        # Join highPH and lowPH DataFrames and include all the columns based on injectionID
        df_test = df_lowPH.merge(df_highPH, on='injectionID', how='outer')
        
        # Compare the number of rows to decide the left DataFrame for the join
        if df_sample.shape[0] <= df_test.shape[0]:
            df_final = df_sample.merge(df_test, on='injectionID', how='left')
        else:
            df_final = df_test.merge(df_sample, on='injectionID', how='left')
        
        self.load_csv('merged_DataCleaning_output.csv', df_final)

        return df_final

    def extract_excel(self, file_name):
        """Load all Excel files in the input folder as separate DataFrames."""
        full_path = os.path.join(self.input_path, file_name)
        df = pd.read_excel(full_path)
 
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

    def test(self):
        """Test the class methods."""
        # Example test method
        df1 = self.extract_excel('sample.xlsx')
        df2 = self.extract_excel('sample.xlsx')

        # Add a new column with all values set to 'test' for testing purpose
        df2['NewColumn'] = 'test'
        # Change the value of a specific cell for testing purpose
        df2.loc[1, 'Group'] = 'TESTING'

        # This is the check columns list, and will be included in the result_df
        columns_list = df2.columns.tolist()

        # This is the demo of how to compare 2 dataframes, the result will contain check_columns and 2 new columns change_type, changes
        df = detect_changes_between_dataframes(
            df1,
            df2,
            check_columns=columns_list,
            unique_key='SampleID',
            detect_column_changes=True
        )
        # print out to verify the changes
        print(df.to_string(index=False))
    

if __name__ == "__main__":
    # Create an instance of the Cleaning class
    cleaning_generator = Cleaning()
    result = cleaning_generator.preprocess()
    print(result.shape)

    # This is a sample of how to detect 2 dataframes, could be adopted to certain use cases
    # cleaning_generator.test()
