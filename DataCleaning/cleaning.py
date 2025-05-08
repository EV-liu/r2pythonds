import pandas as pd
import os
import sys

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import enforce_schema
from Schema import SampleSchema, HighPHSchema, LowPHSchema

UNWANTEDCOLUMNS = ['Batch']

class Cleaing:
    """This class includes methods for loading excel files, mergeing multiple files, droping duplicates and filtering."""

    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(__file__), 'input')
        self.dataframes = {}

    def preprocess(self):
        """Preprocess the data loaded, cleanup and merge."""
        df_sample = self.load_excel('sample.xlsx')
        df_highPH = self.load_excel('highPH.xlsx')
        df_lowPH = self.load_excel('lowPH.xlsx')
        
        # filter unwanted columns of highPH and lowPH
        df_highPH = df_highPH.drop(columns=UNWANTEDCOLUMNS, errors='ignore')
        df_lowPH = df_lowPH.drop(columns=UNWANTEDCOLUMNS, errors='ignore')

        # rename columns of highPH and lowPH to match sample
        df_highPH.columns = df_highPH.columns.str.slice(4)
        df_lowPH.columns = df_lowPH.columns.str.slice(4)

        # give the first column a name
        df_highPH.columns.values[0] = 'injectionID'
        df_lowPH.columns.values[0] = 'injectionID'

        # Remove duplicate columns from highPH that are also in lowPH
        unique_columns = ['injectionID'] + list(set(df_highPH.columns) - set(df_lowPH.columns) - {'injectionID'})
        df_highPH = df_highPH[unique_columns]

        # Ensure schema for dataframes
        df_sample = enforce_schema(df_sample, SampleSchema)
        df_highPH = enforce_schema(df_highPH, HighPHSchema)
        df_lowPH = enforce_schema(df_lowPH, LowPHSchema)

        # Join highPH and lowPH DataFrames
        df_test = pd.concat([df_highPH, df_lowPH], axis=0, ignore_index=True)

        # Compare the number of rows to decide the left DataFrame for the join
        if df_sample.shape[0] <= df_test.shape[0]:
            df_final = df_sample.merge(df_test, on='injectionID', how='left')
        else:
            df_final = df_test.merge(df_sample, on='injectionID', how='left')

        return df_final

    def load_excel(self, file_name):
        """Load all Excel files in the input folder as separate DataFrames."""
        full_path = os.path.join(self.input_path, file_name)
        df = pd.read_excel(full_path)
 
        return df
    

if __name__ == "__main__":
    # Create an instance of the Cleaing class
    cleaning = Cleaing()
    result = cleaning.preprocess()
    print(result.shape)
