# DataProcessor/ProcessorClass.py
import os
import shutil
import pandas as pd


class ProcessorClass:
    """Base class for shared path setup and file I/O helpers."""

    def __init__(self, current_file):
        module_dir = os.path.dirname(os.path.abspath(current_file))
        self.input_path = os.path.join(module_dir, "input")
        self.output_path = os.path.join(module_dir, "output")

    def extract_excel(self, file_name, **kwargs):
        full_path = os.path.join(self.input_path, file_name)
        return pd.read_excel(full_path, **kwargs)

    def extract_csv(self, file_name, **kwargs):
        full_path = os.path.join(self.input_path, file_name)
        return pd.read_csv(full_path, **kwargs)
    
    def extract_txt(self, file_name, **kwargs):
        """
        Encoding should be provided in kwargs if the text file is not in UTF-8.
        e.g.:utf-8
            utf-8-sig
            cp1252
            latin1
        """
        full_path = os.path.join(self.input_path, file_name)
        return pd.read_csv(full_path, sep="\t", **kwargs)

    def load_csv(self, file_name, df, **kwargs):
        """For both comma-separated and semicolon-separated CSV files."""
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, file_name)
        df.to_csv(output_file, index=False, **kwargs)
        return output_file

    def load_txt(self, file_name, df, **kwargs):
        """For both tab-separated and space-separated text files."""
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, file_name)
        df.to_csv(output_file, index=False, sep="\t", encoding="utf-8-sig", **kwargs)
        return output_file
    
    def load_excel(self, file_name, df, **kwargs):
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, file_name)
        df.to_excel(output_file, index=False, **kwargs)
        return output_file

    def copy_output_file(self, file_name, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        source_file = os.path.join(self.output_path, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        shutil.copy2(source_file, destination_file)
        return destination_file