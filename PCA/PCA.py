import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import impute_left_censored, enforce_numeric_datatype, detect_changes_between_dataframes
from Schema.sample_schema import SampleSchema


class PCAPrcomp:
    """This class includes methods for Principal Components Analysis.
    Find orthogonal directions (principal components) that capture maximum variance in the data and project observations onto them."""

    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(__file__), 'input')
        self.output_path = os.path.join(os.path.dirname(__file__), 'output')
        self.output_copy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PCA', 'input')
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_copy_path, exist_ok=True)

    def pca_plot_variance(self, pca, scores, df):
        """Create a PCA plot."""
        # Variance explained
        variance_explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        variance_data = pd.DataFrame({
            'Principal_Component': np.arange(1, len(variance_explained) + 1),
            'Variance_Explained': variance_explained,
            'Cumulative_Variance': cumulative_variance,
        })

        # Plot proportion of variance explained by principal components
        plt.figure(figsize=(8, 6))
        plt.plot(variance_data['Principal_Component'], variance_data['Variance_Explained'], marker='o', linestyle='-')
        plt.title("Proportion of Variance Explained by Principal Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.grid(True)

        output_file_name = "variance_explained_plot.png"  # Default filename
        output_file_path = os.path.join(self.output_path, output_file_name)
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()
    
    def pca_plot_pc1_pc2(self, pca, scores, df):
        """Create a PCA plot."""
        # Variance explained
        variance_explained = pca.explained_variance_ratio_

        # Plot PC1 vs PC2
        scores_df = pd.DataFrame({
            'PC1': scores[:, 0],
            'PC2': scores[:, 1],
            'group': df['Group'],
            'sample': df['SampleID']
        })

        # Convert 'group' to a categorical variable
        scores_df['group'] = pd.Categorical(scores_df['group'])

        # Identify the points with the highest PC1 and PC2 values
        highest_PC1 = scores_df.loc[scores_df['PC1'].idxmax()]
        highest_PC2 = scores_df.loc[scores_df['PC2'].idxmin()]

        # Plot PC1 vs PC2 with improvements
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=scores_df, x='PC1', y='PC2', hue='group', style='group', s=100, edgecolor='black')

        # Add confidence ellipses for each group
        for group, data in scores_df.groupby('group'):
            cov = np.cov(data[['PC1', 'PC2']].T)
            mean = data[['PC1', 'PC2']].mean()
            ellipse = Ellipse(xy=mean, width=2*np.sqrt(cov[0, 0]), height=2*np.sqrt(cov[1, 1]),
                              edgecolor='black', fc='none')
            plt.gca().add_patch(ellipse)

        # Highlight highest PC1 and lowest PC2 points
        plt.scatter(highest_PC1['PC1'], highest_PC1['PC2'], color='red', label='Highest PC1', edgecolor='black', zorder=5)
        plt.scatter(highest_PC2['PC1'], highest_PC2['PC2'], color='blue', label='Lowest PC2', edgecolor='black', zorder=5)

        # Add axis labels with variance explained
        plt.title("PCA: PC1 vs PC2")
        plt.xlabel(f"Principal Component 1 ({variance_explained[0]*100:.2f}%)")
        plt.ylabel(f"Principal Component 2 ({variance_explained[1]*100:.2f}%)")

        # Add grid, legend, and theme adjustments
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.grid(True)
        plt.legend(title='Group', loc='center left', bbox_to_anchor=(1, 0.5))
        sns.set_style("white")
        
        output_file_name = "PC1_PC2_plot.png"  # Default filename
        output_file_path = os.path.join(self.output_path, output_file_name)
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        plt.show()
 
    def pca_prcomp(self, df):
        """Perform PCA using the prcomp method in python."""
        df = df.iloc[:, 3:]
        feature_cols = list(df.columns)
        df = enforce_numeric_datatype(df, (feature_cols))

        # Standardize (prcomp(scale.=TRUE))
        scaler = StandardScaler()
        Xz = scaler.fit_transform(df)

        # PCA (all components)
        pca = PCA()  # or PCA(n_components=0.95) to keep 95% variance
        scores = pca.fit_transform(Xz)  # like prcomp$x

        summary_df = self.pca_summary(pca)
        self.load_csv(f"PCA_summary_output.csv", summary_df)

        return pca, scores

    def pca_summary(self, pca):
        # Standard deviation, proportion, and cumulative proportion
        sdev = np.sqrt(pca.explained_variance_)
        prop = pca.explained_variance_ratio_
        cumprop = np.cumsum(prop)

        # Combine summary statistics and loadings
        summary = pd.DataFrame(
            {"StdDev": sdev, "Proportion": prop, "Cumulative": cumprop},
            index=[f"PC{i+1}" for i in range(pca.n_components_)]
        )

        return summary
    
    def extract_csv(self, file_name):
        """Load a CSV file from the input folder, case-insensitively if needed."""
        # Exact match first
        full_path = os.path.join(self.input_path, file_name)
        if os.path.exists(full_path):
            return pd.read_csv(full_path)
        # Case-insensitive lookup
        fname_lower = file_name.lower()
        for f in os.listdir(self.input_path):
            if f.lower() == fname_lower:
                return pd.read_csv(os.path.join(self.input_path, f))
        raise FileNotFoundError(f"Input file not found: {file_name} in {self.input_path}")
    
    def load_csv(self, file_name, df):
        """Save result dataframes as CSV files in the output folder."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        df.to_csv(os.path.join(self.output_path, file_name), index=False)

        # Copy the final output to the output_copy_path
        if not os.path.exists(self.output_copy_path):
            os.makedirs(self.output_copy_path, exist_ok=True)
        df.to_csv(os.path.join(self.output_copy_path, file_name), index=False)

    
if __name__ == "__main__":
    # Create an instance of the PCA class
    pca_generator = PCAPrcomp()
    # Step 0: Load imputed data
    df = pca_generator.extract_csv('merged_imputed_output.csv')
    # Step 1: Perform PCA and save result
    pca, scores = pca_generator.pca_prcomp(df)
    # Step 2: Plot PCA results
    pca_generator.pca_plot_variance(pca, scores, df)
    pca_generator.pca_plot_pc1_pc2(pca, scores, df)
