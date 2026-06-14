# Parallel Third step of this project: performs ANOVA on the imputed dataset 
# with the same logic as the R code, but with Python libraries. The ANOVA results are saved as CSV files in the output folder.

import datetime

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

from scipy.stats import shapiro
from scipy.stats import levene
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import mannwhitneyu

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)

from Helpers import enforce_numeric_datatype, detect_changes_between_dataframes

class ANOVAtests:
    """This class includes methods for ANOVA and Wilcoxon tests(for statistical analysis purposes)."""

    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(__file__), 'input')
        self.output_path = os.path.join(os.path.dirname(__file__), 'output')
        # self.output_copy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PCA', 'input')
        os.makedirs(self.output_path, exist_ok=True)
        # os.makedirs(self.output_copy_path, exist_ok=True)
        # Add a timestamp attribute for file naming
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def screen_normality(self, df, group_col, alpha=0.05):
        results = []

        feature_cols = [col for col in df.columns if col != group_col]

        for col in feature_cols:
            shapiro_result = self.Shapiro_Wilk_test(df, group_col=group_col, value_col=col, alpha=alpha)
            levene_result = self.Levene_test(df, group_col=group_col, value_col=col, alpha=alpha)

            results.append({
                "Target": col,
                "Shapiro_Statistic": shapiro_result["Statistic"],
                "Shapiro_P_Value": shapiro_result["P_Value"],
                "Shapiro_Pass": shapiro_result["Pass"],
                "Levene_Statistic": levene_result["Statistic"],
                "Levene_P_Value": levene_result["P_Value"],
                "Levene_Pass": levene_result["Pass"],
                "Use_ANOVA": shapiro_result["Pass"] and levene_result["Pass"]
            })

        return pd.DataFrame(results)

    def Shapiro_Wilk_test(self, df, group_col=None, value_col=None, alpha=0.05):
        """
        Run a Shapiro-Wilk normality test on one feature column.

        Note:
        - group_col is kept in the signature for consistency with the other test methods,
        but it is not used here because the R code tests the pooled target values.
        """
        values = pd.to_numeric(df[value_col], errors="coerce").dropna()

        if len(values) < 3:
            return {
                "Target": value_col,
                "Statistic": np.nan,
                "P_Value": np.nan,
                "Pass": False,
                "Reason": "Need at least 3 non-null values for Shapiro-Wilk"
            }

        stat, p_value = shapiro(values)

        return {
            "Target": value_col,
            "Statistic": float(stat),
            "P_Value": float(p_value),
            "Pass": bool(p_value > alpha),
            "Reason": None
        }

    def Levene_test(self, df, group_col, value_col, alpha=0.05):
        """Run Levene's test for equal variances across groups."""
        working_df = df[[group_col, value_col]].copy()
        working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
        working_df = working_df.dropna(subset=[group_col, value_col])

        grouped_values = [
            group[value_col].to_numpy()
            for _, group in working_df.groupby(group_col)
            if len(group[value_col]) > 0
        ]

        if len(grouped_values) < 2:
            return {
                "Target": value_col,
                "Statistic": np.nan,
                "P_Value": np.nan,
                "Pass": False,
                "Reason": "Need at least 2 groups with data for Levene's test"
            }
        
        if any(len(values) < 2 for values in grouped_values):
            return {
                "Target": value_col,
                "Statistic": np.nan,
                "P_Value": np.nan,
                "Pass": False,
                "Reason": "Each group needs at least 2 values for Levene's test"
            }

        stat, p_value = levene(*grouped_values)

        return {
            "Target": value_col,
            "Statistic": float(stat),
            "P_Value": float(p_value),
            "Pass": bool(p_value > alpha),
            "Reason": None
        }
    
    def normalize_comparison_pair(self, left, right):
        left = str(left).strip()
        right = str(right).strip()

        ordered = sorted([left, right])

        return ordered[0], ordered[1]

    def OneWay_ANOVA(self, df, group_col, value_col):
        """
        Run one-way ANOVA for one target, then Tukey HSD pairwise comparisons.

        Equivalent to the R block:
        aov(as.formula(paste(target, "~ disease")), data = data)
        TukeyHSD(anova_result)
        """
        working_df = df[[group_col, value_col]].copy()
        working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
        working_df = working_df.dropna(subset=[group_col, value_col])

        if working_df[group_col].nunique() < 2:
            return pd.DataFrame(columns=["Target", "Comparison", "P_Value"])

        formula = f'Q("{value_col}") ~ C(Q("{group_col}"))'
        model = ols(formula, data=working_df).fit()

        tukey = pairwise_tukeyhsd(
            endog=working_df[value_col],
            groups=working_df[group_col],
            alpha=0.05
        )

        tukey_df = pd.DataFrame(
            tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )

        tukey_df["Comparison"] = tukey_df.apply(
            lambda row: "-".join(
                self.normalize_comparison_pair(row["group1"], row["group2"])
            ),
            axis=1
        )

        return tukey_df[["Comparison", "p-adj"]].rename(
            columns={"p-adj": "P_Value"}
        ).assign(Target=value_col)[["Target", "Comparison", "P_Value"]]

    def run_ANOVA_tests(self, df, anova_targets, group_col):
        """
        Run ANOVA/Tukey for all approved targets and build long/wide p- and q-value outputs.

        Returns, in order:
        - ANOVA_p_long
        - ANOVA_q_long
        - ANOVA_p_wide
        - ANOVA_q_wide
        """
        anova_results = []

        for target in anova_targets:
            result_df = self.OneWay_ANOVA(
                df,
                group_col=group_col,
                value_col=target
            )

            if not result_df.empty:
                anova_results.append(result_df)

        if not anova_results:
            empty_long_p = pd.DataFrame(columns=["Target", "Comparison", "P_Value"])
            empty_long_q = pd.DataFrame(columns=["Target", "Comparison", "q_value"])
            empty_wide_p = pd.DataFrame(columns=["Comparison"])
            empty_wide_q = pd.DataFrame(columns=["Comparison"])
            return empty_long_p, empty_long_q, empty_wide_p, empty_wide_q

        ANOVA = pd.concat(anova_results, ignore_index=True)
        ANOVA["P_Value"] = pd.to_numeric(ANOVA["P_Value"], errors="coerce")
        ANOVA = ANOVA.dropna(subset=["P_Value"])

        if ANOVA.empty:
            empty_long_p = pd.DataFrame(columns=["Target", "Comparison", "P_Value"])
            empty_long_q = pd.DataFrame(columns=["Target", "Comparison", "q_value"])
            empty_wide_p = pd.DataFrame(columns=["Comparison"])
            empty_wide_q = pd.DataFrame(columns=["Comparison"])
            return empty_long_p, empty_long_q, empty_wide_p, empty_wide_q

        ANOVA["q_value"] = multipletests(ANOVA["P_Value"], method="fdr_bh")[1]

        ANOVA_p_long = ANOVA[["Target", "Comparison", "P_Value"]].copy()
        ANOVA_q_long = ANOVA[["Target", "Comparison", "q_value"]].copy()

        ANOVA_p_wide = (
            ANOVA_p_long
            .pivot(index="Comparison", columns="Target", values="P_Value")
            .reset_index()
        )

        ANOVA_q_wide = (
            ANOVA_q_long
            .pivot(index="Comparison", columns="Target", values="q_value")
            .reset_index()
        )

        ANOVA_p_wide.columns.name = None
        ANOVA_q_wide.columns.name = None

        return ANOVA_p_long, ANOVA_q_long, ANOVA_p_wide, ANOVA_q_wide

    def prep_ANOVA_data(self, df):
        """Prepare data for ANOVA by removing first two columns and keeping the group column."""
        analysis_df = pd.concat(
            [
                df.iloc[:, 2:],
                df[["Group"]]
            ],
            axis=1
        )

        analysis_df = analysis_df.loc[:, ~analysis_df.columns.duplicated()]

        return analysis_df
    
    def OneWay_Wilcoxon(self, df, group_col, value_col):
        """
        Run pairwise Wilcoxon rank-sum style comparisons for one target.

        Returns a long table with:
        - Target
        - Comparison
        - P_Value
        """
        working_df = df[[group_col, value_col]].copy()
        working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
        working_df = working_df.dropna(subset=[group_col, value_col])

        group_names = working_df[group_col].dropna().unique().tolist()

        if len(group_names) < 2:
            return pd.DataFrame(columns=["Target", "Comparison", "P_Value"])

        results = []

        for group1, group2 in combinations(group_names, 2):
            values1 = working_df.loc[working_df[group_col] == group1, value_col].to_numpy()
            values2 = working_df.loc[working_df[group_col] == group2, value_col].to_numpy()

            if len(values1) == 0 or len(values2) == 0:
                continue

            stat, p_value = mannwhitneyu(
                values1,
                values2,
                alternative="two-sided"
            )

            column_first, column_second = self.normalize_comparison_pair(group1, group2)

            results.append({
                "Target": value_col,
                "Comparison": f"{column_first}-{column_second}",
                "P_Value": float(p_value)
            })

        if not results:
            return pd.DataFrame(columns=["Target", "Comparison", "P_Value"])

        return pd.DataFrame(results)
    
    def run_Wilcoxon_tests(self, df, wilcoxon_targets, group_col):
        """
        Run pairwise Wilcoxon-style tests for all approved targets and build
        long/wide p- and q-value outputs.

        Returns, in order:
        - Wilcoxon_p_long
        - Wilcoxon_q_long
        - Wilcoxon_p_wide
        - Wilcoxon_q_wide
        """
        wilcoxon_results = []

        for target in wilcoxon_targets:
            result_df = self.OneWay_Wilcoxon(
                df,
                group_col=group_col,
                value_col=target
            )

            if not result_df.empty:
                wilcoxon_results.append(result_df)

        if not wilcoxon_results:
            empty_long_p = pd.DataFrame(columns=["Target", "Comparison", "P_Value"])
            empty_long_q = pd.DataFrame(columns=["Target", "Comparison", "q_value"])
            empty_wide_p = pd.DataFrame(columns=["Comparison"])
            empty_wide_q = pd.DataFrame(columns=["Comparison"])
            return empty_long_p, empty_long_q, empty_wide_p, empty_wide_q

        Wilcoxon = pd.concat(wilcoxon_results, ignore_index=True)
        Wilcoxon["P_Value"] = pd.to_numeric(Wilcoxon["P_Value"], errors="coerce")
        Wilcoxon = Wilcoxon.dropna(subset=["P_Value"])

        if Wilcoxon.empty:
            empty_long_p = pd.DataFrame(columns=["Target", "Comparison", "P_Value"])
            empty_long_q = pd.DataFrame(columns=["Target", "Comparison", "q_value"])
            empty_wide_p = pd.DataFrame(columns=["Comparison"])
            empty_wide_q = pd.DataFrame(columns=["Comparison"])
            return empty_long_p, empty_long_q, empty_wide_p, empty_wide_q

        Wilcoxon["q_value"] = multipletests(Wilcoxon["P_Value"], method="fdr_bh")[1]

        Wilcoxon_p_long = Wilcoxon[["Target", "Comparison", "P_Value"]].copy()
        Wilcoxon_q_long = Wilcoxon[["Target", "Comparison", "q_value"]].copy()

        Wilcoxon_p_wide = (
            Wilcoxon_p_long
            .pivot(index="Comparison", columns="Target", values="P_Value")
            .reset_index()
        )

        Wilcoxon_q_wide = (
            Wilcoxon_q_long
            .pivot(index="Comparison", columns="Target", values="q_value")
            .reset_index()
        )

        Wilcoxon_p_wide.columns.name = None
        Wilcoxon_q_wide.columns.name = None

        return Wilcoxon_p_long, Wilcoxon_q_long, Wilcoxon_p_wide, Wilcoxon_q_wide

    def merge_test_results(
        self,
        ANOVA_p_long,
        ANOVA_q_long,
        Wilcoxon_p_long,
        Wilcoxon_q_long,
    ):
        """ Merge ANOVA and Wilcoxon outputs into final combined p/q tables."""
        q_for_all_long = pd.concat(
            [ANOVA_q_long, Wilcoxon_q_long],
            ignore_index=True,
            sort=False
        )

        p_for_all_long = pd.concat(
            [ANOVA_p_long, Wilcoxon_p_long],
            ignore_index=True,
            sort=False
        )

        combined_long = pd.merge(
            p_for_all_long,
            q_for_all_long,
            on=["Target", "Comparison"],
            how="outer"
        )

        return combined_long, q_for_all_long

    def filter_significant_results(self, q_for_all_long, alpha=0.05):
        """
        Filter significant q-values and build long/wide significance outputs.

        Returns: filtered_sig
        """
        filtered_sig = q_for_all_long[q_for_all_long["q_value"] < alpha].copy()

        if filtered_sig.empty:
            empty_long = pd.DataFrame(columns=q_for_all_long.columns.tolist() + ["group"])
            return empty_long

        filtered_sig["Target"] = (
            filtered_sig["Target"]
            .astype(str)
            .str.replace(r"^A_", "", regex=True)
        )

        filtered_sig["group"] = (
            filtered_sig["Target"]
            .str.replace(r"\d.*$", "", regex=True)
            .str.replace(r"_$", "", regex=True)
        )

        return filtered_sig

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

        # # Copy the final output to the output_copy_path
        # if not os.path.exists(self.output_copy_path):
        #     os.makedirs(self.output_copy_path, exist_ok=True)
        # df.to_csv(os.path.join(self.output_copy_path, file_name), index=False)

    def compare_anova_results(self, type, source, result):
        """Compare two dataframes to detect changes in anova results."""
        def split_comparison_groups(df):
            """
            Split Comparison like '2-1' into ordered Group1='1', Group2='2'.
            """
            result = df.copy()

            groups = result["Comparison"].astype(str).str.split("-", n=1, expand=True)
            result["Group1_raw"] = groups[0].str.strip()
            result["Group2_raw"] = groups[1].str.strip()

            def order_pair(row):
                left = row["Group1_raw"]
                right = row["Group2_raw"]

                # Prefer numeric ordering when possible; otherwise fall back to string ordering.
                try:
                    left_num = float(left)
                    right_num = float(right)
                    ordered = sorted([(left_num, left), (right_num, right)], key=lambda item: item[0])
                    return pd.Series([ordered[0][1], ordered[1][1]])
                except ValueError:
                    ordered = sorted([left, right])
                    return pd.Series(ordered)

            result[["Group1", "Group2"]] = result.apply(order_pair, axis=1)

            result = result.drop(columns=["Group1_raw", "Group2_raw"])

            # Optional: normalize Comparison too, so 2-1 and 1-2 become the same key.
            result["Comparison"] = result["Group1"].astype(str) + "-" + result["Group2"].astype(str)

            return result

        df1 = split_comparison_groups(source)
        df1["Target"] = (
            df1["Target"]
            .astype(str)
            .str.strip()
            .str.replace(r"^A_", "", regex=True)
        )
        df2 = split_comparison_groups(result)

        check_df = detect_changes_between_dataframes(
            df1,
            df2,
            check_columns=["P_Value", "q_value"],
            unique_key=["Target", "Group1", "Group2"],
            detect_column_changes=True
        )

        check_df = check_df[["Target", "Group1", "Group2", "change_type", "changes"]]

        self.load_csv(f"{type}_check_{self.timestamp}.csv", check_df)


if __name__ == "__main__":
    # Create an instance of the ANOVA class
    anova_generator = ANOVAtests()

    # Step 0: Load imputed data
    raw_df = anova_generator.extract_csv("merged_imputed_output.csv")
    analysis_df = anova_generator.prep_ANOVA_data(raw_df)

    # Step 1: Perform Normality / screening tests and save result
    normality_results = anova_generator.screen_normality(
        analysis_df,
        group_col="Group",
        alpha=0.05
    )

    # Step 2: Branch to ANOVA or Wilcoxon
    anova_targets = normality_results[normality_results["Use_ANOVA"]==True]["Target"].tolist()
    ANOVA_p_long, ANOVA_q_long, ANOVA_p_wide, ANOVA_q_wide = (
        anova_generator.run_ANOVA_tests(
            analysis_df,
            anova_targets,
            group_col="Group"
        )
    )
    anova_results = ANOVA_p_long.merge(ANOVA_q_long, on=["Target", "Comparison"], how="outer")
    
    wilcoxon_targets = normality_results[normality_results["Use_ANOVA"]==False]["Target"].tolist()  

    Wilcoxon_p_long, Wilcoxon_q_long, Wilcoxon_p_wide, Wilcoxon_q_wide = (
        anova_generator.run_Wilcoxon_tests(
            analysis_df,
            wilcoxon_targets,
            group_col="Group"
        )
    )
    wilcoxon_results = Wilcoxon_p_long.merge(Wilcoxon_q_long, on=["Target", "Comparison"], how="outer")
    # test block for comparing the results with the previous run in R studio
    # r_anova_results = anova_generator.extract_csv("ANOVA_results.csv")
    # r_wilcoxon_results = anova_generator.extract_csv("Wilcoxon_results.csv")
    # anova_generator.compare_anova_results("ANOVA", r_anova_results, anova_results)
    # anova_generator.compare_anova_results("Wilcoxon", r_wilcoxon_results, wilcoxon_results)

    # Step 3: Merge ANOVA and Wilcoxon outputs
    combined_all_long, q_for_all_long = (
    anova_generator.merge_test_results(
            ANOVA_p_long,
            ANOVA_q_long,
            Wilcoxon_p_long,
            Wilcoxon_q_long,
        )
    )

    anova_generator.load_csv("combined_all_long.csv", combined_all_long)

    filtered_sig = anova_generator.filter_significant_results(
        q_for_all_long,
        alpha=0.05
    )

    anova_generator.load_csv("significant_targets_long.csv", filtered_sig)