import os
import sys
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# This tells Python where the root directory of your project is
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)


def Migration_line_plots(df_long: pd.DataFrame, output_dir: str = os.path.join(os.path.dirname(__file__), 'output')) -> pd.DataFrame:
    df = df_long.copy()

    # Step 1: create the plotting group, same idea as case_when in R
    conditions = [
        (df["cell"] == "HCC1143") & (df["group"] == "Control"),
        (df["cell"] == "HCC1143") & (df["group"] == "Migration"),
        (df["cell"] == "HCC38") & (df["group"] == "Control"),
        (df["cell"] == "HCC38") & (df["group"] == "Migration"),
        (df["cell"] == "media") | (df["group"] == "media"),
    ]
    labels = [
        "1143control",
        "1143migration",
        "38control",
        "38migration",
        "media",
    ]

    df["line_group"] = pd.Series(pd.NA, index=df.index)
    for condition, label in zip(conditions, labels):
        df.loc[condition, "line_group"] = label

    # Step 2: force time ordering
    time_order = [0, 12, 18, 22]
    df["time"] = pd.Categorical(df["time"], categories=time_order, ordered=True)

    # Step 3: summarize for plotting
    plot_df = (
        df.groupby(["compound", "time", "line_group"], dropna=False)["value"]
        .agg(
            mean_value="mean",
            sd_value=lambda x: x.std(ddof=1),
            se_value=lambda x: x.std(ddof=1) / math.sqrt(x.notna().sum()) if x.notna().sum() > 0 else float("nan"),
        )
        .reset_index()
    )

    # Step 4: convert time back to numeric for plotting
    plot_df["time"] = plot_df["time"].astype(float)

    # Step 5: make one plot per compound
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Intotal compounds to plot: {plot_df['compound'].nunique()}")
    for compound, df_sub in plot_df.groupby("compound"):
        print(f"Start plotting compound: {compound}, data shape: {df_sub.shape}, save to: {output_path / f'all_{compound}.png'} ")
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for line_group, group_data in df_sub.groupby("line_group"):
            if pd.isna(line_group):
                continue

            group_data = group_data.sort_values("time")

            ax.plot(
                group_data["time"],
                group_data["mean_value"],
                marker="o",
                linewidth=1.0,
                label=line_group,
            )

        ax.set_title(compound)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean value")
        ax.set_xticks([0, 12, 18, 22])
        ax.set_xlim(0, 22)
        ax.legend(
            title="Condition",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
        )        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.subplots_adjust(right=0.78)

        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(compound))
        fig.savefig(output_path / f"all_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return plot_df

def Migration_box_plots(df_long: pd.DataFrame, df_pvalue: pd.DataFrame, output_dir: str = os.path.join(os.path.dirname(__file__), 'output')) -> pd.DataFrame:
    df_all = df_long.copy()
    df_all = df_all[df_all["time"].isin([0, 22])].copy()

    df_cells = df_all[df_all["cell"].isin(["HCC1143", "HCC38"])].copy()

    # Force plotting order
    time_order = [0, 22]
    group_order = ["Control", "Migration"]
    x_order = [
        (0, "Control"),
        (0, "Migration"),
        (22, "Control"),
        (22, "Migration"),
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Intotal compounds to plot: {df_cells['compound'].nunique()}")

    for (compound, cell), df_sub in df_cells.groupby(["compound", "cell"]):
        print(f"Start plotting compound: {compound}, cell: {cell}, data shape: {df_sub.shape}, save to: {output_path / f'box_{compound}_{cell}.png'} ")
        fig, ax = plt.subplots(figsize=(8, 5))

        box_data = []
        x_labels = []

        for time_value, group_value in x_order:
            values = df_sub.loc[
                (df_sub["time"] == time_value) &
                (df_sub["group"] == group_value),
                "value"
            ].dropna()

            box_data.append(values)
            x_labels.append(f"{time_value}-{group_value}")

        ax.boxplot(box_data, labels=x_labels, patch_artist=True)
        for i, values in enumerate(box_data, start=1):
            if len(values) == 0:
                continue

            x_jitter = pd.Series(range(len(values)), dtype=float)
            x_jitter = (x_jitter - x_jitter.mean()) * 0.03
            x_positions = i + x_jitter

            ax.scatter(
                x_positions,
                values,
                color="black",
                alpha=0.7,
                s=18,
                zorder=3,
            )

        # Add Control-vs-Migration significance marks for each time point.
        pvalue_subset = df_pvalue.copy()
        pvalue_subset["time"] = pd.to_numeric(
            pvalue_subset["time"],
            errors="coerce",
        )

        plotted_values = pd.concat(box_data, ignore_index=True)
        value_range = plotted_values.max() - plotted_values.min()

        # Keep a visible gap even when all values are identical.
        if pd.isna(value_range) or value_range == 0:
            value_range = 1

        annotation_positions = {
            0: (1, 2),  # 0-Control and 0-Migration
            22: (3, 4),  # 22-Control and 22-Migration
        }

        for time_value, (control_position, migration_position) in annotation_positions.items():
            matching_pvalue = pvalue_subset.loc[
                (pvalue_subset["lipid"] == compound) &
                (pvalue_subset["cellline"] == cell) &
                (pvalue_subset["time"] == time_value),
                "mark",
            ]

            if matching_pvalue.empty:
                continue

            mark = matching_pvalue.iloc[0]

            pair_values = pd.concat(
                [
                    box_data[control_position - 1],
                    box_data[migration_position - 1],
                ],
                ignore_index=True,
            )

            if pair_values.empty:
                continue

            bar_height = pair_values.max() + (0.08 * value_range)
            bar_tip = bar_height + (0.02 * value_range)
            text_height = bar_tip + (0.02 * value_range)

            ax.plot(
                [
                    control_position,
                    control_position,
                    migration_position,
                    migration_position,
                ],
                [bar_height, bar_tip, bar_tip, bar_height],
                color="black",
                linewidth=1,
                zorder=4,
            )

            ax.text(
                (control_position + migration_position) / 2,
                text_height,
                mark,
                ha="center",
                va="bottom",
                fontsize=12,
                zorder=5,
            )


        ax.set_title(f"{compound} | {cell}")
        ax.set_xlabel("Time-Group")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        safe_compound = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(compound))
        safe_cell = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(cell))
        fig.savefig(output_path / f"box_{safe_compound}_{safe_cell}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return df_all

def pValue_cleanup(df_pvalue: pd.DataFrame) -> pd.DataFrame:
    df = df_pvalue.copy()
    df = df[["contrast", "cellline", "time", "lipid", "p.value"]]
    df["mark"] = df["p.value"].apply(lambda x: "***" if x < 0.001 else ("**" if x >= 0.001 and x < 0.01 else ("*" if x >= 0.01 and x < 0.05 else "")))
    df = df[df["mark"] != ""]
    return df

def estimation_cleanup(df_estimation: pd.DataFrame) -> pd.DataFrame:
    df = df_estimation.copy()
    df = df[["cellline", "time", "lipid", "estimate"]]
    return df

def Migration_heatmap_plots(
        df_estimation: pd.DataFrame,
        output_dir: str = os.path.join(os.path.dirname(__file__), "output"),
    ) -> pd.DataFrame:
    df = df_estimation.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    time_order = [0, 12, 18, 22]
    lipid_order = sorted(df["lipid"].dropna().unique())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    muted_cmap = LinearSegmentedColormap.from_list(
        "muted_blue_red",
        ["#6F8FAF", "#F4F1EC", "#B97878"],
    )

    for cellline, df_cell in df.groupby("cellline"):
        df_pivot = df_cell.pivot_table(
            index="lipid",
            columns="time",
            values="estimate",
            aggfunc="mean",
        )

        # Ensure every heatmap uses the same row and column positions.
        df_pivot = df_pivot.reindex(
            index=lipid_order,
            columns=time_order,
        )

        # Normalize each lipid's four time values independently to 0-1.
        row_min = df_pivot.min(axis=1)
        row_range = df_pivot.max(axis=1) - row_min

        # Avoid division by zero for lipids with four identical estimates.
        row_range = row_range.replace(0, 1)

        df_color = df_pivot.sub(row_min, axis=0).div(row_range, axis=0)

        figure_height = max(6, len(lipid_order) * 0.25)
        fig, ax = plt.subplots(figsize=(8, figure_height))

        heatmap = ax.imshow(
            df_color.to_numpy(),
            aspect="auto",
            cmap=muted_cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        fig.colorbar(heatmap, ax=ax, label="Relative estimate within lipid (low to high)")

        ax.set_xticks(range(len(time_order)))
        ax.set_xticklabels(time_order)

        # Move the time labels and "Time" axis label above the heatmap.
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Time")

        ax.set_yticks(range(len(lipid_order)))
        ax.set_yticklabels(lipid_order)
        ax.set_ylabel("Lipid")

        ax.set_title(
            f"Estimates Heatmap | {cellline}",
            y=-0.16,
        )

        safe_cell = "".join(
            character if character.isalnum() or character in "_-"
            else "_"
            for character in str(cellline)
        )

        fig.tight_layout()
        fig.savefig(
            output_path / f"heatmap_{safe_cell}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Create one combined heatmap: four times for each cell line per lipid.
    cell_order = ["HCC1143", "HCC38"]

    df_combined = df.pivot_table(
        index="lipid",
        columns=["cellline", "time"],
        values="estimate",
        aggfunc="mean",
    )

    combined_columns = pd.MultiIndex.from_product(
        [cell_order, time_order],
        names=["cellline", "time"],
    )

    # Same lipid order, with 8 cells per lipid row.
    df_combined = df_combined.reindex(
        index=lipid_order,
        columns=combined_columns,
    )

    # Normalize across all 8 values in each lipid row:
    # 4 time points for HCC1143 plus 4 time points for HCC38.
    row_min = df_combined.min(axis=1)
    row_range = df_combined.max(axis=1) - row_min
    row_range = row_range.replace(0, 1)

    df_combined_color = (
        df_combined
        .sub(row_min, axis=0)
        .div(row_range, axis=0)
    )

    figure_height = max(6, len(lipid_order) * 0.25)
    fig, ax = plt.subplots(figsize=(12, figure_height))

    heatmap = ax.imshow(
        df_combined_color.to_numpy(),
        aspect="auto",
        cmap=muted_cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Relative estimate within lipid (low to high)",
    )

    # Label each of the eight colored cells by cell line and time.
    x_labels = [
        f"{cellline}\n{time_value}"
        for cellline, time_value in combined_columns
    ]

    ax.set_xticks(range(len(combined_columns)))
    ax.set_xticklabels(x_labels)

    # Move the Cell line + time labels above the heatmap.
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Cell line and time")

    ax.set_yticks(range(len(lipid_order)))
    ax.set_yticklabels(lipid_order)
    ax.set_ylabel("Lipid")

    # Visually separate the two groups of four time points.
    ax.axvline(3.5, color="black", linewidth=1)

    ax.set_title(
        "Estimates Heatmap | HCC1143 and HCC38",
        y=-0.16,
    )

    fig.tight_layout()
    fig.savefig(
        output_path / "heatmap_HCC1143_HCC38.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return df

if __name__ == "__main__":
    # Read the Migration project df_final.csv file, should come up with a more project relevant naming
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input', "df_final.csv"))
    print("df shape:", df.shape)
    # Shape the final df to df_long format
    df_wide = df.drop(df.columns[[0, 1, 5, 6]], axis=1)
    df_long = df_wide.melt(
        id_vars=["cell", "group", "time"],
        var_name="compound",
        value_name="value"
    )
    print("df_long shape:", df_long.shape)
    # Plot line plots for each compound and save them to the output directory
    plot_df = Migration_line_plots(df_long)
    df_calculated = pd.read_csv(os.path.join(os.path.dirname(__file__), 'input', "condition_contrasts.csv"))
    df_pvalue = pValue_cleanup(df_calculated)
    # Plot box plots for each compound and cell line and save them to the output directory
    box_df = Migration_box_plots(df_long, df_pvalue)
    df_estimation = estimation_cleanup(df_calculated)
    # Plot heatmaps for each cell line and save them to the output directory
    heatmap_df = Migration_heatmap_plots(df_estimation)