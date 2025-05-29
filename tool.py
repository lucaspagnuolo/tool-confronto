import streamlit as st
import pandas as pd
import numpy as np
import io

# Utility functions for AHP

def compute_ahp_weights(matrix: np.ndarray):
    geom_means = np.prod(matrix, axis=1) ** (1.0 / matrix.shape[1])
    weights = geom_means / geom_means.sum()
    return weights


def consistency_ratio(matrix: np.ndarray, weights: np.ndarray):
    n = matrix.shape[0]
    lamda_max = np.sum(np.dot(matrix, weights) / weights) / n
    ci = (lamda_max - n) / (n - 1)
    RI_dict = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = RI_dict.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0
    return ci, cr, lamda_max

# Streamlit App

def main():
    st.title("AHP Multi-Lot Evaluation App")

    # Sidebar: parameters
    st.sidebar.header("Parameters")
    num_lots = st.sidebar.number_input("Number of Lots", min_value=1, value=1, step=1)
    num_criteria = st.sidebar.number_input("Number of Criteria", min_value=1, value=3, step=1)
    num_competitors = st.sidebar.number_input("Number of Competitors", min_value=2, value=3, step=1)

    st.sidebar.markdown("---")
    lot_names = [st.sidebar.text_input(f"Name for Lot {i+1}", value=f"Lotto{i+1}") for i in range(num_lots)]
    criterion_names = [st.sidebar.text_input(f"Criterion {j+1}", value=f"Criterio{j+1}") for j in range(num_criteria)]
    competitor_names = [st.sidebar.text_input(f"Competitor {k+1}", value=f"Concorrente{k+1}") for k in range(num_competitors)]

    st.markdown("---")

    # Store results for download
    all_results = {}

    for idx, lot in enumerate(lot_names):
        st.header(f"Evaluation for {lot}")
        stim_crit_matrix = np.ones((num_criteria, num_criteria))
        for i in range(num_criteria):
            for j in range(i+1, num_criteria):
                key = f"lot{idx}_crit_{i}_{j}"
                val = st.slider(f"Comparison {criterion_names[i]} vs {criterion_names[j]}",
                                min_value=1.0, max_value=9.0, value=1.0, step=0.5,
                                key=key)
                stim_crit_matrix[i, j] = val
                stim_crit_matrix[j, i] = 1/val
        crit_weights = compute_ahp_weights(stim_crit_matrix)
        ci, cr, _ = consistency_ratio(stim_crit_matrix, crit_weights)

        st.subheader("Criterion Weights and Consistency")
        st.table(pd.DataFrame({'Weight': crit_weights}, index=criterion_names))
        st.write(f"CI: {ci:.4f}, CR: {cr:.4f}")
        if cr > 0.1:
            st.warning("CR > 0.1: revise judgments.")

        # Competitor comparisons per criterion
        lot_df = pd.DataFrame(index=competitor_names)
        for crit in criterion_names:
            st.subheader(f"Competitor Comparison for {crit}")
            comp_matrix = np.ones((num_competitors, num_competitors))
            for i in range(num_competitors):
                for j in range(i+1, num_competitors):
                    key = f"lot{idx}_{crit}_{i}_{j}"
                    val = st.slider(f"{competitor_names[i]} vs {competitor_names[j]}",
                                    min_value=1.0, max_value=9.0, value=1.0, step=0.5,
                                    key=key)
                    comp_matrix[i, j] = val
                    comp_matrix[j, i] = 1/val
            comp_weights = compute_ahp_weights(comp_matrix)
            ci_c, cr_c, _ = consistency_ratio(comp_matrix, comp_weights)
            lot_df[crit] = comp_weights
            st.write(pd.DataFrame({'Weight': comp_weights}, index=competitor_names))
            st.write(f"CI: {ci_c:.4f}, CR: {cr_c:.4f}")
            if cr_c > 0.1:
                st.warning(f"CR > 0.1 for {crit}: revise judgments.")

        # Aggregate and show
        st.subheader("Overall Scores")
        overall = lot_df.dot(pd.Series(crit_weights, index=criterion_names))
        overall_df = pd.DataFrame({'Overall Score': overall})
        st.table(overall_df.sort_values(by='Overall Score', ascending=False))

        # Save for download
        all_results[lot] = {
            'Criteria Weights': pd.DataFrame({'Weight': crit_weights}, index=criterion_names),
            'Competitor Scores': lot_df,
            'Overall Scores': overall_df
        }
        st.markdown("---")

    # Download section
    if all_results:
        bytes_io = io.BytesIO()
        # Create Excel writer and save sheets
        with pd.ExcelWriter(bytes_io, engine='xlsxwriter') as writer:
            for lot, dfs in all_results.items():
                for sheet_name, df in dfs.items():
                    df.to_excel(writer, sheet_name=f"{lot}_{sheet_name[:20]}")
            # No explicit save(), context manager handles it
        bytes_io.seek(0)

        st.download_button(
            label="Download Excel Report",
            data=bytes_io,
            file_name="AHP_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == '__main__':
    main()
