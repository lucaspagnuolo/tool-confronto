import streamlit as st
import pandas as pd
import numpy as np

# Utility functions for AHP

def compute_ahp_weights(matrix: np.ndarray):
    # geometric mean method
    geom_means = np.prod(matrix, axis=1) ** (1.0 / matrix.shape[1])
    weights = geom_means / geom_means.sum()
    return weights


def consistency_ratio(matrix: np.ndarray, weights: np.ndarray):
    n = matrix.shape[0]
    # Consistency Index (CI)
    lamda_max = np.sum(np.dot(matrix, weights) / weights) / n
    ci = (lamda_max - n) / (n - 1)
    # Random Index (RI) values for n
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

    # For each lot, perform AHP
    for idx, lot in enumerate(lot_names):
        st.header(f"Evaluation for {lot}")
        # Dictionary to hold criterion weights
        crit_weights = {}
        # Let user input pairwise comparisons for criteria
        st.subheader("Pairwise Comparison: Criteria")
        crit_matrix = np.ones((num_criteria, num_criteria))
        for i in range(num_criteria):
            for j in range(i+1, num_criteria):
                val = st.slider(f"Comparison {criterion_names[i]} vs {criterion_names[j]}",
                                min_value=1.0, max_value=9.0, value=1.0, step=0.5,
                                key=f"lot{idx}_crit_{i}_{j}")
                crit_matrix[i, j] = val
                crit_matrix[j, i] = 1/val
        crit_weights_array = compute_ahp_weights(crit_matrix)
        ci, cr, lamda_max = consistency_ratio(crit_matrix, crit_weights_array)
        crit_weights = dict(zip(criterion_names, crit_weights_array))

        st.write("**Criterion Weights:**")
        st.table(pd.DataFrame.from_dict(crit_weights, orient='index', columns=["Weight"]))
        st.write(f"Consistency Index (CI): {ci:.4f}")
        st.write(f"Consistency Ratio (CR): {cr:.4f}")
        if cr > 0.1:
            st.warning("CR > 0.1: Consider revising judgments for consistency.")

        # Sub-evaluation per criterion: competitors
        lot_result = pd.DataFrame(index=competitor_names)
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
            lot_result[crit] = comp_weights
            st.write(pd.DataFrame({"Weight": comp_weights}, index=competitor_names))
            st.write(f"CI: {ci_c:.4f}, CR: {cr_c:.4f}")
            if cr_c > 0.1:
                st.warning(f"CR > 0.1 for criterion {crit}: revise your judgments.")

        # Aggregate results
        st.subheader("Overall Scores")
        overall = lot_result.dot(pd.Series(crit_weights))
        overall_df = pd.DataFrame({"Overall Score": overall})
        st.table(overall_df.sort_values(by="Overall Score", ascending=False))
        st.markdown("---")

if __name__ == '__main__':
    main()
