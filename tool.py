import streamlit as st
import pandas as pd
import numpy as np
import io

# Utility functions for AHP

def compute_ahp_weights(matrix: np.ndarray):
    geom_means = np.prod(matrix, axis=1) ** (1.0 / matrix.shape[1])
    weights = geom_means / geom_means.sum()
    return geom_means, weights


def consistency_ratio(matrix: np.ndarray, weights: np.ndarray):
    n = matrix.shape[0]
    lamda_max = np.sum(np.dot(matrix, weights) / weights) / n
    ci = (lamda_max - n) / (n - 1) if n > 1 else 0
    RI_dict = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = RI_dict.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0
    return lamda_max, ci, cr

# Streamlit App

def main():
    st.title("AHP Multi-Lot Evaluation App with Transparency")

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

    # Collect all results and transparency data
    report_data = {}

    for idx, lot in enumerate(lot_names):
        st.header(f"Evaluation for {lot}")

        # Criteria pairwise matrix
        crit_matrix = np.ones((num_criteria, num_criteria))
        for i in range(num_criteria):
            for j in range(i+1, num_criteria):
                val = st.slider(f"{criterion_names[i]} vs {criterion_names[j]}",
                                min_value=1.0, max_value=9.0, value=1.0, step=0.5,
                                key=f"crit_{idx}_{i}_{j}")
                crit_matrix[i, j] = val
                crit_matrix[j, i] = 1/val

        geom_means, crit_weights = compute_ahp_weights(crit_matrix)
        lamda_max, ci, cr = consistency_ratio(crit_matrix, crit_weights)

        # Display
        st.subheader("Criterion Weights & Consistency")
        st.write("**Comparison Matrix:**")
        st.dataframe(pd.DataFrame(crit_matrix, index=criterion_names, columns=criterion_names))
        st.write("**Geometric Means:**")
        st.dataframe(pd.DataFrame(geom_means, index=criterion_names, columns=["GeomMean"]))
        st.write("**Normalized Weights:**")
        st.dataframe(pd.DataFrame(crit_weights, index=criterion_names, columns=["Weight"]))
        st.write(f"\nλ_max: {lamda_max:.4f}, CI: {ci:.4f}, CR: {cr:.4f}")

        # Competitor matrices per criterion
        comp_results = {}
        st.subheader("Competitor Evaluations per Criterion")
        for crit in criterion_names:
            st.write(f"\n**Criterion: {crit}**")
            comp_matrix = np.ones((num_competitors, num_competitors))
            for i in range(num_competitors):
                for j in range(i+1, num_competitors):
                    key = f"comp_{idx}_{crit}_{i}_{j}"
                    val = st.slider(f"{competitor_names[i]} vs {competitor_names[j]}",
                                    min_value=1.0, max_value=9.0, value=1.0, step=0.5,
                                    key=key)
                    comp_matrix[i, j] = val
                    comp_matrix[j, i] = 1/val
            gm_comp, weights_comp = compute_ahp_weights(comp_matrix)
            lamda_c, ci_c, cr_c = consistency_ratio(comp_matrix, weights_comp)

            st.write("Comparison Matrix:")
            st.dataframe(pd.DataFrame(comp_matrix, index=competitor_names, columns=competitor_names))
            st.write("Geom Means & Weights:")
            df_comp = pd.DataFrame({'GeomMean': gm_comp, 'Weight': weights_comp}, index=competitor_names)
            st.dataframe(df_comp)
            st.write(f"λ_max: {lamda_c:.4f}, CI: {ci_c:.4f}, CR: {cr_c:.4f}")

            comp_results[crit] = {
                'matrix': pd.DataFrame(comp_matrix, index=competitor_names, columns=competitor_names),
                'geom_mean_weight': df_comp,
                'metrics': pd.Series({'lambda_max': lamda_c, 'CI': ci_c, 'CR': cr_c})
            }

        # Aggregate overall scores
        overall_scores = pd.Series({name: sum(comp_results[crit]['geom_mean_weight'].loc[name,'Weight'] * crit_weights[i]
                                               for i, crit in enumerate(criterion_names))
                                    for name in competitor_names}, name='Overall Score')
        st.subheader("Overall Scores")
        st.dataframe(overall_scores.sort_values(ascending=False))

        # Store all for report
        report_data[lot] = {
            'crit_matrix': pd.DataFrame(crit_matrix, index=criterion_names, columns=criterion_names),
            'crit_geom': pd.DataFrame(geom_means, index=criterion_names, columns=['GeomMean']),
            'crit_weight': pd.DataFrame(crit_weights, index=criterion_names, columns=['Weight']),
            'crit_metrics': pd.Series({'lambda_max': lamda_max, 'CI': ci, 'CR': cr}),
            'comp': comp_results,
            'overall': overall_scores
        }

    # Download report
    if report_data:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            for lot, data in report_data.items():
                # Criteria sheets
                data['crit_matrix'].to_excel(writer, sheet_name=f"{lot}_CritMatrix")
                data['crit_geom'].to_excel(writer, sheet_name=f"{lot}_CritGeom")
                data['crit_weight'].to_excel(writer, sheet_name=f"{lot}_CritWeight")
                data['crit_metrics'].to_frame(name='Value').to_excel(writer, sheet_name=f"{lot}_CritMetrics")
                # Competitor sheets per criterion
                for crit, cr_data in data['comp'].items():
                    safe = crit[:15]
                    cr_data['matrix'].to_excel(writer, sheet_name=f"{lot}_{safe}_Mat")
                    cr_data['geom_mean_weight'].to_excel(writer, sheet_name=f"{lot}_{safe}_GM_Wt")
                    cr_data['metrics'].to_frame(name='Value').to_excel(writer, sheet_name=f"{lot}_{safe}_Metrics")
                # Overall
                data['overall'].to_frame(name='Score').to_excel(writer, sheet_name=f"{lot}_Overall")
        buf.seek(0)
        st.download_button(
            label="Download Detailed Excel Report",
            data=buf,
            file_name="AHP_Detailed_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == '__main__':
    main()
