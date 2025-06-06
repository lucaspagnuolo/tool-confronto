import streamlit as st
import pandas as pd
import numpy as np
import io

# ======================================
# Funzioni di utilità AHP (invariate)
# ======================================

def compute_ahp_weights(matrix: np.ndarray):
    """
    Calcola i pesi AHP normalizzati a partire dalla matrice di confronto a coppie.
    Restituisce (geometric_means, normalized_weights).
    """
    geom_means = np.prod(matrix, axis=1) ** (1.0 / matrix.shape[1])
    weights = geom_means / geom_means.sum()
    return geom_means, weights

def consistency_ratio(matrix: np.ndarray, weights: np.ndarray):
    """
    Calcola λ_max, CI e CR per la matrice di confronto e il vettore di pesi.
    """
    n = matrix.shape[0]
    lamda_max = np.sum(np.dot(matrix, weights) / weights) / n
    ci = (lamda_max - n) / (n - 1) if n > 1 else 0.0
    # Valori RI tratti dalla macro (con qualche arrotondamento)
    RI_dict = {
        1: 0.0, 2: 0.0, 3: 0.489, 4: 0.805, 5: 1.059,
        6: 1.18, 7: 1.252, 8: 1.317, 9: 1.373, 10: 1.406,
        11: 1.421, 12: 1.45, 13: 1.464, 14: 1.482, 15: 1.497,
        16: 1.508, 17: 1.515, 18: 1.526, 19: 1.531
    }
    ri = RI_dict.get(n, 1.537)
    cr = ci / ri if ri != 0 else 0.0
    return lamda_max, ci, cr

# ======================================
# App Streamlit
# ======================================

def main():
    st.set_page_config(page_title="AHP Multi-Lot Evaluation", layout="wide")
    st.title("AHP Multi-Lot Evaluation App")

    # -----------------------------
    # Sidebar: parametri generali
    # -----------------------------
    st.sidebar.header("Parametri generali")
    num_lots = st.sidebar.number_input("Numero di lotti", min_value=1, value=1, step=1)
    num_criteria = st.sidebar.number_input("Numero di criteri", min_value=1, value=3, step=1)
    num_competitors = st.sidebar.number_input("Numero di concorrenti", min_value=2, value=3, step=1)

    st.sidebar.markdown("---")
    lot_names = [st.sidebar.text_input(f"Nome Lotto {i+1}", value=f"Lotto{i+1}") 
                 for i in range(num_lots)]
    criterion_names = [st.sidebar.text_input(f"Criterio {j+1}", value=f"Criterio{j+1}") 
                       for j in range(num_criteria)]
    competitor_names = [st.sidebar.text_input(f"Concorrente {k+1}", value=f"Conc{k+1}") 
                        for k in range(num_competitors)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Assegna PTmax ai criteri")
    ptmax_values = []
    for j, crit in enumerate(criterion_names):
        v = st.sidebar.number_input(f"PTmax per '{crit}'", min_value=0.0, value=1.0, step=0.5, key=f"ptmax_{j}")
        ptmax_values.append(v)

    # Normalizzo i PTmax per ottenere i pesi dei criteri
    total_pt = sum(ptmax_values) if sum(ptmax_values) > 0 else 1.0
    criterion_weights = np.array(ptmax_values) / total_pt

    st.markdown("___")
    st.subheader("Pesi dei criteri (normalizzati da PTmax)")
    df_crit_weights = pd.DataFrame({
        "Criterio": criterion_names,
        "PTmax": ptmax_values,
        "Peso normalizzato": criterion_weights
    }).set_index("Criterio")
    st.dataframe(df_crit_weights.style.format({"PTmax": "{:.2f}", "Peso normalizzato": "{:.4f}"}))

    st.markdown("___")

    # -------------------------------------
    # Ciclo sui lotti: per ogni lotto,
    # chiediamo i confronti AHP concorrenti
    # -------------------------------------
    report_data = {}

    for idx, lot in enumerate(lot_names):
        st.header(f"Valutazione Lotto: **{lot}**")

        # Per ciascun criterio, costruiamo una matrice AHP sui concorrenti
        comp_results = {}
        for i_crit, crit in enumerate(criterion_names):
            st.markdown(f"**Criterio {i_crit+1}: '{crit}'**")

            # Matrice di confronto a coppie (n_concorr × n_concorr) inizializzata a 1
            comp_matrix = np.ones((num_competitors, num_competitors))
            for i in range(num_competitors):
                for j in range(i+1, num_competitors):
                    slider_key = f"lot_{idx}_crit_{i_crit}_comp_{i}_{j}"
                    val = st.slider(
                        f"{competitor_names[i]} vs {competitor_names[j]}",
                        min_value=1.0, max_value=9.0, value=1.0, step=0.5,
                        key=slider_key
                    )
                    comp_matrix[i, j] = val
                    comp_matrix[j, i] = 1.0 / val

            # Calcolo pesi locali AHP e CR
            geom_comp, weights_comp = compute_ahp_weights(comp_matrix)
            lam_c, ci_c, cr_c = consistency_ratio(comp_matrix, weights_comp)

            # Visualizzo la matrice di confronto e i risultati
            with st.expander(f"Dettagli '{crit}'"):
                st.write("Matrice di confronto:")
                df_mat = pd.DataFrame(comp_matrix, 
                                      index=competitor_names, 
                                      columns=competitor_names)
                st.dataframe(df_mat.style.format("{:.3f}"))

                st.write("Geometric Means & Pesi locali:")
                df_gm = pd.DataFrame({
                    "GeomMean": geom_comp,
                    "Peso locale": weights_comp
                }, index=competitor_names)
                st.dataframe(df_gm.style.format({
                    "GeomMean": "{:.4f}",
                    "Peso locale": "{:.4f}"
                }))

                st.write(f"λ_max = **{lam_c:.4f}**  ·  CI = **{ci_c:.4f}**  ·  CR = **{cr_c:.4f}**")
                if cr_c > 0.1:
                    st.warning("**Attenzione:** CR > 0.10 → possibili incoerenze nei giudizi!")

            comp_results[crit] = {
                "matrix": pd.DataFrame(comp_matrix, index=competitor_names, columns=competitor_names),
                "geomemos": geom_comp,
                "weights": weights_comp,
                "metrics": {"lambda_max": lam_c, "CI": ci_c, "CR": cr_c}
            }

        # --------------------------------------------------
        # Una volta ottenuti i pesi locali per ogni criterio,
        # calcolo il punteggio complessivo di ciascun concorrente:
        #   OverallScore = Σ [ peso_locale(crit) * peso_criterio(normalizzato) * PTmax_totale ]
        # In pratica posso usare direttamente: Σ [ peso_locale(crit) * peso_criterio(normalizzato) ]
        # oppure moltiplicare i pesi locali per PTmax_norm. Qui:
        # PTmax_norm = PTmax_crit / Σ(PTmax)
        # --------------------------------------------------
        overall_scores = {}
        for c_name in competitor_names:
            score = 0.0
            for j, crit in enumerate(criterion_names):
                w_local = comp_results[crit]["weights"][competitor_names.index(c_name)]
                w_crit = criterion_weights[j]
                score += w_local * w_crit
            overall_scores[c_name] = score

        series_overall = pd.Series(overall_scores, name="Punteggio complessivo")
        series_overall = series_overall.sort_values(ascending=False)

        st.subheader("Punteggi complessivi dei concorrenti")
        st.dataframe(series_overall.to_frame().style.format("{:.4f}"))

        # ----------------------
        # Salvo i dati per il report
        # ----------------------
        report_data[lot] = {
            "criteri": {
                "names": criterion_names,
                "ptmax": ptmax_values,
                "peso_criteri": criterion_weights
            },
            "competitors": competitor_names,
            "comp_results": comp_results,
            "overall": series_overall
        }

        st.markdown("---")

    # =====================================
    # Se ci sono risultati, attivo il download Excel
    # =====================================
    if report_data:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # Per ogni lotto, creo un foglio per:
            #   - Peso criteri + PTmax
            #   - Per ciascun criterio: matrice di confronto e pesi locali
            #   - Punteggi complessivi
            for lot, data in report_data.items():
                # 1) Foglio summary criteri
                df_crit = pd.DataFrame({
                    "Criterio": data["criteri"]["names"],
                    "PTmax": data["criteri"]["ptmax"],
                    "Peso normalizzato": data["criteri"]["peso_criteri"]
                })
                df_crit.to_excel(writer, sheet_name=f"{lot}_Criteri", index=False)

                # 2) Per ogni criterio, matrice e pesi locali
                for crit in data["criteri"]["names"]:
                    safe_crit = crit[:20].replace(" ", "_")
                    # matrice di confronto
                    mat = data["comp_results"][crit]["matrix"]
                    mat.to_excel(writer, sheet_name=f"{lot}_{safe_crit}_Mat")
                    # pesi locali
                    df_local = pd.DataFrame({
                        "Concorrente": data["competitors"],
                        "GeomMean": data["comp_results"][crit]["geomemos"],
                        "Peso locale": data["comp_results"][crit]["weights"]
                    })
                    df_local.to_excel(writer, sheet_name=f"{lot}_{safe_crit}_Pesi", index=False)
                    # metriche
                    m = data["comp_results"][crit]["metrics"]
                    df_met = pd.DataFrame.from_dict(m, orient="index", columns=["Value"])
                    df_met.to_excel(writer, sheet_name=f"{lot}_{safe_crit}_Metriche")

                # 3) Punteggi complessivi
                df_over = data["overall"].to_frame().reset_index()
                df_over.columns = ["Concorrente", "Punteggio complessivo"]
                df_over.to_excel(writer, sheet_name=f"{lot}_Overall", index=False)

        buf.seek(0)
        st.download_button(
            label="Scarica report Excel dettagliato",
            data=buf,
            file_name="AHP_Detailed_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()
