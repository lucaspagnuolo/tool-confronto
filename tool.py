import streamlit as st
import pandas as pd
import numpy as np
import io
import xlsxwriter

# ======================================
# Funzioni di utilità AHP
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
    RI_dict = {
        1: 0.0,   2: 0.0,   3: 0.489,  4: 0.805,  5: 1.059,
        6: 1.18,  7: 1.252, 8: 1.317, 9: 1.373, 10: 1.406,
        11: 1.419,12: 1.445,13: 1.46, 14: 1.471,15: 1.485
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

        comp_results = {}
        for i_crit, crit in enumerate(criterion_names):
            st.markdown(f"**Criterio {i_crit+1}: '{crit}'**")

            # Matrice di confronto a coppie inizializzata a 1
            comp_matrix = np.ones((num_competitors, num_competitors))
            for i in range(num_competitors):
                for j in range(i+1, num_competitors):
                    slider_key = f"lot_{idx}_crit_{i_crit}_comp_{i}_{j}"
                    # Slider centrato da –8 a +8, default=0
                    v = st.slider(
                        f"{competitor_names[i]} vs {competitor_names[j]}",
                        min_value=-8, max_value=8, value=0, step=1,
                        key=slider_key
                    )
                    if v > 0:
                        val = 1.0 + v
                        desc = f"Preferisco **{competitor_names[i]}** di **{val:.0f}×**"
                    elif v < 0:
                        val = 1.0 / (1.0 + abs(v))
                        desc = f"Preferisco **{competitor_names[j]}** di **{1+abs(v):.0f}×**"
                    else:
                        val = 1.0
                        desc = "Perfetto equilibrio"
                    comp_matrix[i, j] = val
                    comp_matrix[j, i] = 1.0 / val
                    st.write(desc)

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
        # Calcolo punteggi complessivi
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
    # Download Excel “full-feature”
    # =====================================
    if report_data:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            workbook: "xlsxwriter.Workbook" = writer.book

            # Formati
            fmt_header = workbook.add_format({
                "bold": True, "font_color": "#FFFFFF", "bg_color": "#4BACC6",
                "align": "center", "valign": "vcenter", "border": 1
            })
            fmt_title = workbook.add_format({
                "bold": True, "font_size": 14, "font_color": "#FFFFFF",
                "bg_color": "#2F75B5", "align": "left", "valign": "vcenter"
            })
            fmt_label = workbook.add_format({
                "bold": True, "font_color": "#444444", "bg_color": "#E7E6E6",
                "align": "left", "valign": "vcenter", "border": 1
            })
            fmt_input = workbook.add_format({
                "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter",
                "locked": False
            })
            fmt_locked = workbook.add_format({
                "border": 1, "align": "center", "valign": "vcenter", "locked": True
            })
            fmt_bordo = workbook.add_format({"border": 1})
            fmt_bordo_center = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})

            # Foglio nascosto RI
            ri_sheet = workbook.add_worksheet("RI_Lookup")
            ri_sheet.hide()
            ri_vals = {
                1: 0.0,   2: 0.0,   3: 0.489,  4: 0.805,  5: 1.059,
                6: 1.18,  7: 1.252, 8: 1.317, 9: 1.373, 10: 1.406,
                11: 1.419,12: 1.445,13: 1.46, 14: 1.471,15: 1.485
            }
            ri_sheet.write(0, 0, "n")
            ri_sheet.write(0, 1, "RI")
            riga = 1
            for n_val, ri_val in ri_vals.items():
                ri_sheet.write(riga, 0, n_val)
                ri_sheet.write(riga, 1, ri_val)
                riga += 1

            # Fogli per lotto
            for lot, data in report_data.items():
                worksheet = workbook.add_worksheet(f"{lot}")
                worksheet.protect(options={
                    "insert_rows": False, "insert_columns": False,
                    "format_cells": False, "format_columns": False,
                    "format_rows": False, "delete_rows": False,
                    "delete_columns": False, "sort": False, "autofilter": False
                })

                worksheet.set_column("A:A", 20)
                worksheet.set_column("B:C", 15)
                worksheet.set_column("D:Z", 12)

                # Parametri in alto
                worksheet.merge_range("A2:C2", "Titolo Iniziativa:", fmt_title)
                worksheet.write("D2", lot, fmt_locked)
                worksheet.write("A3", "Num Criteri:", fmt_label)
                worksheet.write("B3", len(data["criteri"]["names"]), fmt_locked)
                worksheet.write("A4", "Num Concorrenti:", fmt_label)
                worksheet.write("B4", len(data["competitors"]), fmt_locked)
                worksheet.write("A6", "1) Inserisci ID Criterio e PTmax nelle corrispondenti celle.", fmt_bordo)
                worksheet.write("A7", "2) Inserisci la valutazione di ogni coppia di offerte nelle celle gialle.", fmt_bordo)

                # ID Criteri e PTmax
                crit_names = data["criteri"]["names"]
                ptmax_list = data["criteri"]["ptmax"]
                worksheet.write("A8", "ID Criterio", fmt_header)
                worksheet.write("B8", "PTmax", fmt_header)
                for j, crit in enumerate(crit_names):
                    worksheet.write(9 + j, 0, crit, fmt_bordo)
                    worksheet.write(9 + j, 1, ptmax_list[j], fmt_input)

                # Blocchi AHP per criterio
                num_c = len(crit_names)
                num_k = len(data["competitors"])
                competitors = data["competitors"]
                blocco_start = 9 + num_c + 2

                for i_crit, crit in enumerate(crit_names):
                    r0 = blocco_start + i_crit * (num_k + 8)

                    worksheet.merge_range(r0, 0, r0, 1, f"Criterio: {crit}", fmt_title)
                    worksheet.write(r0 + 1, 0, "Concorrente", fmt_header)
                    for j in range(num_k):
                        worksheet.write(r0 + 1, 1 + j, competitors[j], fmt_header)
                    for i in range(num_k):
                        worksheet.write(r0 + 1 + i + 1, 0, competitors[i], fmt_header)

                    matrix = data["comp_results"][crit]["matrix"].values
                    for i in range(num_k):
                        for j in range(num_k):
                            if j > i:
                                cell_fmt = fmt_input
                            else:
                                cell_fmt = fmt_locked
                            val = matrix[i, j]
                            worksheet.write(r0 + 1 + i + 1, 1 + j, val, cell_fmt)

                    # GeomMean & PesoLocale
                    col_geom = 1 + num_k + 1
                    col_wloc = col_geom + 1
                    worksheet.write(r0 + 1, col_geom, "GeomMean", fmt_header)
                    worksheet.write(r0 + 1, col_wloc, "PesoLocale", fmt_header)
                    geom_start = xlsxwriter.utility.xl_rowcol_to_cell(r0 + 2, col_geom)
                    geom_end   = xlsxwriter.utility.xl_rowcol_to_cell(r0 + 1 + num_k, col_geom)
                    for i in range(num_k):
                        riga = r0 + 1 + i + 1
                        formula_g = f"=GEOMEAN({xlsxwriter.utility.xl_range(riga,1,riga,1+num_k-1)})"
                        worksheet.write_formula(riga, col_geom, formula_g, fmt_locked)
                        formula_w = f"=IFERROR({xlsxwriter.utility.xl_rowcol_to_cell(riga,col_geom)}/SUM({geom_start}:{geom_end}),0)"
                        worksheet.write_formula(riga, col_wloc, formula_w, fmt_locked)

                    # λ_max, CI, CR (e formato condizionale)
                    label_lambda = r0 + 1 + num_k + 2
                    label_ci     = label_lambda + 1
                    label_cr     = label_lambda + 2
                    mat_start = xlsxwriter.utility.xl_rowcol_to_cell(r0 + 2,1)
                    mat_end   = xlsxwriter.utility.xl_rowcol_to_cell(r0 + 1 + num_k,1+num_k-1)
                    wloc_start= xlsxwriter.utility.xl_rowcol_to_cell(r0 + 2,col_wloc)
                    wloc_end  = xlsxwriter.utility.xl_rowcol_to_cell(r0 + 1 + num_k,col_wloc)

                    worksheet.write(label_lambda, 0, "λ_max", fmt_label)
                    worksheet.write_formula(label_lambda, 1,
                        f"=SUMPRODUCT({mat_start}:{mat_end},{wloc_start}:{wloc_end})/{num_k}", fmt_locked)
                    worksheet.write(label_ci, 0, "CI", fmt_label)
                    worksheet.write_formula(label_ci, 1,
                        f"=IF({num_k}>1,({xlsxwriter.utility.xl_rowcol_to_cell(label_lambda,1)}-{num_k})/({num_k}-1),0)", fmt_locked)
                    worksheet.write(label_cr, 0, "CR", fmt_label)
                    worksheet.write_formula(label_cr, 1,
                        f"=IFERROR({xlsxwriter.utility.xl_rowcol_to_cell(label_ci,1)}/VLOOKUP({num_k},RI_Lookup!$A:$B,2,FALSE),0)", fmt_locked)

                    cell_cr = xlsxwriter.utility.xl_rowcol_to_cell(label_cr, 1)
                    worksheet.conditional_format(cell_cr, {
                        "type":"cell","criteria":">","value":0.1,
                        "format":workbook.add_format({"bg_color":"#FFC7CE","font_color":"#9C0006"})
                    })
                    worksheet.conditional_format(cell_cr, {
                        "type":"cell","criteria":"<=","value":0.1,
                        "format":workbook.add_format({"bg_color":"#C6EFCE","font_color":"#006100"})
                    })

                    # Protezione finale
                    worksheet.protect(options={
                        "select_locked_cells": False,
                        "select_unlocked_cells": True
                    })

            writer.close()
            buf.seek(0)

        st.download_button(
            label="Scarica Versione Completa (Excel con Macro-Style)",
            data=buf,
            file_name="AHP_VersioneCompleta.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
