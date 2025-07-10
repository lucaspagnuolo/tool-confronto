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
    RI_dict = {1: 0.0, 2: 0.0, 3: 0.489, 4: 0.805, 5: 1.059,
               6: 1.18, 7: 1.252, 8: 1.317, 9: 1.373, 10: 1.406,
               11: 1.419, 12: 1.445, 13: 1.46, 14: 1.471, 15: 1.485}
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
    lot_names = [st.sidebar.text_input(f"Nome Lotto {i+1}", value=f"Lotto{i+1}") for i in range(num_lots)]
    criterion_names = [st.sidebar.text_input(f"Criterio {j+1}", value=f"Criterio{j+1}") for j in range(num_criteria)]
    competitor_names = [st.sidebar.text_input(f"Concorrente {k+1}", value=f"Conc{k+1}") for k in range(num_competitors)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Assegna PTmax ai criteri")
    ptmax_values = []
    for j, crit in enumerate(criterion_names):
        v = st.sidebar.number_input(f"PTmax per '{crit}'", min_value=0.0, value=1.0, step=0.5, key=f"ptmax_{j}")
        ptmax_values.append(v)

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

    report_data = {}
    for idx, lot in enumerate(lot_names):
        st.header(f"Valutazione Lotto: **{lot}**")

        comp_results = {}
        for i_crit, crit in enumerate(criterion_names):
            st.markdown(f"**Criterio {i_crit+1}: '{crit}'**")
            comp_matrix = np.ones((num_competitors, num_competitors))
            for i in range(num_competitors):
                for j in range(i+1, num_competitors):
                    slider_key = f"lot_{idx}_crit_{i_crit}_comp_{i}_{j}"
                    v = st.slider(f"{competitor_names[i]} vs {competitor_names[j]}", min_value=-8.0, max_value=8.0, value=0.0, step=0.5, key=slider_key)
                    if v > 0:
                        val = 1.0 + v
                        desc = f"Preferisco **{competitor_names[i]}** di **{val:.1f}×**"
                    elif v < 0:
                        val = 1.0 / (1.0 + abs(v))
                        desc = f"Preferisco **{competitor_names[j]}** di **{(1+abs(v)):.1f}×**"
                    else:
                        val = 1.0
                        desc = "Perfetto equilibrio"
                    comp_matrix[i, j] = val
                    comp_matrix[j, i] = 1.0 / val
                    st.write(desc)

            geom_comp, weights_comp = compute_ahp_weights(comp_matrix)
            lam_c, ci_c, cr_c = consistency_ratio(comp_matrix, weights_comp)

            comp_results[crit] = {
                "matrix": pd.DataFrame(comp_matrix, index=competitor_names, columns=competitor_names),
                "weights": weights_comp,
                "metrics": {"lambda_max": lam_c, "CI": ci_c, "CR": cr_c}
            }

        # Sezione normalizzazione parziale
        st.subheader("Punteggi parziali normalizzati per criterio")
        norm_data = {}
        for crit in criterion_names:
            w_loc = comp_results[crit]["weights"]
            max_w = w_loc.max()
            norm_data[crit] = np.round(w_loc / max_w, 4)
        df_norm = pd.DataFrame(norm_data, index=competitor_names)
        st.dataframe(df_norm.style.format("{:.4f}"))

        # Calcolo punteggi complessivi
        overall_scores = {}
        for c_name in competitor_names:
            score = 0.0
            for j, crit in enumerate(criterion_names):
                w_local = comp_results[crit]["weights"][competitor_names.index(c_name)]
                w_crit = criterion_weights[j]
                score += w_local * w_crit
            overall_scores[c_name] = score

        series_overall = pd.Series(overall_scores, name="Punteggio complessivo").sort_values(ascending=False)
        st.subheader("Punteggi complessivi dei concorrenti")
        st.dataframe(series_overall.to_frame().style.format("{:.4f}"))
        st.markdown("---")

        report_data[lot] = {
            "criteri": {"names": criterion_names, "ptmax": ptmax_values, "peso_criteri": criterion_weights},
            "competitors": competitor_names,
            "comp_results": comp_results,
            "overall": series_overall
        }

    # Esportazione Excel completa
    if report_data:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            workbook = writer.book
            # Formati e sheet RI come prima
            fmt_header = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#4BACC6", "align": "center", "valign": "vcenter", "border": 1})
            fmt_title = workbook.add_format({"bold": True, "font_size": 14, "font_color": "#FFFFFF", "bg_color": "#2F75B5", "align": "left", "valign": "vcenter"})
            fmt_label = workbook.add_format({"bold": True, "font_color": "#444444", "bg_color": "#E7E6E6", "align": "left", "valign": "vcenter", "border": 1})
            fmt_input = workbook.add_format({"bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter", "locked": False})
            fmt_locked = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "locked": True})
            fmt_bordo = workbook.add_format({"border": 1})
            fmt_bordo_center = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})

            ri_sheet = workbook.add_worksheet("RI_Lookup")
            ri_sheet.hide()
            ri_vals = {1:0.0,2:0.0,3:0.489,4:0.805,5:1.059,6:1.18,7:1.252,8:1.317,9:1.373,10:1.406,11:1.419,12:1.445,13:1.46,14:1.471,15:1.485}
            ri_sheet.write(0,0,"n"); ri_sheet.write(0,1,"RI")
            r=1
            for n_val, ri_val in ri_vals.items():
                ri_sheet.write(r,0,n_val); ri_sheet.write(r,1,ri_val); r+=1

            for lot, data in report_data.items():
                ws = workbook.add_worksheet(lot)
                ws.protect(options={"format_cells":False})
                ws.set_column("A:A",20); ws.set_column("B:C",15); ws.set_column("D:Z",12)
                ws.merge_range("A2:C2","Titolo Iniziativa:",fmt_title)
                ws.write("D2", lot, fmt_locked)
                ws.write("A3","Num Criteri:",fmt_label);
                ws.write("B3",len(data["criteri"]["names"]),fmt_locked)
                ws.write("A4","Num Concorrenti:",fmt_label);
                ws.write("B4",len(data["competitors"]),fmt_locked)
                ws.write("A6","1) Inserisci ID Criterio e PTmax nelle corrispondenti celle.",fmt_bordo)
                ws.write("A7","2) Inserisci la valutazione di ogni coppia di offerte nelle celle gialle.",fmt_bordo)

                # ID criteri e PTmax
                ws.write("A8","ID Criterio",fmt_header); ws.write("B8","PTmax",fmt_header)
                for j, crit in enumerate(data["criteri"]["names"]):
                    ws.write(9+j,0,crit,fmt_bordo); ws.write(9+j,1,data["criteri"]["ptmax"][j],fmt_input)

                num_c = len(data["criteri"]["names"]); num_k = len(data["competitors"])
                start = 9+num_c+2
                for i_crit, crit in enumerate(data["criteri"]["names"]):
                    r0 = start + i_crit*(num_k+8)
                    ws.merge_range(r0,0,r0,1,f"Criterio: {crit}",fmt_title)
                    ws.write(r0+1,0,"Concorrente",fmt_header)
                    for j, comp in enumerate(data["competitors"]): ws.write(r0+1,1+j,comp,fmt_header)
                    for i, comp in enumerate(data["competitors"]): ws.write(r0+2+i,0,comp,fmt_header)
                    mat = data["comp_results"][crit]["matrix"].values
                    for i in range(num_k):
                        for j in range(num_k):
                            fmt = fmt_input if j>i else fmt_locked
                            ws.write(r0+2+i,1+j,mat[i,j],fmt)
                    # Geom & Peso
                    cg,col = 1+num_k+1,1+num_k+2
                    ws.write(r0+1,cg,"GeomMean",fmt_header); ws.write(r0+1,col,"PesoLocale",fmt_header)
                    gm_start = xlsxwriter.utility.xl_rowcol_to_cell(r0+2,cg)
                    gm_end   = xlsxwriter.utility.xl_rowcol_to_cell(r0+1+num_k,cg)
                    for i in range(num_k):
                        r1=r0+2+i
                        ws.write_formula(r1,cg,f"=GEOMEAN({xlsxwriter.utility.xl_range(r1,1,r1,1+num_k-1)})",fmt_locked)
                        ws.write_formula(r1,col,f"=IFERROR({xlsxwriter.utility.xl_rowcol_to_cell(r1,cg)}/SUM({gm_start}:{gm_end}),0)",fmt_locked)
                    # Lambda, CI, CR
                    lm=r0+2+num_k+2; ci=r0+2+num_k+3; cr=r0+2+num_k+4
                    ws.write(lm,0,"λ_max",fmt_label)
                    ws.write_formula(lm,1,f"=SUMPRODUCT({xlsxwriter.utility.xl_rowcol_to_cell(r0+2,1)}:{xlsxwriter.utility.xl_rowcol_to_cell(r0+1+num_k,1+num_k-1)},{xlsxwriter.utility.xl_rowcol_to_cell(r0+2,col)}:{xlsxwriter.utility.xl_rowcol_to_cell(r0+1+num_k,col)})/{num_k}",fmt_locked)
                    ws.write(ci,0,"CI",fmt_label)
                    ws.write_formula(ci,1,f"=( {xlsxwriter.utility.xl_rowcol_to_cell(lm,1)} - {num_k} ) / ({num_k}-1)",fmt_locked)
                    ws.write(cr,0,"CR",fmt_label)
                    ws.write_formula(cr,1,f"=IFERROR({xlsxwriter.utility.xl_rowcol_to_cell(ci,1)}/VLOOKUP({num_k},RI_Lookup!$A:$B,2,FALSE),0)",fmt_locked)
                    cell_cr = xlsxwriter.utility.xl_rowcol_to_cell(cr,1)
                    ws.conditional_format(cell_cr,{"type":"cell","criteria":">","value":0.1,"format":workbook.add_format({"bg_color":"#FFC7CE","font_color":"#9C0006"})})
                    ws.conditional_format(cell_cr,{"type":"cell","criteria":"<=","value":0.1,"format":workbook.add_format({"bg_color":"#C6EFCE","font_color":"#006100"})})
                    ws.protect(options={"select_locked_cells":False,"select_unlocked_cells":True})

            writer.close()
            buf.seek(0)

        st.download_button(label="Scarica Versione Completa (Excel con Macro-Style)", data=buf, file_name="AHP_VersioneCompleta.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
