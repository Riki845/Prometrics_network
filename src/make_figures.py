#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py - genera TODAS las figuras del manuscrito (ProMetrics)
=====================================================================
Entradas (formato BibFusion, solo estas dos tablas; acepta nombres por fuente o integradas):
    Article.csv  | All_Articles.csv        columnas: SR, year, ismainarticle
    ArticleAuthor.csv | All_ArticleAuthor.csv   columnas: SR, PersonID

Salidas (carpeta --out, por defecto "figures"): cada figura en SVG (vector, texto en trazos),
PDF (vector, fuentes incrustadas) y PNG a 600 dpi, dibujadas a 6,575 in de ancho
(columna del manuscrito) con tipografía serif de 10 pt (ticks y leyendas 9 pt).

    fig1_article_composition   Fig. 1  composición del grupo de alta intermediación
    fig2_period_networks       Fig. 2  redes por ventana (réplica)
    fig3_network_structure     Fig. 3  grado, documentos por año, cobertura
    fig4_marginal_effects      Fig. 4  efectos marginales H1 y H2
    fig5_robustness_forest     Fig. 5  robustez (bosque de razones)
    fig6_model_diagnostics     Fig. 6  diagnóstico del modelo de conteo
    figS1_composition_replication   (suplementaria)

Uso:
    python src/make_figures.py --data data/bibfusion_output/integrated --out figures
    python src/make_figures.py --data data/bibfusion_output/integrated --fig1 data      # Fig. 1 calculada con TUS tablas

--fig1 manuscript (por defecto) dibuja la Figura 1 con las cifras ya reportadas en el texto
(78/334 y 62/137). Con las tablas integradas usa --fig1 data para que salga de los datos.
Todo el cálculo (red, métricas, modelos) vive en prometrics_network_analysis.py; este archivo
solo orquesta y dibuja, de modo que las figuras y las tablas nunca se desincronizan.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prometrics_network_analysis as pna


def robustness_specs(aa, R, cache, max_team, boot, seed):
    """Especificaciones del bosque de razones (Fig. 5): mismas definiciones que el script principal."""
    h1, h2 = R["h1"], R["h2"]
    t1 = pna.coef_table(h1["nb"], pna.H1_TERMS).iloc[1]
    tp = pna.coef_table(h1["pois"], pna.H1_TERMS).iloc[1]
    t2 = pna.coef_table(h2["fit"], pna.H2_TERMS).iloc[1]

    def spec(label, r, main=False, lo=None, hi=None):
        return {"label": label, "est": r["ratio"], "lo": r["ratio_lo"] if lo is None else lo,
                "hi": r["ratio_hi"] if hi is None else hi, "main": main}

    s1 = [spec("NB2 (principal)", t1, True), spec("Poisson QMLE", tp)]
    s2 = [spec("GEE logit (principal)", t2, True)]
    if boot > 0:
        c1, _ = pna.cluster_bootstrap(h1["X"], h1["y"], h1["g"], "nb2", 1, boot, seed, start=h1["nb"]["p"])
        c2, _ = pna.cluster_bootstrap(h2["X"], h2["y"], h2["g"], "logit", 1, boot, seed)
        s1.append(spec("NB2, bootstrap CI", t1, lo=c1[0], hi=c1[1]))
        s2.append(spec("GEE logit, bootstrap CI", t2, lo=c2[0], hi=c2[1]))
    Rb = pna.run_variant(aa, "big", max_team=max_team, metrics_cache=cache, verbose=False)
    a1 = pna.coef_table(Rb["h1"]["nb"], pna.H1_TERMS).iloc[1]
    a2 = pna.coef_table(Rb["h2"]["fit"], pna.H2_TERMS).iloc[1]
    s1.append(spec(f"Excluding teams ≥ {max_team}", a1))
    s2.append(spec(f"Excluding teams ≥ {max_team}", a2))
    for q in (0.80, 0.95):
        Rq = pna.run_variant(aa, "q", q=q, metrics_cache=cache, verbose=False)
        aq = pna.coef_table(Rq["h2"]["fit"], pna.H2_TERMS).iloc[1]
        s2.append(spec(f"Top {int(round((1 - q) * 100))}% threshold", aq))
    return s1, s2


def composition_rows(R):
    """Filas (ventana, retenidos, nuevos o retornados) calculadas desde los datos."""
    rows = []
    for i in range(1, len(pna.PNAMES)):
        hb, hn = R["high_sets"][pna.PNAMES[i - 1]], R["high_sets"][pna.PNAMES[i]]
        ret = len(hb & hn)
        rows.append((pna.PNAMES[i].replace("-", "–"), ret, len(hn) - ret))
    return tuple(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/bibfusion_output/integrated", help="carpeta con Article/ArticleAuthor o All_Articles/All_ArticleAuthor")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--fig1", choices=["manuscript", "data"], default="manuscript")
    ap.add_argument("--max-team", type=int, default=25)
    ap.add_argument("--boot", type=int, default=200, help="réplicas bootstrap para la Fig. 5 (0 = omitir)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pna.log("1) Cargando tablas y estimando modelos ...")
    aa, docs, ids, names, info = pna.load_data(Path(a.data))
    cache = {}
    R = pna.run_variant(aa, "principal", metrics_cache=cache, verbose=False)
    pna.log("2) Dibujando figuras ...")
    if a.fig1 == "data":
        pna.fig_article_composition(out, rows=composition_rows(R))
    else:
        pna.fig_article_composition(out)
    pna.fig_replication_composition(R, out)
    pna.fig2_networks(R, out)
    pna.fig3_structure(R, docs, out)
    pna.fig4_marginals(R, out)
    s1, s2 = robustness_specs(aa, R, cache, a.max_team, a.boot, a.seed)
    pna.fig5_forest(s1, s2, out)
    pna.fig6_diagnostics(R, out)
    pna.log(f"Listo en {time.time() - t0:.0f}s -> {out.resolve()}")


if __name__ == "__main__":
    main()
