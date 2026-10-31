#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProMetrics - Liderazgo relacional y colaboración en la red científica
=====================================================================
Reconstruye la red de coautoría desde las tablas BibFusion (Article.csv,
ArticleAuthor.csv, AuthorPerson.csv), estima los modelos inferenciales del
artículo (H1: binomial negativa NB2; H2: logística GEE con independencia) y
genera tablas y figuras listas para el manuscrito.

Uso (desde la raíz del repositorio):
    python src/prometrics_network_analysis.py --data data/bibfusion_output/integrated --out results
    python src/prometrics_network_analysis.py --selftest     # verifica estimadores

Privacidad: por defecto las listas de nodos/aristas y el GEXF se exportan con códigos
numéricos anónimos. Los nombres y PersonID solo se escriben con --export-identities
(úsese únicamente para trabajo interno; no publicar esos archivos).

Dependencias mínimas: numpy, pandas, scipy, networkx, matplotlib.
(statsmodels es OPCIONAL: si está instalado se hace un contraste cruzado.)
"""
import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no requiere pantalla: las figuras siempre se guardan
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln

import os
if not os.environ.get("PM_DEBUG"):
    warnings.filterwarnings("ignore")

PERIODS = {"2017-2019": (2017, 2019), "2020-2022": (2020, 2022), "2023-2025": (2023, 2025)}
PNAMES = list(PERIODS)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "gray": "#8c8c8c", "sky": "#56B4E9", "dark": "#222222"}

plt.rcParams.update({
    # Tipografía = texto del manuscrito (Times New Roman 10 pt). Liberation Serif es métricamente
    # idéntica a Times New Roman; si no está instalada se usa la siguiente de la lista.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.4,
    "legend.frameon": False,
    # Exportación vectorial: SVG con texto convertido en trazos (se ve igual en cualquier equipo),
    # PDF con fuentes TrueType incrustadas (editable en Illustrator/Inkscape).
    "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "figure.dpi": 110,
})


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# 1. CARGA Y RED
# ----------------------------------------------------------------------------
def _first(data_dir, *candidates):
    """Primer archivo existente: nombres de tablas por fuente (Article.csv) o integradas (All_Articles.csv)."""
    for c in candidates:
        if (data_dir / c).exists():
            return data_dir / c
    raise FileNotFoundError(f"No se encontró ninguno de {candidates} en {data_dir}")


def load_data(data_dir):
    """Devuelve tabla autor-documento (una fila por SR x persona) y tabla de documentos."""
    art = pd.read_csv(_first(data_dir, "Article.csv", "All_Articles.csv"), usecols=["SR", "year", "ismainarticle"],
                      low_memory=False)
    art["year"] = pd.to_numeric(art["year"], errors="coerce")
    art["ismainarticle"] = art["ismainarticle"].astype(str).str.lower().eq("true")
    # SR repetido en Article.csv: el año siempre coincide, se colapsa por SR
    docs = art.groupby("SR", as_index=False).agg(year=("year", "min"),
                                                  main=("ismainarticle", "max"))
    aa = pd.read_csv(_first(data_dir, "ArticleAuthor.csv", "All_ArticleAuthor.csv"), usecols=["SR", "PersonID"],
                     low_memory=False).dropna()
    n_raw = len(aa)
    aa = aa.drop_duplicates()
    n_dedup = len(aa)
    aa = aa.merge(docs, on="SR", how="inner")  # excluye autorías sin metadatos del artículo
    info = {"authorship_rows_raw": int(n_raw),
            "authorship_rows_after_dedup": int(n_dedup),
            "authorship_rows_without_article_metadata": int(n_dedup - len(aa)),
            "documents_in_article_table": int(len(docs)),
            "documents_with_authors": int(aa["SR"].nunique()),
            "documents_without_year": int(docs["year"].isna().sum()),
            "main_corpus_documents": int(docs["main"].sum())}
    codes, uniques = pd.factorize(aa["PersonID"])
    aa["person"] = codes
    ids = pd.Series(uniques, name="PersonID")
    names = None
    p = data_dir / "AuthorPerson.csv"
    if p.exists():
        ap = pd.read_csv(p, usecols=["PersonID", "CanonicalName", "Orcid"], low_memory=False)
        names = ids.to_frame().merge(ap.drop_duplicates("PersonID"), on="PersonID", how="left")
    elif (data_dir / "All_Authors.csv").exists():
        ap = pd.read_csv(data_dir / "All_Authors.csv", usecols=["PersonID", "AuthorFullName", "Orcid"], low_memory=False)
        ap = ap.rename(columns={"AuthorFullName": "CanonicalName"})
        names = ids.to_frame().merge(ap.drop_duplicates("PersonID"), on="PersonID", how="left")
    return aa, docs, ids, names, info


def make_pairs(aa):
    """Todos los pares de coautores por documento (clique); k autores -> k(k-1)/2 pares."""
    rows = []
    for sr, g in aa.groupby("SR"):
        ppl = sorted(g["person"].unique())
        if len(ppl) < 2:
            continue
        y = g["year"].iloc[0]
        rows.extend((u, v, sr, y) for u, v in itertools.combinations(ppl, 2))
    return pd.DataFrame(rows, columns=["u", "v", "SR", "year"])


def giant_component(aa, pairs):
    G = nx.Graph()
    G.add_nodes_from(aa["person"].unique())
    G.add_edges_from(pairs[["u", "v"]].drop_duplicates().itertuples(index=False, name=None))
    comp = max(nx.connected_components(G), key=len)
    return G, comp


# ----------------------------------------------------------------------------
# 2. MÉTRICAS POR PERIODO Y PANELES
# ----------------------------------------------------------------------------
def period_metrics(aa_c, pairs_c):
    """Grado, intermediación exacta normalizada y cuota de colaboración recurrente por ventana."""
    out = {}
    for name, (a, b) in PERIODS.items():
        t0 = time.time()
        adp = aa_c[(aa_c["year"] >= a) & (aa_c["year"] <= b)]
        active = adp["person"].unique()
        ndocs = adp.groupby("person")["SR"].nunique()
        pp = pairs_c[(pairs_c["year"] >= a) & (pairs_c["year"] <= b)]
        w = pp.groupby(["u", "v"]).size().reset_index(name="w")
        G = nx.Graph()
        G.add_nodes_from(active)
        G.add_weighted_edges_from(w[["u", "v", "w"]].itertuples(index=False, name=None))
        deg = pd.Series(dict(G.degree()))
        bc = pd.Series(nx.betweenness_centrality(G, normalized=True))
        rec_n = pd.concat([w.loc[w["w"] >= 2, "u"], w.loc[w["w"] >= 2, "v"]]).value_counts()
        df = pd.DataFrame({"ndocs": ndocs, "deg": deg, "bc": bc}).loc[active]
        df["rec_n"] = rec_n.reindex(df.index).fillna(0)
        df["rec"] = np.where(df["deg"] > 0, df["rec_n"] / df["deg"].where(df["deg"] > 0, 1), 0.0)
        out[name] = {"df": df, "G": G}
        log(f"   {name}: {len(df):,} activos, {G.number_of_edges():,} vínculos "
            f"({time.time() - t0:.1f}s)")
    return out


def build_panels(metrics, aa_c, pairs_c, q=0.90):
    """Panel identificador-periodo para H1 (todos los activos) y H2 (grupo alto base)."""
    d = aa_c.dropna(subset=["year"])
    first_year = d.groupby("person")["year"].min()
    pf = (pairs_c.dropna(subset=["year"]).groupby(["u", "v"])["year"].min()
          .reset_index(name="fy"))
    pf2 = pd.concat([pf.rename(columns={"u": "p", "v": "o"}),
                     pf.rename(columns={"v": "p", "u": "o"})])
    high_sets, thr = {}, {}
    for n in PNAMES:
        df = metrics[n]["df"]
        t = np.quantile(df["bc"], q)
        thr[n] = t
        df["high"] = ((df["bc"] > 0) & (df["bc"] >= t)).astype(int)
        high_sets[n] = set(df.index[df["high"] == 1])
    frames = []
    for i in range(len(PNAMES) - 1):
        b, n = PNAMES[i], PNAMES[i + 1]
        sb, eb = PERIODS[b][0], PERIODS[b][1]
        sn, en = PERIODS[n]
        df = metrics[b]["df"].copy()
        newc = pf2[(pf2["fy"] >= sn) & (pf2["fy"] <= en)].groupby("p").size()
        df["y_new"] = newc.reindex(df.index).fillna(0).astype(int)
        df["persist"] = [int(p in high_sets[n]) if h == 1 else np.nan
                         for p, h in zip(df.index, df["high"])]
        df["tenure"] = eb - first_year.reindex(df.index) + 1
        df["trans"] = i + 1
        df["base"], df["next"] = b, n
        df["person"] = df.index
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["tb"] = np.log1p(10000 * panel["bc"])
    panel["l_docs"] = np.log1p(panel["ndocs"])
    panel["l_deg"] = np.log1p(panel["deg"])
    panel["l_ten"] = np.log1p(panel["tenure"])
    panel["period2"] = (panel["trans"] == 2).astype(int)
    return panel, high_sets, thr


# ----------------------------------------------------------------------------
# 3. ESTIMADORES (NB2 por ML y GLM/GEE con covarianza robusta por conglomerados)
# ----------------------------------------------------------------------------
def _cluster_meat(scores, groups):
    s = pd.DataFrame(scores).groupby(np.asarray(groups)).sum().values
    return s.T @ s, len(s)


def fit_glm(X, y, family, groups, correction):
    """IRLS (Newton). 'logit' con covarianza de Liang-Zeger = GEE con independencia."""
    n, k = X.shape
    b = np.zeros(k)
    if family == "poisson":
        b[0] = np.log(max(y.mean(), 1e-6)) if np.allclose(X[:, 0], 1) else 0.0
    for _ in range(100):
        eta = np.clip(X @ b, -30, 30)
        mu = np.exp(eta) if family == "poisson" else expit(eta)
        w = mu if family == "poisson" else mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + 1e-10 * np.eye(k)
        step = np.linalg.solve(H, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ b, -30, 30)
    mu = np.exp(eta) if family == "poisson" else expit(eta)
    w = mu if family == "poisson" else mu * (1 - mu)
    bread = np.linalg.inv(X.T @ (X * w[:, None]) + 1e-10 * np.eye(k))
    meat, G = _cluster_meat(X * (y - mu)[:, None], groups)
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if correction and G > 1 else 1.0
    V = c * bread @ meat @ bread
    return {"b": b, "V": V, "mu": mu, "n": n, "G": G, "k": k, "converged": bool(np.max(np.abs(step)) < 1e-6)}


def _nb_nll_grad(p, X, y):
    b, la = p[:-1], p[-1]
    a = np.exp(la); r = 1.0 / a
    mu = np.exp(np.clip(X @ b, -30, 30))
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu))
          + y * np.log(mu / (r + mu)))
    sb = X * ((y - mu) / (1 + a * mu))[:, None]
    dr = digamma(y + r) - digamma(r) + np.log(r) + 1 - np.log(r + mu) - (r + y) / (r + mu)
    sla = -r * dr
    return -ll.sum(), -np.concatenate([sb.sum(0), [sla.sum()]]), np.column_stack([sb, sla])


def fit_nb2(X, y, groups, start=None):
    """Binomial negativa NB2 (enlace log), beta y dispersión conjuntamente por ML."""
    n, k = X.shape
    if start is None:
        pois = fit_glm(X, y, "poisson", groups, True)
        mu = pois["mu"]
        a0 = max(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum(), 0.05)
        start = np.concatenate([pois["b"], [np.log(a0)]])
    f = lambda p: _nb_nll_grad(p, X, y)[:2]
    res = minimize(f, start, jac=True, method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
    p = res.x
    # Hessiana numérica a partir del gradiente analítico
    h = 1e-5
    H = np.zeros((k + 1, k + 1))
    for j in range(k + 1):
        e = np.zeros(k + 1); e[j] = h
        H[:, j] = (f(p + e)[1] - f(p - e)[1]) / (2 * h)
    H = (H + H.T) / 2
    bread = np.linalg.inv(H)
    scores = -_nb_nll_grad(p, X, y)[2] * -1  # scores por observación (ll)
    scores = _nb_nll_grad(p, X, y)[2]
    meat, G = _cluster_meat(scores, groups)
    c = (G / (G - 1)) * ((n - 1) / (n - (k + 1)))
    V = c * bread @ meat @ bread
    a = float(np.exp(p[-1])); r = 1 / a
    mu = np.exp(X @ p[:-1])
    pearson = ((y - mu) ** 2 / (mu + a * mu ** 2)).sum() / (n - k - 1)
    return {"b": p[:-1], "V": V[:k, :k], "alpha": a, "mu": mu, "n": n, "G": G, "k": k,
            "loglik": float(-res.fun), "pearson": float(pearson),
            "zero_pred": float(np.mean((1 + a * mu) ** (-r))),
            "converged": bool(np.max(np.abs(res.jac)) < 1e-2), "p": p}


def coef_table(fit, names, exp=True):
    se = np.sqrt(np.diag(fit["V"]))
    z = fit["b"] / se
    t = pd.DataFrame({"term": names, "coef": fit["b"], "se": se, "z": z,
                      "p": 2 * stats.norm.sf(np.abs(z)),
                      "lo": fit["b"] - 1.96 * se, "hi": fit["b"] + 1.96 * se})
    if exp:
        t["ratio"], t["ratio_lo"], t["ratio_hi"] = np.exp(t["coef"]), np.exp(t["lo"]), np.exp(t["hi"])
    return t


H1_TERMS = ["const", "z_betweenness", "log_docs", "log_degree", "log_tenure", "period_2020_22_to_23_25"]
H2_TERMS = ["const", "recurrence_per_10pp", "z_betweenness", "log_docs", "log_degree", "log_tenure",
            "period_2020_22_to_23_25"]


def design_h1(panel):
    d = panel.copy()
    mu_, sd_ = d["tb"].mean(), d["tb"].std(ddof=1)
    d["z"] = (d["tb"] - mu_) / sd_
    X = np.column_stack([np.ones(len(d)), d["z"], d["l_docs"], d["l_deg"], d["l_ten"], d["period2"]])
    return d, X, d["y_new"].values.astype(float), d["person"].values, (mu_, sd_)


def design_h2(panel):
    d = panel[panel["high"] == 1].copy()
    mu_, sd_ = d["tb"].mean(), d["tb"].std(ddof=1)
    d["z"] = (d["tb"] - mu_) / sd_
    d["rec10"] = d["rec"] * 10
    X = np.column_stack([np.ones(len(d)), d["rec10"], d["z"], d["l_docs"], d["l_deg"], d["l_ten"], d["period2"]])
    return d, X, d["persist"].values.astype(float), d["person"].values, (mu_, sd_)


def cluster_bootstrap(X, y, groups, family, target, B, seed, start=None):
    rng = np.random.default_rng(seed)
    uid = pd.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uid}
    est = []
    for _ in range(B):
        s = rng.choice(uid, size=len(uid), replace=True)
        ix = np.concatenate([idx_by[g] for g in s])
        gnew = np.concatenate([np.full(len(idx_by[g]), i) for i, g in enumerate(s)])
        try:
            if family == "nb2":
                f = fit_nb2(X[ix], y[ix], gnew, start=start)
            else:
                f = fit_glm(X[ix], y[ix], "logit", gnew, False)
            est.append(f["b"][target])
        except Exception:
            continue
    est = np.array(est)
    return np.exp(np.percentile(est, [2.5, 97.5])), len(est)


# ----------------------------------------------------------------------------
# 4. PIPELINE COMPLETO (permite re-ejecutar con variantes de sensibilidad)
# ----------------------------------------------------------------------------
def run_variant(aa, label, q=0.90, max_team=None, metrics_cache=None, verbose=True):
    removed = {}
    if max_team is not None:
        sizes = aa.groupby("SR")["person"].nunique()
        big = sizes[sizes >= max_team].index
        removed = {"docs_removed": int(len(big)), "max_team_removed": int(sizes.loc[big].max()) if len(big) else 0}
        aa = aa[~aa["SR"].isin(big)]
    pairs = make_pairs(aa)
    G, comp = giant_component(aa, pairs)
    aa_c = aa[aa["person"].isin(comp)]
    pairs_c = pairs[pairs["u"].isin(comp)]
    if verbose:
        log(f"[{label}] componente gigante: {len(comp):,} nodos / "
            f"{pairs_c[['u','v']].drop_duplicates().shape[0]:,} vínculos")
    key = max_team
    if metrics_cache is not None and key in metrics_cache:
        metrics = metrics_cache[key]
    else:
        metrics = period_metrics(aa_c, pairs_c)
        if metrics_cache is not None:
            metrics_cache[key] = metrics
    panel, high_sets, thr = build_panels(metrics, aa_c, pairs_c, q=q)
    d1, X1, y1, g1, sc1 = design_h1(panel)
    nb = fit_nb2(X1, y1, g1)
    po = fit_glm(X1, y1, "poisson", g1, True)
    d2, X2, y2, g2, sc2 = design_h2(panel)
    lg = fit_glm(X2, y2, "logit", g2, False)
    return {"label": label, "G": G, "comp": comp, "aa_c": aa_c, "pairs_c": pairs_c, "metrics": metrics,
            "panel": panel, "high_sets": high_sets, "thr": thr, "removed": removed,
            "h1": {"d": d1, "X": X1, "y": y1, "g": g1, "nb": nb, "pois": po, "sc": sc1},
            "h2": {"d": d2, "X": X2, "y": y2, "g": g2, "fit": lg, "sc": sc2}}


# ----------------------------------------------------------------------------
# 5. FIGURAS (vectoriales; tipografía y tamaño = texto del manuscrito)
# ----------------------------------------------------------------------------
FIGW = 6.575  # ancho útil del manuscrito en pulgadas (16,7 cm): las figuras se dibujan a tamaño real


def savefig(fig, out, name):
    """Guarda SVG (texto convertido a trazos), PDF vectorial y PNG de 600 dpi."""
    fig.savefig(out / f"{name}.svg")
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)
    log(f"   figura -> {name}.svg / .pdf / .png")


def _wilson(k, n):
    z = 1.96; p = k / n
    cen = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n ** 2)) / (1 + z * z / n)
    return cen - half, cen + half


def fig_article_composition(out, rows=(("2020–2022", 78, 334), ("2023–2025", 62, 137))):
    """Figura 1 del artículo, dibujada con las cifras reportadas en el texto
    (conservadas: retenidos / nuevos o retornados por ventana)."""
    dark, light = "#25556B", "#BBCDD4"
    fig, ax = plt.subplots(figsize=(FIGW, 2.9), layout="constrained")
    x = np.arange(len(rows))
    ret = np.array([r[1] for r in rows]); new = np.array([r[2] for r in rows])
    ax.bar(x, ret, 0.45, color=dark, label="Retained from previous period")
    ax.bar(x, new, 0.45, bottom=ret, color=light, label="Entered or returned")
    for i in range(len(rows)):
        ax.text(i, ret[i] / 2, f"{ret[i]}", ha="center", va="center", color="white")
        ax.text(i, ret[i] + new[i] / 2, f"{new[i]}", ha="center", va="center")
        ax.text(i, ret[i] + new[i] + 8, f"{ret[i] + new[i]} total", ha="center", va="bottom")
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0, 520)
    ax.set_ylabel("High-betweenness identifiers"); ax.legend(loc="upper right")
    savefig(fig, out, "fig1_article_composition")


def fig_replication_composition(R, out):
    """Suplementaria: composición y persistencia en la réplica (no se usa en el manuscrito)."""
    rows = []
    for i in range(2):
        b, n = PNAMES[i], PNAMES[i + 1]
        hb, hn = R["high_sets"][b], R["high_sets"][n]
        ret = len(hb & hn)
        rows.append((n, ret, len(hn) - ret, len(hb)))
    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 5.4), layout="constrained")
    x = np.arange(2)
    ret = np.array([r[1] for r in rows]); new = np.array([r[2] for r in rows])
    ax[0].bar(x, ret, 0.45, color=C["blue"], label="Retained from previous group")
    ax[0].bar(x, new, 0.45, bottom=ret, color=C["orange"], label="Entered or returned")
    for i in range(2):
        ax[0].text(i, ret[i] / 2, f"{ret[i]}", ha="center", va="center", color="white")
        ax[0].text(i, ret[i] + new[i] / 2, f"{new[i]}", ha="center", va="center")
    ax[0].set_xticks(x, [r[0] for r in rows]); ax[0].set_ylim(0, max(r[1] + r[2] for r in rows) * 1.3)
    ax[0].set_ylabel("High-betweenness identifiers"); ax[0].set_title("(A) Composition of later groups")
    ax[0].legend(loc="upper right")
    prop = np.array([r[1] / r[3] for r in rows])
    ci_ = np.array([_wilson(r[1], r[3]) for r in rows])
    ax[1].bar(x, prop * 100, 0.45, color=C["green"])
    ax[1].errorbar(x, prop * 100, yerr=[(prop - ci_[:, 0]) * 100, (ci_[:, 1] - prop) * 100],
                   fmt="none", color="black", capsize=4)
    for i in range(2):
        ax[1].text(i + 0.28, prop[i] * 100 / 2, f"{prop[i]*100:.1f}%\n({rows[i][1]}/{rows[i][3]})", va="center")
    ax[1].set_xticks(x, [f"{PNAMES[i]} → {PNAMES[i+1]}" for i in range(2)])
    ax[1].set_ylim(0, ci_[:, 1].max() * 100 + 8); ax[1].set_ylabel("Persistence (%)")
    ax[1].set_title("(B) Persistence of the baseline group (Wilson 95% CI)")
    savefig(fig, out, "figS1_composition_replication")


def _draw_subgraph(G, bc, high, ax, label, max_draw=1200, seed=7):
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp).copy()
    if H.number_of_nodes() > max_draw:  # vista del 2-core si la red es muy grande
        core = nx.k_core(H, 2)
        if core.number_of_nodes() > 0:
            H = core
        if H.number_of_nodes() > max_draw:
            keep = sorted(H.degree, key=lambda t: -t[1])[:max_draw]
            H = H.subgraph([k for k, _ in keep]).copy()
            H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    pos = nx.spring_layout(H, seed=seed, k=1.4 / np.sqrt(max(H.number_of_nodes(), 1)), iterations=80)
    nodes = list(H.nodes)
    hi = np.array([n in high for n in nodes])
    size = 3 + 260 * np.sqrt(np.array([bc.get(n, 0) for n in nodes]))
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.2, alpha=0.25, edge_color=C["gray"])
    P = np.array([pos[n] for n in nodes])
    ax.scatter(P[~hi, 0], P[~hi, 1], s=size[~hi], c=C["sky"], alpha=0.85, linewidths=0)
    if hi.any():
        ax.scatter(P[hi, 0], P[hi, 1], s=size[hi], c=C["red"], alpha=0.95, linewidths=0.25, edgecolors="black")
    ax.set_title(label, loc="center")
    ax.set_xlabel(f"{H.number_of_nodes():,} nodes, {H.number_of_edges():,} links\n"
                  f"{int(hi.sum())} high-betweenness", linespacing=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig2_networks(R, out):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 3.2), layout="constrained")
    for ax, n in zip(axes, PNAMES):
        m = R["metrics"][n]
        _draw_subgraph(m["G"], m["df"]["bc"].to_dict(), R["high_sets"][n], ax, n)
    h = [Line2D([], [], marker="o", ls="", color=C["red"], mec="black", mew=0.3, ms=6, label="High-betweenness (top decile)"),
         Line2D([], [], marker="o", ls="", color=C["sky"], ms=6, label="Other active identifiers")]
    fig.legend(handles=h, loc="outside lower center", ncol=2)
    savefig(fig, out, "fig2_period_networks")


def fig3_structure(R, docs_table, out):
    G = R["G"].subgraph(R["comp"])
    deg = np.array([d for _, d in G.degree()])
    fig = plt.figure(figsize=(FIGW, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    v, cnt = np.unique(deg, return_counts=True)
    ccdf = 1 - np.cumsum(cnt) / cnt.sum() + cnt / cnt.sum()
    a.loglog(v, ccdf, "o", ms=3, color=C["blue"])
    a.set_xlabel("Degree k"); a.set_ylabel("P(K ≥ k)"); a.set_title("(A) Degree distribution")
    yrs = docs_table.dropna(subset=["year"])
    yrs = yrs[(yrs["year"] >= 2010) & (yrs["year"] <= 2026)]
    t = yrs.groupby(["year", "main"]).size().unstack(fill_value=0).reindex(columns=[False, True], fill_value=0)
    for (lo, hi_), alpha in zip(PERIODS.values(), [0.16, 0.08, 0.16]):
        b.axvspan(lo - 0.5, hi_ + 0.5, color=C["gray"], alpha=alpha, lw=0)
    b.bar(t.index, t[False], color=C["sky"], label="Reference-only")
    b.bar(t.index, t[True], bottom=t[False], color=C["red"], label="Main corpus")
    b.set_xticks(range(2010, 2027, 4)); b.set_xlabel("Publication year"); b.set_ylabel("Documents")
    b.set_title("(B) Documents per year"); b.legend(loc="upper left")
    rows = []
    for n in PNAMES:
        m = R["metrics"][n]
        rows.append((len(m["df"]), m["G"].number_of_edges(), int((m["df"]["bc"] > 0).sum()), len(R["high_sets"][n])))
    x = np.arange(3); w = 0.2
    for i, (lab, col) in enumerate(zip(["Active identifiers", "Distinct links", "Positive betweenness", "High betweenness"],
                                       [C["blue"], C["orange"], C["green"], C["red"]])):
        c.bar(x + (i - 1.5) * w, [r[i] for r in rows], w, label=lab, color=col)
    c.set_yscale("log"); c.set_xticks(x, PNAMES); c.set_ylabel("Count (log scale)")
    c.set_ylim(top=c.get_ylim()[1] * 4); c.set_title("(C) Coverage by window"); c.legend(loc="upper left", ncol=4, columnspacing=1.0, handlelength=1.0)
    savefig(fig, out, "fig3_network_structure")


def _binned(x, y, nb=10):
    d = pd.DataFrame({"x": x, "y": y})
    d["bin"] = pd.qcut(d["x"].rank(method="first"), nb, labels=False)
    g = d.groupby("bin").agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"), s=("y", "std")).reset_index()
    g["se"] = g["s"] / np.sqrt(g["n"])
    return g


def _bin_labels(x, y, lab):
    d = pd.DataFrame({"x": x, "y": y, "lab": lab})
    g = d.groupby("lab").agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"), s=("y", "std")).reset_index()
    g["se"] = g["s"].fillna(0) / np.sqrt(g["n"])
    return g


def fig4_marginals(R, out):
    h1, h2 = R["h1"], R["h2"]
    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 6.0), layout="constrained")
    X, nb = h1["X"], h1["nb"]
    zgrid = np.linspace(np.percentile(h1["d"]["z"], 1), np.percentile(h1["d"]["z"], 99), 100)
    G = np.tile(X.mean(0), (len(zgrid), 1)); G[:, 1] = zgrid
    eta = G @ nb["b"]; se = np.sqrt(np.einsum("ij,jk,ik->i", G, nb["V"], G))
    ax[0].fill_between(zgrid, np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se), color=C["blue"], alpha=0.2, lw=0)
    ax[0].plot(zgrid, np.exp(eta), color=C["blue"], lw=1.6, label="NB2 expected count (95% band)")
    d1 = h1["d"]; pos = d1["bc"].values > 0
    lab = np.zeros(len(d1), dtype=int)
    lab[pos] = 1 + pd.qcut(pd.Series(d1["z"].values[pos]).rank(method="first"), 6, labels=False).values
    bn = _bin_labels(d1["z"].values, h1["y"], lab); bn = bn[bn["n"] >= 8]
    ax[0].errorbar(bn["x"], bn["y"], yerr=1.96 * bn["se"], fmt="o", color=C["orange"], ms=4, capsize=2, label="Observed means")
    ax[0].set_xlabel("Standardized transformed betweenness (z)"); ax[0].set_ylabel("New collaborators")
    ax[0].set_title("(A) H1: expansion of collaboration"); ax[0].legend(loc="upper left")
    X2, f2 = h2["X"], h2["fit"]
    rgrid = np.linspace(0, min(10, h2["d"]["rec10"].max()), 100)
    G2 = np.tile(X2.mean(0), (len(rgrid), 1)); G2[:, 1] = rgrid
    eta = G2 @ f2["b"]; se = np.sqrt(np.einsum("ij,jk,ik->i", G2, f2["V"], G2))
    ax[1].fill_between(rgrid * 10, expit(eta - 1.96 * se), expit(eta + 1.96 * se), color=C["blue"], alpha=0.2, lw=0)
    ax[1].plot(rgrid * 10, expit(eta), color=C["blue"], lw=1.6, label="GEE-logit probability (95% band)")
    lab2 = pd.cut(h2["d"]["rec10"].values * 10, [-1, 0, 10, 25, 50, 100.1], labels=False)
    bn = _bin_labels(h2["d"]["rec10"].values * 10, h2["y"], lab2); bn = bn[bn["n"] >= 8]
    ax[1].errorbar(bn["x"], bn["y"], yerr=1.96 * bn["se"], fmt="o", color=C["orange"], ms=4, capsize=2, label="Observed shares")
    ax[1].set_xlabel("Recurrent-collaboration share (%)"); ax[1].set_ylabel("P(retain high betweenness)")
    ax[1].set_ylim(0, 1); ax[1].set_title("(B) H2: positional persistence"); ax[1].legend(loc="upper left")
    savefig(fig, out, "fig4_marginal_effects")


def fig5_forest(spec_h1, spec_h2, out):
    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 5.4), layout="constrained")
    for a, spec, ttl, xl in [(ax[0], spec_h1, "(A) H1: ratio per SD of transformed betweenness", "Expected-count ratio (log scale)"),
                             (ax[1], spec_h2, "(B) H2: odds ratio per +10 percentage points of recurrence", "Odds ratio (log scale)")]:
        yy = np.arange(len(spec))[::-1]
        for yv, s in zip(yy, spec):
            col = C["blue"] if s["main"] else C["gray"]
            a.errorbar(s["est"], yv, xerr=[[s["est"] - s["lo"]], [s["hi"] - s["est"]]], fmt="o", ms=4, color=col, capsize=3)
        a.axvline(1, color=C["red"], ls="--", lw=1)
        a.set_yticks(yy, [s["label"] for s in spec]); a.set_xscale("log")
        lo_, hi_ = min(s["lo"] for s in spec), max(s["hi"] for s in spec)
        tk = [t for t in (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3) if lo_ / 1.15 <= t <= hi_ * 1.15]
        a.set_xticks(tk); a.set_xticklabels([f"{t:g}" for t in tk]); a.minorticks_off()
        a.set_xlabel(xl); a.set_title(ttl)
    savefig(fig, out, "fig5_robustness_forest")


def fig6_diagnostics(R, out):
    h1 = R["h1"]; nb = h1["nb"]; y = h1["y"]; mu = nb["mu"]; a = nb["alpha"]; r = 1 / a
    kmax = 6
    obs = np.array([np.mean(y == k) for k in range(kmax)] + [np.mean(y >= kmax)])
    pn = np.array([np.mean(stats.nbinom.pmf(k, r, r / (r + mu))) for k in range(kmax)]); pn = np.append(pn, 1 - pn.sum())
    pp = np.array([np.mean(stats.poisson.pmf(k, h1["pois"]["mu"])) for k in range(kmax)]); pp = np.append(pp, 1 - pp.sum())
    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 5.6), layout="constrained")
    x = np.arange(kmax + 1); w = 0.27
    ax[0].bar(x - w, obs, w, label="Observed", color=C["dark"])
    ax[0].bar(x, pn, w, label="NB2", color=C["blue"])
    ax[0].bar(x + w, pp, w, label="Poisson", color=C["orange"])
    ax[0].set_xticks(x, [str(k) for k in range(kmax)] + [f"{kmax}+"]); ax[0].set_yscale("log")
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 2.2)
    ax[0].set_xlabel("New collaborators"); ax[0].set_ylabel("Proportion (log scale)")
    ax[0].set_title(f"(A) Count distribution: $\\hat{{\\alpha}}$ = {a:.2f}, Pearson $\\chi^2$/df = {nb['pearson']:.2f}")
    ax[0].legend(loc="upper right", ncol=3)
    d = pd.DataFrame({"mu": mu, "y": y}); d["bin"] = pd.qcut(d["mu"].rank(method="first"), 10, labels=False)
    g = d.groupby("bin").mean(); lim = max(g["mu"].max(), g["y"].max()) * 1.05
    ax[1].plot([0, lim], [0, lim], "--", color=C["gray"], lw=1)
    ax[1].plot(g["mu"], g["y"], "o", ms=5, color=C["blue"])
    ax[1].set_xlabel("Mean fitted count (decile)"); ax[1].set_ylabel("Mean observed count")
    ax[1].set_title("(B) Calibration by decile of fitted value")
    savefig(fig, out, "fig6_model_diagnostics")


# ----------------------------------------------------------------------------
# 6. TABLAS, EXPORTACIÓN Y AUTOPRUEBA
# ----------------------------------------------------------------------------
def network_tables(R, aa_all, docs, info):
    G = R["G"].subgraph(R["comp"])
    deg = np.array([d for _, d in G.degree()])
    main_ids = set(aa_all.loc[aa_all["main"], "person"])
    n = G.number_of_nodes(); m = G.number_of_edges()
    t1 = {
        "Author identifiers": n, "Distinct coauthorship links": m,
        "Identifiers attached to main corpus": len(main_ids & set(R["comp"])),
        "Reference-only identifiers": n - len(main_ids & set(R["comp"])),
        "Mean degree": round(float(deg.mean()), 2), "Median degree": float(np.median(deg)),
        "Density (%)": round(100 * 2 * m / (n * (n - 1)), 4),
        "Associated documents": int(R["aa_c"]["SR"].nunique()),
        "Documents with usable year": int(R["aa_c"].dropna(subset=["year"])["SR"].nunique()),
        "Giant component share of expanded network (%)": round(100 * n / aa_all["person"].nunique(), 1),
    }
    rows = []
    for nme in PNAMES:
        mm = R["metrics"][nme]
        a, b = PERIODS[nme]
        rows.append({"Window": nme,
                     "Documents": int(R["aa_c"][(R["aa_c"]["year"] >= a) & (R["aa_c"]["year"] <= b)]["SR"].nunique()),
                     "Active identifiers": len(mm["df"]), "Distinct links": mm["G"].number_of_edges(),
                     "Positive betweenness": int((mm["df"]["bc"] > 0).sum()),
                     "High-betweenness group": len(R["high_sets"][nme])})
    return t1, pd.DataFrame(rows)


def export_network(R, ids, names, aa_all, out, with_identities=False):
    """Exporta la red. Por defecto SIN nombres ni PersonID (códigos numéricos anónimos)."""
    G = R["G"].subgraph(R["comp"]).copy()
    main_ids = set(aa_all.loc[aa_all["main"], "person"])
    nd = aa_all[aa_all["person"].isin(R["comp"])].groupby("person")["SR"].nunique()
    fy = aa_all.dropna(subset=["year"]).groupby("person")["year"].min()
    rows = []
    for n_ in G.nodes:
        row = {"id": int(n_), "degree": int(G.degree(n_)), "n_docs": int(nd.get(n_, 0)),
               "in_main_corpus": bool(n_ in main_ids), "first_year": float(fy.get(n_, np.nan))}
        if with_identities:
            lab = names.iloc[n_]["CanonicalName"] if names is not None else ids.iloc[n_]
            row.update({"PersonID": ids.iloc[n_], "label": str(lab) if pd.notna(lab) else str(ids.iloc[n_])})
        rows.append(row)
    nodes = pd.DataFrame(rows)
    w = R["pairs_c"].groupby(["u", "v"]).size().reset_index(name="weight")
    nodes.to_csv(out / "network_nodes.csv", index=False)
    w.to_csv(out / "network_edges.csv", index=False)
    H = nx.Graph()
    for r in nodes.to_dict("records"):
        attrs = {k: v for k, v in r.items() if k not in ("id", "PersonID", "label") and not (isinstance(v, float) and np.isnan(v))}
        H.add_node(int(r["id"]), label=r.get("label", f"node_{r['id']}"), **attrs)
    H.add_weighted_edges_from(w.itertuples(index=False, name=None))
    nx.write_gexf(H, out / "network_giant_component.gexf")
    log(f"   exportado -> network_nodes.csv, network_edges.csv, network_giant_component.gexf "
        f"({'CON identidades' if with_identities else 'anónimo'})")


def crosscheck_statsmodels(R, tab):
    """Contraste OPCIONAL con statsmodels (si está instalado)."""
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Independence
        h1, h2 = R["h1"], R["h2"]
        nb = sm.NegativeBinomial(h1["y"], h1["X"]).fit(disp=0, cov_type="cluster",
                                                        cov_kwds={"groups": h1["g"]})
        gee = sm.GEE(h2["y"], h2["X"], groups=h2["g"], family=sm.families.Binomial(),
                     cov_struct=Independence()).fit()
        a = float(np.exp(nb.params[1])); b = float(np.exp(gee.params[1]))
        log(f"   statsmodels: H1 ratio={a:.4f} (propio {np.exp(h1['nb']['b'][1]):.4f}) | "
            f"H2 OR={b:.4f} (propio {np.exp(h2['fit']['b'][1]):.4f})")
        return {"h1_ratio_statsmodels": a, "h2_or_statsmodels": b}
    except Exception as e:
        log(f"   (contraste con statsmodels omitido: {type(e).__name__})")
        return {}


def selftest():
    """Verifica los estimadores con datos simulados de parámetros conocidos."""
    rng = np.random.default_rng(1)
    n = 6000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([-0.3, 0.5, -0.2]); alpha = 1.5
    mu = np.exp(X @ beta); r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    f = fit_nb2(X, y, np.arange(n))
    ok1 = np.all(np.abs(f["b"] - beta) < 0.1) and abs(f["alpha"] - alpha) < 0.3
    log(f"NB2: beta_hat={np.round(f['b'],3)} alpha_hat={f['alpha']:.2f}  -> {'OK' if ok1 else 'FALLA'}")
    p = expit(X @ np.array([0.2, 0.8, -0.5])); yb = rng.binomial(1, p).astype(float)
    g = fit_glm(X, yb, "logit", np.arange(n), False)
    ok2 = np.all(np.abs(g["b"] - np.array([0.2, 0.8, -0.5])) < 0.12)
    log(f"Logit: beta_hat={np.round(g['b'],3)} -> {'OK' if ok2 else 'FALLA'}")
    # gradiente analítico vs numérico
    from scipy.optimize import check_grad
    p0 = np.concatenate([beta, [np.log(alpha)]])
    err = check_grad(lambda q: _nb_nll_grad(q, X, y)[0], lambda q: _nb_nll_grad(q, X, y)[1], p0 + 0.05)
    log(f"Gradiente NB2: error {err:.3e} -> {'OK' if err < 1e-1 else 'FALLA'}")
    try:
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000).fit(X, yb)
        log(f"Logit vs scikit-learn: dif. máx {np.max(np.abs(sk.coef_[0]-g['b'])):.2e}")
    except Exception:
        pass
    return ok1 and ok2


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/bibfusion_output/integrated", help="carpeta con Article/ArticleAuthor (por fuente) o All_Articles/All_ArticleAuthor (integradas)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-team", type=int, default=25, help="tamaño de equipo que se excluye en la sensibilidad")
    ap.add_argument("--boot", type=int, default=200, help="réplicas bootstrap por conglomerados (0 = omitir)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-identities", action="store_true",
                    help="escribe nombres y PersonID en los archivos de red (NO publicar)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figd, tabd, netd = out / "figures", out / "tables", out / "network"
    for d_ in (figd, tabd, netd):
        d_.mkdir(exist_ok=True)
    t0 = time.time()
    log("1) Cargando tablas ..."); aa, docs, ids, names, info = load_data(data_dir)
    for k, v in info.items():
        log(f"   {k}: {v:,}")
    log("2) Red, componente gigante y métricas por ventana ...")
    cache = {}
    R = run_variant(aa, "principal", q=0.90, metrics_cache=cache)
    h1, h2 = R["h1"], R["h2"]
    log(f"   panel H1: {len(h1['d']):,} obs / {h1['d']['person'].nunique():,} identificadores; "
        f"panel H2: {len(h2['d'])} obs ({int(h2['y'].sum())} persisten)")

    # ---- Tablas descriptivas
    t1, t2 = network_tables(R, aa, docs, info)
    pd.Series(t1).to_csv(tabd / "table1_network.csv", header=["value"])
    t2.to_csv(tabd / "table2_periods.csv", index=False)

    # ---- Modelos principales
    nbt = coef_table(h1["nb"], H1_TERMS); pot = coef_table(h1["pois"], H1_TERMS)
    lgt = coef_table(h2["fit"], H2_TERMS)
    nbt.to_csv(tabd / "model_H1_NB2_full.csv", index=False); lgt.to_csv(tabd / "model_H2_GEE_full.csv", index=False)
    pot.to_csv(tabd / "model_H1_PoissonQMLE_full.csv", index=False)
    h1_main = nbt.iloc[1]; h2_main = lgt.iloc[1]

    # ---- Modelo "hurdle" y heterogeneidad temporal (extensiones inferenciales)
    anyc = fit_glm(h1["X"], (h1["y"] > 0).astype(float), "logit", h1["g"], True)
    anyt = coef_table(anyc, H1_TERMS)
    Xi = np.column_stack([h1["X"], h1["X"][:, 1] * h1["X"][:, 5]])
    nbi = fit_nb2(Xi, h1["y"], h1["g"])
    inti = coef_table(nbi, H1_TERMS + ["z_betweenness_x_period2"])
    inti.to_csv(tabd / "model_H1_NB2_interaction.csv", index=False)

    # ---- Bootstrap por conglomerados
    boot = {}
    if args.boot > 0:
        log(f"3) Bootstrap por conglomerados (B={args.boot}) ...")
        ci1, ok1 = cluster_bootstrap(h1["X"], h1["y"], h1["g"], "nb2", 1, args.boot, args.seed, start=h1["nb"]["p"])
        ci2, ok2 = cluster_bootstrap(h2["X"], h2["y"], h2["g"], "logit", 1, args.boot, args.seed)
        boot = {"H1": (ci1, ok1), "H2": (ci2, ok2)}
        log(f"   H1 IC bootstrap {ci1.round(3)} ({ok1} réplicas) | H2 IC bootstrap {ci2.round(3)} ({ok2} réplicas)")

    # ---- Sensibilidad
    log("4) Sensibilidad ...")
    spec1 = [{"label": "NB2 (principal)", "est": h1_main["ratio"], "lo": h1_main["ratio_lo"], "hi": h1_main["ratio_hi"], "main": True},
             {"label": "Poisson QMLE", "est": pot.iloc[1]["ratio"], "lo": pot.iloc[1]["ratio_lo"], "hi": pot.iloc[1]["ratio_hi"], "main": False}]
    spec2 = [{"label": "GEE logit (principal)", "est": h2_main["ratio"], "lo": h2_main["ratio_lo"], "hi": h2_main["ratio_hi"], "main": True}]
    if boot:
        spec1.append({"label": "NB2, bootstrap CI", "est": h1_main["ratio"], "lo": boot["H1"][0][0], "hi": boot["H1"][0][1], "main": False})
        spec2.append({"label": "GEE logit, bootstrap CI", "est": h2_main["ratio"], "lo": boot["H2"][0][0], "hi": boot["H2"][0][1], "main": False})
    sens_rows = [{"variant": "principal", "H1_ratio": h1_main["ratio"], "H1_lo": h1_main["ratio_lo"], "H1_hi": h1_main["ratio_hi"],
                  "H1_p": h1_main["p"], "H2_OR": h2_main["ratio"], "H2_lo": h2_main["ratio_lo"], "H2_hi": h2_main["ratio_hi"],
                  "H2_p": h2_main["p"], "H2_n": h2["fit"]["n"]}]
    Rbig = run_variant(aa, f"sin equipos >= {args.max_team}", max_team=args.max_team, metrics_cache=cache)
    for tag, RR in [(f"Excluding teams ≥ {args.max_team}", Rbig)]:
        a1 = coef_table(RR["h1"]["nb"], H1_TERMS).iloc[1]; a2 = coef_table(RR["h2"]["fit"], H2_TERMS).iloc[1]
        spec1.append({"label": tag, "est": a1["ratio"], "lo": a1["ratio_lo"], "hi": a1["ratio_hi"], "main": False})
        spec2.append({"label": tag, "est": a2["ratio"], "lo": a2["ratio_lo"], "hi": a2["ratio_hi"], "main": False})
        sens_rows.append({"variant": tag, "H1_ratio": a1["ratio"], "H1_lo": a1["ratio_lo"], "H1_hi": a1["ratio_hi"], "H1_p": a1["p"],
                          "H2_OR": a2["ratio"], "H2_lo": a2["ratio_lo"], "H2_hi": a2["ratio_hi"], "H2_p": a2["p"], "H2_n": RR["h2"]["fit"]["n"],
                          **RR["removed"]})
    for qq in (0.80, 0.95):
        RQ = run_variant(aa, f"umbral top {int(round((1-qq)*100))}%", q=qq, metrics_cache=cache, verbose=False)
        a2 = coef_table(RQ["h2"]["fit"], H2_TERMS).iloc[1]
        spec2.append({"label": f"Top {int(round((1-qq)*100))}% threshold", "est": a2["ratio"], "lo": a2["ratio_lo"], "hi": a2["ratio_hi"], "main": False})
        sens_rows.append({"variant": f"top {int(round((1-qq)*100))}%", "H2_OR": a2["ratio"], "H2_lo": a2["ratio_lo"], "H2_hi": a2["ratio_hi"],
                          "H2_p": a2["p"], "H2_n": RQ["h2"]["fit"]["n"]})
    pd.DataFrame(sens_rows).to_csv(tabd / "table_sensitivity.csv", index=False)

    # ---- Resumen compacto para el manuscrito
    t3 = pd.DataFrame([
        {"Association": "H1 prior betweenness -> new collaborators (NB2)", "Ratio": h1_main["ratio"], "CI_lo": h1_main["ratio_lo"],
         "CI_hi": h1_main["ratio_hi"], "p": h1_main["p"], "N": h1["nb"]["n"]},
        {"Association": "H2 recurrence -> persistence (GEE logit)", "Ratio": h2_main["ratio"], "CI_lo": h2_main["ratio_lo"],
         "CI_hi": h2_main["ratio_hi"], "p": h2_main["p"], "N": h2["fit"]["n"]}])
    t3.to_csv(tabd / "table3_main_associations.csv", index=False)

    # ---- Persistencia por transición
    pers = []
    for i in range(2):
        d = h2["d"][h2["d"]["trans"] == i + 1]
        pers.append({"transition": f"{PNAMES[i]} -> {PNAMES[i+1]}", "eligible": len(d), "persisted": int(d["persist"].sum()),
                     "proportion": float(d["persist"].mean()), "distinct_baseline_ids": int(d["person"].nunique())})
    pers_df = pd.DataFrame(pers); pers_df.to_csv(tabd / "table_persistence.csv", index=False)

    # ---- Paneles
    R["panel"].drop(columns=["person"]).assign(PersonCode=R["panel"]["person"].values).to_csv(tabd / "panel_identifier_period.csv", index=False)

    # ---- Figuras
    log("5) Figuras ...")
    fig_article_composition(figd); fig_replication_composition(R, figd)
    fig2_networks(R, figd); fig3_structure(R, docs, figd)
    fig4_marginals(R, figd); fig5_forest(spec1, spec2, figd); fig6_diagnostics(R, figd)
    export_network(R, ids, names, aa.merge(docs[["SR"]], on="SR"), netd, with_identities=args.export_identities)
    xc = crosscheck_statsmodels(R, None)

    # ---- JSON de resultados
    res = {"data_info": info, "table1": t1, "table2": t2.to_dict("records"),
           "H1": {"ratio": h1_main["ratio"], "ci": [h1_main["ratio_lo"], h1_main["ratio_hi"]], "p": h1_main["p"], "n": h1["nb"]["n"],
                  "n_clusters": h1["nb"]["G"], "alpha": h1["nb"]["alpha"], "pearson_dispersion": h1["nb"]["pearson"],
                  "mean_y": float(h1["y"].mean()), "var_y": float(h1["y"].var(ddof=1)), "share_zero_obs": float((h1["y"] == 0).mean()),
                  "share_zero_pred": h1["nb"]["zero_pred"], "converged": h1["nb"]["converged"],
                  "bootstrap_ci": boot["H1"][0].tolist() if boot else None,
                  "any_new_collab_odds_ratio": [anyt.iloc[1]["ratio"], anyt.iloc[1]["ratio_lo"], anyt.iloc[1]["ratio_hi"]],
                  "interaction_z_x_period2_ratio": [inti.iloc[6]["ratio"], inti.iloc[6]["ratio_lo"], inti.iloc[6]["ratio_hi"], inti.iloc[6]["p"]]},
           "H2": {"or": h2_main["ratio"], "ci": [h2_main["ratio_lo"], h2_main["ratio_hi"]], "p": h2_main["p"], "n": h2["fit"]["n"],
                  "persisted": int(h2["y"].sum()), "n_clusters": h2["fit"]["G"],
                  "bootstrap_ci": boot["H2"][0].tolist() if boot else None, "converged": h2["fit"]["converged"]},
           "persistence": pers, "thresholds_bc": R["thr"], "sensitivity": sens_rows, "statsmodels_crosscheck": xc}
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"\nListo en {time.time() - t0:.0f}s. Resultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
