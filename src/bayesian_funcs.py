from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from igraph import Graph, plot
from pgmpy.estimators import HillClimbSearch, BIC

@dataclass
class IBDNetworkResult:
    df: pd.DataFrame
    df_cd: pd.DataFrame

    full_bn: pd.DataFrame
    cd_bn: pd.DataFrame

    model_full: object
    model_cd: object

    g_full: Graph
    g_cd: Graph

    bic_full: float
    bic_cd: float

    betweenness_full: list
    betweenness_cd: list


def generate_ibd_networks(csv_path="../data/final_cleaned.csv", drop_nodes=False):
    """
    Authors: Fatehjot (original code), Elias (refactor into workflow func)

    Generate Bayesian networks for:
    1) full cohort
    2) Crohn's Disease only

    Parameters
    ----------
    csv_path : str
        Path to local CSV file.
    drop_nodes : bool, str, list[str]
        Column name or list of column names to drop before fitting.
        Default = False (drop nothing).

    Returns
    -------
    results : dict
        Dictionary containing cleaned data, models, graphs, and BIC scores.
    """

# TO ADD EVENTUALLY: Variations to scoring method (ie Bdeu)

    df = pd.read_csv(csv_path)

    categorical_cols = [
        "Duo_Inflammation",
        "Gastric_Inflammation",
        "LeftColon_Inflammation",
        "TI_Inflammation",
        "PGA"
    ]

    biomarker_cols = [
        "Hematocrit",
        "ESR",
        "CRP",
        "Albumin",
        "Vitamin D"
    ]

    # IN CASE WE WANT TO DROP VIT D
    if drop_nodes is not False:
        if isinstance(drop_nodes, str):
            drop_nodes = [drop_nodes]
        df = df.drop(columns=drop_nodes, errors="ignore")

        categorical_cols = [c for c in categorical_cols if c in df.columns]
        biomarker_cols = [c for c in biomarker_cols if c in df.columns]

    for col in biomarker_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = pd.qcut(
            df[col],
            q=3,
            labels=[1, 2, 3],
            duplicates="drop"
        )
    bn_cols = categorical_cols + biomarker_cols
    df[bn_cols] = df[bn_cols].astype(int)
    df_cd = df[df["Diagnosis"] == "Crohn's Disease"].copy()
    full_bn = df.drop(columns=["Diagnosis"]).copy()
    cd_bn = df_cd.drop(columns=["Diagnosis"]).copy()


    hc_full = HillClimbSearch(full_bn)
    model_full = hc_full.estimate(scoring_method=BIC(full_bn))
    bic_full = BIC(full_bn).score(model_full)

    hc_cd = HillClimbSearch(cd_bn)
    model_cd = hc_cd.estimate(scoring_method=BIC(cd_bn))
    bic_cd = BIC(cd_bn).score(model_cd)

    g_full = Graph.TupleList(model_full.edges(), directed=True)
    g_cd = Graph.TupleList(model_cd.edges(), directed=True)

    print("Dataset shape:", df.shape)
    print("Crohn's cases:", df_cd.shape)
    print(f"Full cohort BIC: {bic_full}")
    print(f"Crohn's Disease BIC: {bic_cd}")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot(
        g_full,
        target=ax,
        vertex_size=18,
        vertex_label=g_full.vs["name"],
        vertex_label_size=10,
        vertex_label_dist=3,
        edge_arrow_size=10,
        edge_width=2,
        edge_curved=False
    )
    ax.set_title("Full Cohort Bayesian Network")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    plot(
        g_cd,
        target=ax,
        vertex_size=18,
        vertex_label=g_cd.vs["name"],
        vertex_label_size=10,
        vertex_label_dist=3,
        edge_arrow_size=10,
        edge_width=2,
        edge_curved=False
    )
    ax.set_title("Crohn's Disease Bayesian Network")
    plt.show()

    g_full.vs["betweenness"] = g_full.betweenness(directed=True, normalized=True)
    full_bet = sorted(
        [(name, b) for name, b in zip(g_full.vs["name"], g_full.vs["betweenness"])],
        key=lambda x: x[1],
        reverse=True
    )

    g_cd.vs["betweenness"] = g_cd.betweenness(directed=True, normalized=True)
    cd_bet = sorted(
        [(name, b) for name, b in zip(g_cd.vs["name"], g_cd.vs["betweenness"])],
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop nodes by betweenness (Full Cohort):")
    for i, (name, val) in enumerate(full_bet):
        print(f"{i}: {name} ({val})")

    print("\nTop nodes by betweenness (Crohn's Disease):")
    for i, (name, val) in enumerate(cd_bet):
        print(f"{i}: {name} ({val})")

    return IBDNetworkResult(
        df=df,
        df_cd=df_cd,
        full_bn=full_bn,
        cd_bn=cd_bn,
        model_full=model_full,
        model_cd=model_cd,
        g_full=g_full,
        g_cd=g_cd,
        bic_full=bic_full,
        bic_cd=bic_cd,
        betweenness_full=full_bet,
        betweenness_cd=cd_bet
    )