from dataclasses import dataclass
from typing import Optional, Any
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

from pgmpy.estimators import HillClimbSearch, BIC, BDeu
from igraph import Graph, plot

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci

from helper_funcs import *

logging.getLogger("pgmpy").setLevel(logging.ERROR)

@dataclass
class IBDNetworkResult:
    algorithm: str

    # source / subsets
    df: pd.DataFrame
    df_cd: pd.DataFrame

    # matrices actually used for fitting
    full_data: pd.DataFrame
    cd_data: pd.DataFrame

    # learned model objects
    model_full: Optional[Any] = None
    model_cd: Optional[Any] = None

    # graph objects or graph-like outputs
    g_full: Optional[Any] = None
    g_cd: Optional[Any] = None

    # optional scores (used for BN, not PC)
    score_full: Optional[float] = None
    score_cd: Optional[float] = None

def generate_ibd_network(
    method: str,
    csv_path="../data/final_cleaned.csv",
    drop_nodes=False,
    alpha=0.05,
    bn_score="bic"
):
    """
    Unified network generation function.

    Parameters
    ----------
    method : str
        "bn" for discretize -> HillClimb -> score-based BN -> directed graph
        "pc" for continuous+ordinal -> PC -> KCI -> partially directed graph
    csv_path : str
        Path to local CSV.
    drop_nodes : bool, str, list[str]
        Columns to drop before fitting. Default False.
    alpha : float
        Significance threshold for PC algorithm.
    bn_score : str
        Scoring method for Bayesian network structure learning.
        Supported: "bic", "bdeu"

    Returns
    -------
    IBDNetworkResult
    """

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

    if drop_nodes is not False:
        if isinstance(drop_nodes, str):
            drop_nodes = [drop_nodes]
        df = df.drop(columns=drop_nodes, errors="ignore")
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        biomarker_cols = [c for c in biomarker_cols if c in df.columns]

    # -------------------------
    # METHOD 1: Bayesian Network
    # discretize biomarkers
    # -------------------------
    if method.lower() == "bn":
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

        df_model = df[bn_cols + ["Diagnosis"]].dropna().copy()
        df_cd = df_model[df_model["Diagnosis"] == "Crohn's Disease"].copy()

        full_data = df_model.drop(columns=["Diagnosis"]).copy()
        cd_data = df_cd.drop(columns=["Diagnosis"]).copy()

        bn_score = bn_score.lower()

        if bn_score == "bic":
            scorer_full = BIC(full_data)
            scorer_cd = BIC(cd_data)
        elif bn_score == "bdeu":
            scorer_full = BDeu(full_data)
            scorer_cd = BDeu(cd_data)
        else:
            raise ValueError("bn_score must be either 'bic' or 'bdeu'")

        hc_full = HillClimbSearch(full_data)
        model_full = hc_full.estimate(scoring_method=scorer_full, show_progress=False)
        score_full = scorer_full.score(model_full)

        hc_cd = HillClimbSearch(cd_data)
        model_cd = hc_cd.estimate(scoring_method=scorer_cd, show_progress=False)
        score_cd = scorer_cd.score(model_cd)

        g_full = Graph.TupleList(model_full.edges(), directed=True)
        g_cd = Graph.TupleList(model_cd.edges(), directed=True)

        return IBDNetworkResult(
            algorithm=f"hillclimb_bn_{bn_score}",
            df=df_model,
            df_cd=df_cd,
            full_data=full_data,
            cd_data=cd_data,
            model_full=model_full,
            model_cd=model_cd,
            g_full=g_full,
            g_cd=g_cd,
            score_full=score_full,
            score_cd=score_cd
        )

    # -------------------------
    # METHOD 2: PC + KCI
    # keep biomarkers continuous
    # -------------------------
    elif method.lower() == "pc":
        for col in biomarker_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        pc_cols = categorical_cols + biomarker_cols
        df_model = df[pc_cols + ["Diagnosis"]].dropna().copy()
        df_cd = df_model[df_model["Diagnosis"] == "Crohn's Disease"].copy()

        full_data = df_model.drop(columns=["Diagnosis"]).astype(float).copy()
        cd_data = df_cd.drop(columns=["Diagnosis"]).astype(float).copy()

        model_full = pc(
            full_data.to_numpy(),
            alpha=alpha,
            indep_test=kci,
            show_progress=True,
            node_names=list(full_data.columns)
        )

        model_cd = pc(
            cd_data.to_numpy(),
            alpha=alpha,
            indep_test=kci,
            show_progress=True,
            node_names=list(cd_data.columns)
        )

        g_full = model_full.G
        g_cd = model_cd.G

        return IBDNetworkResult(
            algorithm="pc_kci",
            df=df_model,
            df_cd=df_cd,
            full_data=full_data,
            cd_data=cd_data,
            model_full=model_full,
            model_cd=model_cd,
            g_full=g_full,
            g_cd=g_cd,
            score_full=None,
            score_cd=None
        )

    else:
        raise ValueError("method must be either 'bn' or 'pc'")

def plot_ibd_network(result, cohort="full", figsize=(8, 8), title=None):
    """
    Plot a network stored in an IBDNetworkResult.

    Parameters
    ----------
    result : IBDNetworkResult
        Dataclass returned by generate_ibd_network.
    cohort : str
        "full" or "cd"
    figsize : tuple
        Matplotlib figure size.
    title : str or None
        Optional custom title.
    """

    if cohort not in ["full", "cd"]:
        raise ValueError("cohort must be 'full' or 'cd'")

    g_obj = result.g_full if cohort == "full" else result.g_cd

    if g_obj is None:
        raise ValueError(f"No graph found for cohort='{cohort}'")

    if result.algorithm == "pc_kci":
        g, labels = pc_graph_to_igraph(g_obj)
    elif result.algorithm.startswith("hillclimb_bn"):
        if not isinstance(g_obj, Graph):
            raise TypeError("Expected igraph.Graph for BN result")
        g = g_obj
        labels = g.vs["name"]
    else:
        if isinstance(g_obj, Graph):
            g = g_obj
            labels = g.vs["name"]
        else:
            g, labels = pc_graph_to_igraph(g_obj)

    if title is None:
        cohort_label = "Full Cohort" if cohort == "full" else "Crohn's Disease"
        if result.algorithm.startswith("hillclimb_bn"):
            title = f"{cohort_label} Bayesian Network"
        elif result.algorithm == "pc_kci":
            title = f"{cohort_label} PC Graph"
        else:
            title = f"{cohort_label} Network"

    fig, ax = plt.subplots(figsize=figsize)

    layout = g.layout("fr")

    plot(
        g,
        target=ax,
        layout=layout,
        vertex_size=22,
        vertex_label=labels,
        vertex_label_size=10,
        vertex_label_dist=2,
        edge_arrow_size=0.8,
        edge_width=1.8,
        edge_curved=False
    )

    ax.set_title(title)
    plt.show()

def plot_centrality_comparison(
        pc_ntwrk, bic_ntwrk, bdeu_ntwrk, node_of_interest="CRP", analtype="betweenness", top_n=None
        ):
    """
    Generate a six-panel comparison of node centrality between PC and BN networks.

    Centrality values are computed using `calculate_betweenness()` and visualized
    using horizontal bar plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))

    panels = [
        ("PC — Full Cohort", pc_ntwrk, "full", axes[0, 0]),
        ("PC — Crohn's Cohort", pc_ntwrk, "cd", axes[1, 0]),
        ("BIC — Full Cohort", bic_ntwrk, "full", axes[0, 1]),
        ("BIC — Crohn's Cohort", bic_ntwrk, "cd", axes[1, 1]),
        ("Bdeu — Full Cohort", bdeu_ntwrk, "full", axes[0, 2]),
        ("Bdeu — Crohn's Cohort", bdeu_ntwrk, "cd", axes[1, 2])
    ]

    for title, result, cohort, ax in panels:
        ranked = calculate_betweenness(
            result,
            cohort=cohort,
            analtype=analtype,
            top_n=top_n
        )

        nodes = [r[0] for r in ranked]
        scores = [r[1] for r in ranked]

        colors = ['#d62728' if n == node_of_interest else '#1f77b4' for n in nodes]

        ax.barh(nodes[::-1], scores[::-1], color=colors[::-1])

        xlabel = "Betweenness Centrality" if analtype == "betweenness" else "Node Degree"

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"{analtype.capitalize()} Centrality Comparison (PC vs BIC vs Bdeu)", fontsize=15)

    legend_elements = [
        Patch(facecolor='#d62728', label=node_of_interest),
        Patch(facecolor='#1f77b4', label='Other nodes')
    ]
    fig.legend(
        handles=legend_elements,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )
    plt.tight_layout()
    plt.show()
    
# def plot_centrality(
#         result, cohort="full", node_of_interest="CRP", analtype="betweenness", top_n=None
#         ):
#     """
#     Single Plot
#     """
#     ranked = calculate_betweenness(
#         result, cohort=cohort,
#         analtype=analtype,
#         top_n=top_n
#     )
#     nodes = [r[0] for r in ranked]
#     scores = [r[1] for r in ranked]
#     colors = ['#d62728' if n == node_of_interest else '#1f77b4' for n in nodes]

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.barh(nodes[::-1], scores[::-1], color=colors[::-1])

#     xlabel = "Betweenness Centrality" if analtype == "betweenness" else "Node Degree"
#     cohort_label = "Full Cohort" if cohort == "full" else "Crohn's Disease"
#     title = f"{analtype.capitalize()} Centrality — {cohort_label}"

#     ax.set_xlabel(xlabel, fontsize=11)
#     ax.set_title(title, fontsize=13, fontweight='bold')
#     ax.tick_params(axis='y', labelsize=9)
#     legend_elements = [
#         Patch(facecolor='#d62728', label=node_of_interest),
#         Patch(facecolor='#1f77b4', label='Other nodes')
#     ]
#     ax.legend(handles=legend_elements, fontsize=8,loc='lower left')
#     plt.tight_layout()
#     plt.show()

def compare_hamming_distance(
        pc_ntwrk=False,
        bic_ntwrk=False,
        bdeu_ntwrk=False,
        cohort="full",
        n_boot=100,
        random_state=None,
        plot=True,
        heatmap=True
        ):
    """
    Bootstrap network stability analysis using Hamming distance.

    Parameters
    ----------
    pc_ntwrk, bic_ntwrk, bdeu_ntwrk : IBDNetworkResult or False
        Pre-fit network result objects to analyze. Any argument left as False is ignored.
    cohort : str
        "full" or "cd"
    n_boot : int
        Number of bootstrap replicates per network.
    random_state : int or None
        Seed for reproducibility.
    plot : bool
        If True, plot histogram(s) of bootstrap-to-original Hamming distance.
    heatmap : bool
        If True, plot edge-frequency heatmap(s).

    Returns
    -------
    dict
        Nested dictionary of bootstrap stability summaries for each supplied network.
    """
    rng = np.random.default_rng(random_state)

    named_networks = {
        "pc": pc_ntwrk,
        "bic": bic_ntwrk,
        "bdeu": bdeu_ntwrk
    }
    named_networks = {k: v for k, v in named_networks.items() if v is not False}

    if len(named_networks) < 1:
        raise ValueError("Not much of a comparison if you're only comparing <2 graphs!")

    if cohort not in ["full", "cd"]:
        raise ValueError("")

    results = {}

    for name, result in named_networks.items():

        if cohort == "full":
            fit_df = result.full_data.copy()
            ref_graph = result.g_full
        else:
            fit_df = result.cd_data.copy()
            ref_graph = result.g_cd

        if fit_df is None or ref_graph is None:
            raise ValueError(f"Missing data or graph for {name} / cohort={cohort}")

        if isinstance(ref_graph, Graph):
            node_order = ref_graph.vs["name"]
        else:
            try:
                _, labels = pc_graph_to_igraph(ref_graph)
                node_order = labels
            except Exception as e:
                raise TypeError(
                    f"Could not extract node names from graph for {name}: {e}"
                )

        node_to_idx = {node: i for i, node in enumerate(node_order)}

        ref_adj = graph_to_adj(ref_graph, node_order, node_to_idx)

        hamming_distances = []
        edge_counts = np.zeros_like(ref_adj, dtype=int)

        for _ in range(n_boot):
            sample_idx = rng.choice(fit_df.index, size=len(fit_df), replace=True)
            boot_df = fit_df.loc[sample_idx].reset_index(drop=True)

            # hard-coded rebuild logic matched to current codebase
            if result.algorithm == "pc_kci":
                boot_model = pc(
                    boot_df.to_numpy(),
                    alpha=0.05,
                    indep_test=kci,
                    show_progress=False,
                    node_names=list(boot_df.columns)
                )
                boot_graph = boot_model.G

            elif result.algorithm == "hillclimb_bn_bic":
                scorer = BIC(boot_df)
                hc = HillClimbSearch(boot_df)
                boot_model = hc.estimate(scoring_method=scorer, show_progress=False)
                boot_graph = Graph.TupleList(boot_model.edges(), directed=True)

            elif result.algorithm == "hillclimb_bn_bdeu":
                scorer = BDeu(boot_df)
                hc = HillClimbSearch(boot_df)
                boot_model = hc.estimate(scoring_method=scorer,show_progress=False)
                boot_graph = Graph.TupleList(boot_model.edges(), directed=True)

            else:
                raise ValueError(f"Unsupported algorithm: {result.algorithm}")

            boot_adj = graph_to_adj(boot_graph, node_order, node_to_idx)
            edge_counts += boot_adj

            d = int(np.sum(ref_adj != boot_adj))
            hamming_distances.append(d)

        edge_freq = edge_counts / n_boot
        edge_freq_df = pd.DataFrame(edge_freq, index=node_order, columns=node_order)

        results[name] = {
            "algorithm": result.algorithm,
            "cohort": cohort,
            "n_boot": n_boot,
            "distances": hamming_distances,
            "mean_hamming": float(np.mean(hamming_distances)),
            "std_hamming": float(np.std(hamming_distances)),
            "median_hamming": float(np.median(hamming_distances)),
            "min_hamming": int(np.min(hamming_distances)),
            "max_hamming": int(np.max(hamming_distances)),
            "edge_frequency": edge_freq_df
        }

    ordered_names = ["pc", "bic", "bdeu"]
    present = [n for n in ordered_names if n in results]
    if plot and len(present) > 0:

        fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5))

        if len(present) == 1:
            axes = [axes]

        for ax, name in zip(axes, present):
            distances = results[name]["distances"]

            ax.hist(distances, bins=min(20, max(5, len(set(distances)))))
            ax.set_title(f"{name.upper()} Stability")
            ax.set_xlabel("Hamming Distance")
            ax.set_ylabel("Bootstrap Count")

        plt.tight_layout()
        plt.show()


    if heatmap and len(present) > 0:

        fig = plt.figure(figsize=(8 * len(present) + 3, 8))

        gs = gridspec.GridSpec(
            1,
            len(present) + 1,
            width_ratios=[0.05] + [1] * len(present),
            wspace=0.8
        )

        cax = fig.add_subplot(gs[0])

        axes = [fig.add_subplot(gs[i + 1]) for i in range(len(present))]

        im = None
        for ax, name in zip(axes, present):
            edge_freq_df = results[name]["edge_frequency"]

            im = ax.imshow(
                edge_freq_df.values,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=1
            )

            ax.set_xticks(range(len(edge_freq_df.columns)))
            ax.set_xticklabels(edge_freq_df.columns, rotation=90)

            ax.set_yticks(range(len(edge_freq_df.index)))
            ax.set_yticklabels(edge_freq_df.index)

            ax.set_title(f"{name.upper()} Edge Stability", pad=14)

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Edge Inclusion Frequency", labelpad=14)

    plt.show()
    return results