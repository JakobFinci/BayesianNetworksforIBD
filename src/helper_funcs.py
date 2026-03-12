import pandas as pd
import numpy as np
from igraph import Graph

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def build_clean_dataset(
    input_path: str = "../data/labs_and_biopsies.csv",
    output_path: str = "../data/final_cleaned.csv",
    unique_cols: list[str] | None = None,
    unique_ordinals: list[str] | None = None,
    unique_label: str | None = None
):
    
    df_raw = pd.read_csv(input_path, sep=",")

    df_raw.replace({
        "Quiescent": "0",
        "Mild": "1",
        "Moderate": "2",
        "Severe": "3"
    }, inplace=True)

    if unique_cols:
        muhcols = unique_cols
    else:
        muhcols = [
            "Duo_Inflammation",
            "Gastric_Inflammation",
            "LeftColon_Inflammation",
            "TI_Inflammation",
            "PGA",
            "Hematocrit",
            "ESR",
            "CRP",
            "Albumin",
            "Vitamin D"
        ]

    df = df_raw[muhcols].apply(pd.to_numeric, errors="coerce")

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        max_iter=10,
        random_state=42
    )

    df_imputed = imputer.fit_transform(df)
    df_complete = pd.DataFrame(df_imputed, columns=muhcols)

    if unique_label:
        df_complete[unique_label] = df_raw[unique_label]
    else:
        df_complete["Diagnosis"] = df_raw["Diagnosis"]

    if unique_ordinals:
        categorical_cols = unique_ordinals
    else:
        categorical_cols = [
            "Duo_Inflammation",
            "Gastric_Inflammation",
            "LeftColon_Inflammation",
            "TI_Inflammation",
            "PGA"
        ]

    df_complete[categorical_cols] = df_complete[categorical_cols].round()
    df_complete[categorical_cols] = df_complete[categorical_cols].clip(0, 3)
    df_complete[categorical_cols] = df_complete[categorical_cols].astype(int)

    print(df_complete.dtypes)
    print(f"\nFinal node shape: {df_complete.shape}")
    print(f"Columns: {list(df_complete.columns)}")

    df_complete.to_csv(output_path, index=False)

    return df_complete

def pc_graph_to_igraph(pc_graph):

    if isinstance(pc_graph, Graph):
        return pc_graph, pc_graph.vs["name"]

    edges = []
    for edge in pc_graph.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()

        endpoint1 = edge.get_endpoint1().name
        endpoint2 = edge.get_endpoint2().name

        # TAIL -> ARROW means directed node1 -> node2
        # ARROW <- TAIL means directed node2 -> node1
        # otherwise approximate as bidirectional
        if endpoint1 == "TAIL" and endpoint2 == "ARROW":
            edges.append((node1, node2))
        elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
            edges.append((node2, node1))
        else:
            edges.append((node1, node2))
            edges.append((node2, node1))

    g = Graph.TupleList(edges, directed=True)
    return g, g.vs["name"]

def calculate_betweenness(result, cohort, analtype, top_n):

    if cohort not in ["full", "cd"]:
        raise ValueError("cohort must be 'full' or 'cd'")

    g = result.g_full if cohort == "full" else result.g_cd

    if result.algorithm == "pc_kci":
        g, _ = pc_graph_to_igraph(g)

    if analtype == "betweenness":
        bet = g.betweenness(directed=True)
    elif analtype == "degree":
        bet = g.degree(mode="all")
    else:
        raise ValueError("cohort must be 'betweenness' or 'degree'")

    ranked = sorted(
        zip(g.vs["name"], bet),
        key=lambda x: x[1],
        reverse=True
    )

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked

def graph_to_adj(g_obj, node_order, node_to_idx):
            A = np.zeros((len(node_order), len(node_order)), dtype=int)

            if isinstance(g_obj, Graph):
                edges = g_obj.get_edgelist()
                names = g_obj.vs["name"]
                for u_idx, v_idx in edges:
                    u = names[u_idx]
                    v = names[v_idx]
                    if u in node_to_idx and v in node_to_idx:
                        A[node_to_idx[u], node_to_idx[v]] = 1
            else:
                g_ig, labels = pc_graph_to_igraph(g_obj)
                edges = g_ig.get_edgelist()
                for u_idx, v_idx in edges:
                    u = labels[u_idx]
                    v = labels[v_idx]
                    if u in node_to_idx and v in node_to_idx:
                        A[node_to_idx[u], node_to_idx[v]] = 1

            return A