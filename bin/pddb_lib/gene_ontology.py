import obonet
import networkx as nx

#exp, phylo, curated, comp, iea
#iea_not, comp_not, conditional_not, derived_not, curated_not, phylo_not, exp_not

ONTOLOGIES_SHORT = {
    "mf": "Molecular Function",
    "cc": "Cellular Component",
    "bp": "Biological Process",
}

EVIDENCE_GROUP_IMPORTANCE_SEQUENCE = [
    "exp",
    "exp_not",
    "phylo",
    "phylo_not",
    "curated",
    "curated_not",
    "derived_not",
    "conditional_not",
    "comp",
    "comp_not",
    "iea",
    "iea_not",
]

CWA_DATASET_NAME = "classic"
OWA_DATASET_NAME = "open_world_assumption"
CN_DATASET_NAME = "conditional_negatives"
SOFT_DATASET_NAME = "soft"

EVIDENCE_REP_STRATEGIES = {
    "classic": {
        "exp": 1.0,
        "phylo": 1.0,
        "curated": 1.0,
        "comp": None,
        "iea": None,
        "iea_not": None,
        "comp_not": None,
        "conditional_not": None,
        "curated_not": None,
        "derived_not": None,
        "phylo_not": None,
        "exp_not": None,
    },
    "open_world_assumption": {
        "exp": 1.0,
        "phylo": 1.0,
        "curated": 1.0,
        "comp": None,
        "iea": None,
        "iea_not": None,
        "comp_not": None,
        "conditional_not": None,
        "curated_not": 0.0,
        "derived_not": 0.0,
        "phylo_not": 0.0,
        "exp_not": 0.0,
    },
    "conditional_negatives": {
        "exp": 1.0,
        "phylo": 1.0,
        "curated": 1.0,
        "comp": None,
        "iea": None,
        "iea_not": None,
        "comp_not": None,
        "conditional_not": 0.0,
        "curated_not": 0.0,
        "derived_not": 0.0,
        "phylo_not": 0.0,
        "exp_not": 0.0,
    },
    "soft": {
        "exp": 1.0,
        "phylo": 0.9,
        "curated": 0.8,
        "comp": None,
        "iea": None,
        "iea_not": None,
        "comp_not": None,
        "conditional_not": 0.15,
        "curated_not": 0.01,
        "derived_not": 0.05,
        "phylo_not": 0.025,
        "exp_not": 0.0,
    },
}

def calc_normalized_y_pred(
    y_pred, go_names, parents_dict, children_dict, go_sortings, verbose=False
):
    n_samples, n_classes = y_pred.shape
    go_to_idx = {go_name: i for i, go_name in enumerate(go_names)}

    preorder = go_sortings["preorder"]
    postorder = go_sortings["postorder"]

    # MATRIZ 1: PROPAGAÇÃO BOTTOM-UP (Fórmula exata da Imagem)
    y_max_children = y_pred.copy()
    for go_id in postorder:
        if go_id not in go_to_idx:
            continue

        node_idx = go_to_idx[go_id]
        child_cols = [
            go_to_idx[c] for c in children_dict.get(go_id, []) if c in go_to_idx
        ]

        if child_cols:
            max_of_children = np.max(y_max_children[:, child_cols], axis=1)
            # A fórmula exata da imagem do CAFA:
            propagated_val = max_of_children * 0.7 + y_max_children[:, node_idx] * 0.3
            # A propagação bottom-up eleva o score se os filhos tiverem pontuação alta
            y_max_children[:, node_idx] = np.maximum(
                y_max_children[:, node_idx], propagated_val
            )

    # MATRIZ 2: PROPAGAÇÃO TOP-DOWN (O inverso da Imagem)
    y_min_parents = y_pred.copy()
    for go_id in preorder:
        if go_id not in go_to_idx:
            continue

        node_idx = go_to_idx[go_id]
        parent_cols = [
            go_to_idx[p] for p in parents_dict.get(go_id, []) if p in go_to_idx
        ]

        if parent_cols:
            min_of_parents = np.min(y_min_parents[:, parent_cols], axis=1)
            # A fórmula exata da imagem, mas para os pais:
            propagated_val = min_of_parents * 0.7 + y_min_parents[:, node_idx] * 0.3
            # A propagação top-down rebaixa o score se os pais tiverem pontuação baixa
            y_min_parents[:, node_idx] = np.minimum(
                y_min_parents[:, node_idx], propagated_val
            )

    # MATRIZ 3: O BLEND FINAL DO TEXTO DO PAPER
    # Média entre a Predição Raw, a Máxima Inferida (Filhos) e a Mínima Inferida (Pais)
    y_final = (y_pred + y_max_children + y_min_parents) / 3.0

    return y_final

def create_ontology_dictionaries(obo_path: str):
    # children_dict: key GO ID, value set of all GO IDs that are direct children of the key GO ID (direct connection, not a path)
    # parents_dict: key GO ID, value set of all GO IDs that are direct parents of the key GO ID (direct connection, not a path)
    # In the obonet graph, the edges are directed from the child to the parent (child -> parent). So operations are inverted
    go_graph = obonet.read_obo(obo_path)
    parents_dict = {go_id: set(go_graph.successors(go_id)) for go_id in go_graph.nodes}
    children_dict = {
        go_id: set(go_graph.predecessors(go_id)) for go_id in go_graph.nodes
    }

    return parents_dict, children_dict


def create_ontology_dictionaries_full0(obo_path: str):
    # children_dict: key GO ID, value set of all GO IDs that are direct children of the key GO ID (direct connection, not a path)
    # parents_dict: key GO ID, value set of all GO IDs that are direct parents of the key GO ID (direct connection, not a path)
    # In the obonet graph, the edges are directed from the child to the parent (child -> parent). So operations are inverted
    go_graph = obonet.read_obo(obo_path)
    parents_dict = {
        go_id: set(nx.descendants(go_graph, go_id)) for go_id in go_graph.nodes
    }
    children_dict = {
        go_id: set(nx.ancestors(go_graph, go_id)) for go_id in go_graph.nodes
    }

    # create_depth_first_order
    go_graph_inverted = go_graph.reverse()
    go_sortings = {}
    for ont_name, go_id in [
        ("MF", "GO:0003674"),
        ("CC", "GO:0005575"),
        ("BP", "GO:0008150"),
    ]:
        order = list(nx.dfs_preorder_nodes(go_graph_inverted, source=go_id))
        inv_order = list(nx.dfs_postorder_nodes(go_graph, source=go_id))

        go_sortings[ont_name] = {"preorder": order, "postorder": inv_order}

    return parents_dict, children_dict, go_sortings


def create_ontology_dictionaries_full(obo_path: str):
    go_graph = obonet.read_obo(obo_path)

    # descendants pega TODOS os ancestrais (pula nós que ficaram de fora dos targets)
    parents_dict = {
        go_id: set(nx.descendants(go_graph, go_id)) for go_id in go_graph.nodes
    }
    # ancestors pega TODOS os descendentes (pula nós que ficaram de fora dos targets)
    children_dict = {
        go_id: set(nx.ancestors(go_graph, go_id)) for go_id in go_graph.nodes
    }

    go_graph_inverted = go_graph.reverse()
    go_sortings = {}
    for ont_name, go_id in [
        ("MF", "GO:0003674"),
        ("CC", "GO:0005575"),
        ("BP", "GO:0008150"),
    ]:
        order = list(nx.dfs_preorder_nodes(go_graph_inverted, source=go_id))
        inv_order = list(nx.dfs_postorder_nodes(go_graph, source=go_id))

        go_sortings[ont_name] = {"preorder": order, "postorder": inv_order}

    return parents_dict, children_dict, go_sortings