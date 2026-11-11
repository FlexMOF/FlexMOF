"""Permutation-invariant block identities and atom maps for repeated blocks."""

from functools import lru_cache

from .linker_data import graph_certificate


@lru_cache(maxsize=16384)
def block_certificate(numbers, edges, linker):
    return graph_certificate(numbers, edges) if linker else f"singleton:{numbers[0]}"


@lru_cache(maxsize=16384)
def occurrence_order(reference_numbers, reference_edges, numbers, edges):
    """Indices in this occurrence, in the representative graph's atom order.

    A certificate alone cannot map cell atoms: equivalent linkers may arrive
    in different orders. Use an element-preserving graph isomorphism and keep
    that mapping when scattering the shared latent back into the cell.
    """
    if (reference_numbers, reference_edges) == (numbers, edges):
        return tuple(range(len(numbers)))
    import networkx as nx

    def graph(elements, bonds):
        result = nx.Graph()
        result.add_nodes_from((i, {"element": z}) for i, z in enumerate(elements))
        result.add_edges_from(bonds)
        return result

    # VF2++ avoids the costly recursive search on repeated symmetric linkers.
    mapping = nx.vf2pp_isomorphism(
        graph(reference_numbers, reference_edges), graph(numbers, edges),
        node_label="element",
    )
    if mapping is None:
        raise ValueError("Equal block certificates did not yield a graph isomorphism")
    return tuple(mapping[i] for i in range(len(reference_numbers)))
