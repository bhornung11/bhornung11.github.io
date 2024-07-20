"""
Radix tree utilities.
"""

from typing import (
    Any,
    Dict,
    Iterable
)

import networkx as nx


def grow_forest_nx(strings: Iterable[str]):
    """
    Grows a forest of radix tree from strings.

    Parameters:
        strings: Iterable[str] : strings

    Returns:
        forest: Dict[str, nx.Digraph] : forest keyed by root character
        bookkeep: Dict[int, Dict[str, Any] : edgewise forest    
    """
    
    i = 0
    bookkeep = {i: {"char": "$", "parent": None, "count": 0}}
    forest = {"children": {}, "id": i, "char": "$"}
    
    for string in strings:
        i = add_string_to_forest_nx(forest, string, i, bookkeep)
        
    return forest, bookkeep


def add_string_to_forest_nx(vertex, string: str, i: int, bookkeep):
    """
    
    """

    for char in string:
        
        # already in the tree
        if char in vertex["children"]:
            
            # increment counter
            i_child = vertex["children"][char]["id"]
            bookkeep[i_child]["count"] += 1
            
            vertex = vertex["children"][char]
        
        # create a new vertex
        else:
            
            # increment id
            i += 1
            
            # add it to the tree
            vertex["children"][char] = {"id": i, "children": {}}
            
            # add to bookkeeping
            bookkeep[i] = {"parent": vertex["id"], "count": 1, "char": char}
            
            # move down
            vertex = vertex["children"][char]

    return i


def make_subgraphs(bookkeep: Dict[str, Dict[str, Any]]):
    """
    Partitions a child--parent dict graph to 
    parent-->child directed trees.
    
    Parameters:
        bookkeep: Dict[str, Dict[str, Any]] : condensed graph
        
    """

    graph = nx.DiGraph()
    
    # remove root
    del bookkeep[0]
    
    # add vertices
    for i, props in bookkeep.items():
        label = f"{props['char']} ({props['count']})"
        graph.add_node(i, char=props["char"], label=label, count=props["count"])

    # add edges
    for i, props in bookkeep.items():
            
        # parent index
        i_parent = props["parent"]
        
        # if the parent is the root
        if i_parent == 0:
            # mark child as root of its tree
            props["parent"] = None
            
            # do not add edge
            continue

        # add all other edges
        graph.add_edge(i_parent, i, weight=props["count"])
    
    # split to connected components
    forest = [
        graph.subgraph(vertices).copy()
        for vertices in nx.weakly_connected_components(graph)
    ]

    # make it a dict
    forest_dict = {}
    for tree in forest:
        i_root = _find_root_vertex(tree)
        char_root = tree.nodes[i_root]["char"]
        forest_dict[char_root] = {"i_root": i_root, "tree": tree}
    
    return forest_dict

    
def _find_root_vertex(graph: nx.Graph) -> int:
    """
    Finds the root vertex.
    
    Parameters:
        graph: nx.Graph : tree

    Returns:
        i_root: int : index of the root vertex
    """
    
    for vertex in graph.nodes:
        if graph.in_degree(vertex) == 0:
            return vertex
        
    raise ValueError("No root has been found.")

