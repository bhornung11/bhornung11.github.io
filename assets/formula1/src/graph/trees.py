"""
Simple tree algorithms.
"""

from collections import deque
from itertools import chain
from typing import Dict, List


import numpy as np


def find_minimum_spanning_tree(
        dmat: np.ndarray
    ) -> Dict[int, List[int]]:
    """
    Prim's algorithm.

    Parameters:
        dmat: np.ndarray : distance matrix

    Returns:
        tree: Dict[int, List[int]] : tree
    """

    tree = {}
    n = len(dmat)

    # this is really inefficient, but for the sake of comparison
    # we start with a vertex of minimum minimm distance
    i = np.argsort(np.sum(dmat, axis=1))[0]
    in_tree = [i]
    not_in_tree = list(range(0, n))
    not_in_tree.remove(i)

    while len(not_in_tree) != 0:
        dists = dmat[in_tree][:, not_in_tree]
        j, i = np.unravel_index(np.argmin(dists), dists.shape)

        # source in the tree
        j = in_tree[j]

        # target outside of the tree
        i = not_in_tree[i]

        # bookkeeping
        in_tree.append(i)
        not_in_tree.remove(i)

        # build a graph
        if j in tree:
            tree[j].append(i)
        else:
            tree.update({j: [i]})

    return tree


def tree_bfs(
        tree: Dict[int, List[int]]
    ) -> List[int]:
    """
    Breadth first search of a tree.

    Parameters:
        tree: Dict[int, List[int]] : tree

    Returns:
        idcs: List[int] : BFS indices
    """

    # collect all parents and children
    unvisited = set(tree.keys()).union(
        set(
            chain.from_iterable(
                tree.values()
            )
        )
    )

    queue = deque()
    u = next(iter(tree.keys()))
    idcs = [u]
    queue.append(u)
    unvisited.remove(u)

    while len(unvisited) != 0:

        u = queue.popleft()
        
        if u not in tree:
            continue
        for v in tree[u]:
            if v in unvisited:
                    
                queue.append(v)
                idcs.append(v)
                unvisited.remove(v)
                
    return idcs
