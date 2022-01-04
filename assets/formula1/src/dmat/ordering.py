"""
Vertex ordering utilities.
"""

import numpy as np


def order_by_preceeding_distances(dmat: np.ndarray) -> np.ndarray:
    """
    Parameters:
        dmat: np.ndarray : distance matrix
        
    Returns:
        idcs: np.ndarray : index order
    """
    
    # start with the most central vertex i.e first in minimum vertex order
    i = np.argsort(np.sum(dmat, axis=1))[0]
    idcs = [i]

    # initialise list of unseen indices
    i_unprocessed = list(range(len(dmat)))
    i_unprocessed.remove(idcs[-1])
    
    while len(i_unprocessed) > 0:
        i = np.argmin(dmat[idcs].sum(axis=0)[i_unprocessed])
        i = i_unprocessed[i]
        idcs.append(i)
        i_unprocessed.remove(i)
    
    return idcs