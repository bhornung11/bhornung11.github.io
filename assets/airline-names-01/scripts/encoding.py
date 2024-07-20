"""
Encoding utlities.
"""

from typing import (
    Dict,
    List,
    Sequence,
    Set,
    Tuple
)

import numpy as np


def encode_name_by_token(
        string: str,
        code_groups: Dict[int, Set[str]],
        code_default: str = "S"
    ) -> Tuple[str]:
    """
    Encodes the words in a string by matching them against
    groups of words.

    Parameters:
        string: str : string to match
        code_groups: Dict[str, Set[str]] : code -- groups of words to match
        code_default: int = "S" : code for non-matching words
        
    Returns:
        encoded: tuple[str] : encoded string
    """

    encoded = []

    for word in string.split():
        for code, group in code_groups.items():
            if word in group:
                encoded.append(code)
                break
        else:
            encoded.append(code_default)

    encoded =tuple(encoded)

    return encoded


def encode_name_by_token_marker(
        name: str,
        tokens: Sequence[str],
        markers: Sequence[str]
    ) -> Tuple[str]:
    """
    Encodes a name using tokens and markers. Token is an entire word.
    A fragment is an incomplete word.

    Parameters:
        name: str,
        tokens: Sequence[str],
        markers: Sequence[str]
    """

    encoded = []

    for word in name.split():

        if word in tokens or word in markers:
            encoded.append("T")
            continue

        if _starts_with_from_set(word, markers):
            encoded.append("t")

            if _ends_with_from_set(word, markers):
                encoded.append("t")
                continue

            encoded.append("s")
            continue

        if _ends_with_from_set(word, markers):
            encoded.append("s")
            encoded.append("t")
            continue

        encoded.append("S")

    encoded = tuple(encoded)

    return encoded

def _ends_with_from_set(string, strings):

    for str_ in strings:
        if string.endswith(str_):
            return True
    return False

def _starts_with_from_set(string, strings):
    
    for str_ in strings:
        if string.startswith(str_):
            return True
    return False



def translate_to_colours(
        encoded: Tuple[str],
        mapping: Dict[int, np.ndarray]
    ) -> List[np.ndarray]:
    """
    Translates a list of codes to colours.

    Parameters:
        encoded: Tuple[str] : codes
        mapping: Dict[int, np.ndarray] : code to colour mapping       

    Returns:
        translated: List[np.ndarray] : codes represented as colours
    """

    translated = [
        mapping[code] for code in encoded
    ]

    return translated

