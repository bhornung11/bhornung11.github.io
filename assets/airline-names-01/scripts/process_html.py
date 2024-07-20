"""
Ad hoc html processing.
"""

from typing import (
    Set
)

from lxml.html import parse

def extract_names_from_html(
        path: str
    ) -> Set[str]:
    """
    Ad hoc function to extract airline names from the saved
    html source.
    
    Parameters:
        path: str : path to the html file
        
    Returns:
        strings: Set[str] : airline names
    """

    tree = parse(path)
    ll = tree.xpath('.//a[contains(text(),"/data/airlines")]')

    strings = set()

    for l in ll:
        try:
            string = l.getnext().getnext().text.lower()
            strings.add(string)
        except:
            pass

    return strings

