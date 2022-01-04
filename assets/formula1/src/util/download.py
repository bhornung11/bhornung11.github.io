"""
Data download utility.
"""


from typing import Any, Dict

import requests


def download(token: str) -> Dict[str, Any]:
    """
    Downloads and prepares a track record (no pun).

    Parameters:
        token: str : json file name
    """

    # download
    url_base = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/"
    url = url_base + token
    resp = requests.get(url)
    content = resp.json()

    # process
    name = content["features"][0]["properties"]["Name"]
    length = content["features"][0]["properties"]["length"]
    line = content["features"][0]["geometry"]["coordinates"]

    # collate
    result = {
        "name": name,
        "length": length,
        "line": line
    }

    return result
