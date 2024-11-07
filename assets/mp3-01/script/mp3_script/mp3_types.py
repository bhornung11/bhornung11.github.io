"""
Container types
"""

from collections import namedtuple
from typing import Dict


Song = namedtuple("Song", ["time_series", "sampling_rate", "n_sample"])

TypeAlbum = Dict[str, Song]

TypeAlbumCollection = Dict[str, TypeAlbum]
