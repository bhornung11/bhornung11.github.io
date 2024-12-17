"""
Container types
"""

from collections import namedtuple
from typing import (
    Any,
    Dict
)


Song = namedtuple("Song", ["time_series", "sampling_rate", "n_sample"])

TypeAlbum = Dict[str, Song]

TypeAlbumCollection = Dict[str, TypeAlbum]

AlbumEntry = namedtuple(
    "AlbumEntry", ["name_album", "name_song", "song"]
)


def serialise_album_entry(entry: AlbumEntry) -> Dict[str, Any]:
    """
    Converts an `AlbumEntry` object to a dict.

    Parameters:
        entry: AlbumEntry : album entry!

    Returns:
        serialised: Dict[str, Any] : entry as a dict
            * `name_album`: str  : album name
            * `name_song`: str : song name
            * `song`: Song : song
    """

    serialised = {
        "name_album": entry.name_album,
        "name_song": entry.name_song,
        "song": serialise_song(entry.song)
    }

    return serialised


def serialise_song(song: Song) -> Dict[str, Any]:
    """
    Converts a `Song` object to a dict.

    Parameters:
        song: Song : song!

    Returns:
        serialised: Dict[str, Any] : song as a dict.
            `time_series`: List[float] : time series or expansion coeffs
            `sampling_rate`: int : sampling rate
            `n_sample`: int : number of samples or expansion coefficients
    """

    serialised =  {
        "time_series": song.time_series.tolist(),
        "sampling_rate": song.sampling_rate,
        "n_sample": song.n_sample
    }

    return serialised
