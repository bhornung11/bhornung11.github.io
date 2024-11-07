"""
En masse transform utility
"""

from typing import (
    Any,
    Callable,
    Dict
)

from mp3_script.mp3_types import (
    TypeAlbumCollection
)

def transfrom_albums(
        albums: TypeAlbumCollection,
        func: Callable,
        kwargs: Dict[str, Any]
    ) -> TypeAlbumCollection:
    """
    Convenience function to transform songs in albums en masse.

    Parameters:
        albums: TypeAlbumCollection : dict of songs dicts per album
        func: Callable : tranform function
            signature: func(song: Song, **kwargs)
        kwargs: Dict[str, Any] : transform function keyword arguments
        
    Returns:
        transformed: TypeAlbumCollection : albums with transformed song
    """

    transformed = {}

    for name_album, songs in albums.items():
        
        transformed[name_album] = {}
        
        for name_song, song in songs.items():

            transformed[name_album][name_song] = func(song, **kwargs)

    return transformed
