"""
En masse transform utility
"""

import json

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable
)

from mp3_script.mp3_types import (
    AlbumEntry,
    TypeAlbumCollection
)

from mp3_script.mp3_io import (
    make_gen_album_entries
)
from mp3_script.transforms.loudness import (
    transform_a_weight_w,
    convert_spl_to_loudness_power_two_w,
    convert_amplitude_to_spl_w
)
from mp3_script.subsampling import (
    smooth_subsample_w
)
from mp3_script.transforms.legenre import (
    expand_legendre_w
)
from mp3_script.trim import (
    trim_series_w
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


def make_gen_transform(
        entries: Iterable[AlbumEntry],
        func: Callable,
        kwargs: Dict[str, Any]
    ) -> Generator:
    """
    Convenience function to transform songs in albums en masse.

    Parameters:
        albums: TypeAlbumCollection : dict of songs dicts per album
        func: Callable : tranform function
            signature: func(song: Song, **kwargs)
        kwargs: Dict[str, Any] : transform function keyword arguments

    Returns:
        inner: Generator : generator of transformed entries
    """

    def inner():
        """
        Tranformed entry generator.

        Parameters:
            None

        Yields:
            transformed: AlbumEntry : transformed entry
        """
        for entry in entries:
            song = func(entry.song, **kwargs)

            transformed = AlbumEntry(entry.name_album, entry.name_song, song)
            yield transformed

    return inner()


def transform_chain(
        dir_artist: str,
        return_stage: str,
        kwargs_spl: Dict[str, Any]={"min_amplitude": 1e-4, "max_amplitude": 1.0, "ref_amplitude": 1e-4},
        kwargs_loudness: Dict[str, Any]={"max_spl": 80, "factor": 10},
        kwargs_lhat: Dict[str, Any]={"spacing": 0.1, "width": 0.05},
        kwargs_trim: Dict[str, Any]={"left": 3, "right": 3},
        kwargs_coeff: Dict[str, Any]={"order": 50},
    ) -> Generator:
    """
    Reads mp3 files from albums and encodes them as loudnes time series in terms of
    Legendre expansion coefficients.

    Parameters:
        dir_artist: str : full path to the folder containing the album folders
        return_stage: str : which stage to return

    Returns:
        gen_coeff : album entries with expansion coefficients
    """

    gen_entries = make_gen_album_entries(dir_artist)

    if return_stage == "wave":
        return gen_entries

    gen_aw = make_gen_transform(
        gen_entries, transform_a_weight_w, {}
    )

    if return_stage == "aw":
        return gen_aw

    gen_spl = make_gen_transform(
        gen_aw, convert_amplitude_to_spl_w, kwargs_spl
    )

    if return_stage == "spl":
        return gen_spl

    gen_loudness = make_gen_transform(
        gen_spl, convert_spl_to_loudness_power_two_w, kwargs_loudness
    )

    if return_stage == "loudness":
        return gen_loudness

    gen_lhat = make_gen_transform(
        gen_loudness, smooth_subsample_w, kwargs_lhat
    )

    if return_stage == "lhat":
        return gen_lhat

    gen_trimmed = make_gen_transform(
        gen_lhat, trim_series_w, kwargs_trim
    )

    if return_stage == "trim":
        return gen_trimmed

    if return_stage == "coeff":

        gen_coeff = make_gen_transform(
            gen_trimmed, expand_legendre_w, kwargs_coeff
        )

        return gen_coeff

    raise ValueError(f"Unrecognised stage {return_stage}")
