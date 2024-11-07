"""
MP3 I/O convenience functions.
"""

import os.path
from typing import (
    List
)

import librosa
import soundfile as sf

from mp3_script.mp3_types import (
    Song,
    TypeAlbumCollection
    
)


def split_album_to_songs(dir_artist: str, name_album: str) -> None:
    """
    Splits an album mp3 file to song mp3 files.

    Parameters:
        dir_artist: str : folder in which the aldum mp3 is deposited
        name_album: str : name of the album

    Returns:
        None : writes the songs to separate files
    """

    path_album_file = os.path.join(
        dir_artist, name_album + ".mp3"
    )

    waveform, sampling_rate = librosa.load(path_album_file)

    intervals = librosa.effects.split(waveform)

    i = 0
    for i1, i2 in intervals:
        
        # if the song is shorter than a minute ignore it
        length = (i2 - i1) / sampling_rate / 60
        if length < 1:
            continue

        i += 1
        fname = f"{name_album}-{i:02d}.mp3"
        path = os.path.join(dir_artist, name_album, fname)

        sf.write(path, y[i1:i2], sampling_rate)
        

def load_albums(
        dir_artist: str
    ) -> TypeAlbumCollection:
    """
    Loads all albums in a folder. An album is a subfolder with mp3 files.

    Parameters:
        dir_artist: str: full path to folder containing the albums

    Returns:
        albums: TypeAlbumCollection : songs as time series grouped by albums
    """

    albums = {}

    folders = select_make_folder_paths(dir_artist)

    for folder in folders:
        files = select_make_file_paths(folder)
        name_album = os.path.split(folder)[-1]

        # load all files
        albums[name_album] = {}

        for file in files:
            
            name_song = os.path.split(file)[-1].split(".")[0]

            waveform, sampling_rate = librosa.load(file)
            albums[name_album][name_song] = Song(
                waveform, sampling_rate, len(waveform)
            )

    return albums


def select_make_folder_paths(
        folder: str
    ) -> List[str]:
    """
    Returns the full paths to the immediate folder in a folder.
    
    Parameters:
        folder: str : full path to folder to search in
        
    Returns:
        paths_folder: List[str] : full paths to the immediate folders
    """
    
    paths = (
        os.path.join(folder, file) for file in os.listdir(folder)
    )
    paths_folder = sorted(path for path in paths if os.path.isdir(path))
    
    return paths_folder


def select_make_file_paths(
        folder: str
    ) -> List[str]:
    """
    Returns the full paths to the immediate proper files in a folder.

    Parameters:
        folder: str : full path to folder to search in

    Returns:
        paths_folder: List[str] : full paths to the immediate files
    """

    paths = (
        os.path.join(folder, file) for file in os.listdir(folder)
    )
    paths_file = sorted(path for path in paths if os.path.isfile(path))

    return paths_file
