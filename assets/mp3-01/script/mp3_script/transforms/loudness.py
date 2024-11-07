"""
Loudness utilities.
"""


import numpy as np

import waveform_analysis

from mp3_script.mp3_types import Song


def convert_amplitude_to_spl_w(
        song: Song,
        min_amplitude: float = 1e-4,
        max_amplitude: float=1.0,
        ref_amplitude: float=1.0
    ) -> Song:
    """
    Wrapped amplitude to sound pressure level converter.

    Parameters:
        song: Song : song!
        min_amplitude: float = 1e-4 : minimum amplitude
        max_aplitude: float=1.0 : maximum amplitude
        ref_amplitude: float=1.0 : reference amplitude

    Returns:
        tranformed: Song : song with sound pressure level time series
    """

    spl = convert_amplitude_to_spl(
        song.time_series, min_amplitude, max_amplitude, ref_amplitude
    )

    transformed = Song(spl, song.sampling_rate, song.n_sample)

    return transformed


def convert_amplitude_to_spl(
        x: np.ndarray,
        min_amplitude: float = 1e-4,
        max_amplitude: float=1.0,
        ref_amplitude: float=1.0
    ) -> np.ndarray:
    """
    Converts the waveform amplitude to sound pressure level.

    Parameters:
        x: np.ndarray : waveform
        min_amplitude: float = 1e-4 : minimum amplitude
        max_aplitude: float=1.0 : maximum amplitude
        ref_amplitude: float=1.0 : reference amplitude

    Returns:
        spl: np.ndarray : sound pressure level
    """

    clipped = np.abs(x)
    clipped = np.where(clipped < min_amplitude, min_amplitude, clipped)
    clipped = np.where(clipped > max_amplitude, max_amplitude, clipped)

    spl = 20 * np.log10(clipped) - 20 * np.log10(ref_amplitude)

    return spl


def convert_spl_to_loudness_power_two_w(
        song: Song,
        max_spl: float=100,
        factor: float=10,
    ) -> Song:
    """
    Wrapped sound pressure level to perceived loudness converter.

    Parameters:
        song: Song : song!
        max_spl: float : maximum sound pressure level
        factor: float : every `factor` SPL increase doubles
            the perceived loudness

    Returns:
        transformed: Song : song with perceived loudness time series
    """

    loudness = convert_spl_to_loudness_power_two(
        song.time_series, max_spl, factor
    )

    transformed = Song(
        loudness, song.sampling_rate, song.n_sample
    )

    return transformed


def convert_spl_to_loudness_power_two(
        x: np.ndarray,
        max_spl: float=80,
        factor: float=10,
    ) -> np.ndarray:
    """
    Converts the sound pressure level to perceived loudness.
    Power two rule.

    Parameters:
        x: np.ndarray : sound pressure level
        max_spl: float : maximum sound pressure level
        factor: float : every `factor` SPL increase doubles
            the perceived loudness
    
    Returns:
        loudness: np.ndarray : perceived loudness
    """

    coeff = 100 / np.power(2, max_spl / factor)

    loudness = coeff * np.power(2, x / factor)

    return loudness


def transform_a_weight_w(
    song: Song
    ) -> Song:
    """
    Wrapper to A-weight a waveform.

    Parameters:
        song: song : Song!

    Returns:
        transformed: Song : A-weighted waveform
    """

    weighted = waveform_analysis.A_weight(
        song.time_series, song.sampling_rate
    )

    transformed = Song(
        weighted, song.sampling_rate, song.n_sample
    )

    return transformed
