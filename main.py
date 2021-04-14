import numpy as np
import librosa

from preprocessing import *
from similarity_analysis import *
from utils import downsampled_average


def main(file_path, title, excerpt_lengths: list, weights=None, show=True):
    """
    Perform computations for a specified audio file.
    :param file_path: Path of the audio file (string)
    :param title: Audio file title to display similarity matrix (string)
    :param excerpt_lengths: List of excerpt lengths
    :param weights: Vector of weights for optimal excerpt selection
    :param show: True to visualize and save similarity matrix, False otherwise
    """
    filename = file_path[5:-4]

    # Parametrize audio
    parameterized_audio, time_axis = parameterize_audio(file_path)

    # Compute similarity matrix
    similarity_matrix = compute_similarity(parameterized_audio)

    # Visualize similarity matrix
    if show:
        visualize_similarity(similarity_matrix, time_axis, filename, title)

    # Find best excerpt of a specified length
    for length in excerpt_lengths:
        excerpt_length = length  # Except length (in seconds)
        best_excerpt_start = find_best_excerpt(similarity_matrix, time_axis, excerpt_length, weights)
        # Convert in seconds and display
        best_excerpt_start_sec = time_axis[best_excerpt_start]
        print("Excerpt length: {} s\nBest excerpt: {} seconds - {} seconds".format(length, int(best_excerpt_start_sec),
                                                                                   int(best_excerpt_start_sec +
                                                                                       excerpt_length)))


if __name__ == '__main__':
    # Set parameters
    file_path = 'data/rolling_stones_satisfaction.mp3'
    title = "Rolling Stones - Satisfaction"
    excerpt_lengths = [10, 20, 30]

    # Compute weights if needed
    use_weights = True
    if use_weights:
        signal, _ = librosa.load(file_path)
        weights = downsampled_average(signal ** 2, 2048)
    else:
        weights = None

    # Perform computations
    main(file_path, title, excerpt_lengths, weights, show=False)

    end = True