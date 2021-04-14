import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import parameterize_audio

__all__ = ['compute_similarity', 'visualize_similarity', 'find_best_excerpt']

def compute_similarity(x):
    """
    Compute similarity matrix of data x with cosine distance.
    :param x: data (n_samples x n_features)
    :return: Similarity matrix (n_samples x n_samples)
    """
    n_samples = x.shape[0]
    similarity_matrix = np.eye(n_samples)

    norms = np.linalg.norm(x, axis=1)  # Compute euclidean norms once for all
    for i in range(1, n_samples):
        for j in range(i):
            dist_i_j = np.vdot(x[i], x[j]) / (norms[i] * norms[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = dist_i_j

    return similarity_matrix


def visualize_similarity(similarity_matrix, time_axis, filename, title: str):
    """
    Visualize similarity matrix with a heat map.
    :param similarity_matrix: Similarity matrix (n_samples x n_samples)
    :param time_axis: Time axis (n_samples)
    :param filename: Plot file name (string)
    :param title: Title of the audio file (string)
    """
    # Display similarities
    plt.figure(figsize=(7, 6))
    plt.title(title)
    #plt.imshow(similarity_matrix, cmap='cool', interpolation='nearest')
    sns.heatmap(similarity_matrix)

    n_samples = time_axis.shape[0]
    no_labels = 7
    step = int(n_samples / (no_labels - 1))
    positions = np.arange(0, n_samples, step)
    labels = time_axis[::step].astype(int)

    plt.xticks(positions, labels)
    plt.yticks(positions, labels)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Time (seconds)")
    plt.savefig('plots/' + filename + '.png')
    plt.show()

def find_best_excerpt(similarity_matrix, time_axis, excerpt_length, weights=None):
    """
    Find the optimal excerpt of specified length, according to the Cooper's automatic summarization method.
    :param similarity_matrix: Similarity_matrix (n_samples, n_samples)
    :param time_axis: Time axis (n_samples)
    :param excerpt_length: Excerpt length, in seconds (int or float)
    :param weights: A set of weights to compute the similarity score (n_samples) (None by default)
    :return: Start index of the optimal excerpt
    """
    n_samples = similarity_matrix.shape[0]
    time_step = time_axis[1] - time_axis[0]
    excerpt_length = int(excerpt_length / time_step)  # Convert excerpt length from seconds to number of indices
    similarity_scores = np.zeros(n_samples - excerpt_length)

    # Weights are ones by default (i.e. similarity score is unweighted)
    if weights is None:
        weights = np.ones(n_samples)

    for start in range(n_samples - excerpt_length):
        score = (similarity_matrix[start: start + excerpt_length] @ weights).sum()  # Compute weighted score
        similarity_scores[start] = score / (n_samples * excerpt_length)  # Normalize score

    best_excerpt_start = similarity_scores.argmax()

    return best_excerpt_start

if __name__ == '__main__':
    # Path to audio file
    file_path = 'data/metallica_fade_to_black.mp3'
    filename = file_path[5:-4]

    # Parametrize audio
    parameterized_audio, time_axis = parameterize_audio(file_path)

    # Compute similarity matrix
    similarity_matrix = compute_similarity(parameterized_audio)

    # Visualize similarity matrix
    title = "Metallica - Fade to Black"
    visualize_similarity(similarity_matrix, time_axis, filename, title)

    # Find best excerpt of a specified length
    excerpt_length = 10  # Except length (in seconds)
    best_excerpt_start = find_best_excerpt(similarity_matrix, time_axis, excerpt_length)

    # Convert in seconds and display
    best_excerpt_start_sec = time_axis[best_excerpt_start]
    print("Best excerpt: {} seconds - {} seconds".format(int(best_excerpt_start_sec),
                                                         int(best_excerpt_start_sec + excerpt_length)))

    end = True