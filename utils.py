import numpy as np

def downsampled_average(arr, n):
    end = n * int(len(arr)/n)
    downsampled_avg = np.mean(arr[:end].reshape(-1, n), 1)
    end_sample_avg = np.array([arr[end:].mean()])
    return np.concatenate([downsampled_avg, end_sample_avg])