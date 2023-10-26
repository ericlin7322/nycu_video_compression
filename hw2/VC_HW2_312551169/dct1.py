import numpy as np
from tqdm import tqdm

def dct1d(data):
    N, = data.shape
    x = np.arange(N)
    dct_result = np.zeros(N)

    for u in range(N):
        sum_val = np.sum(data * np.cos((2 * x + 1) * u * np.pi / (2 * N)))
        
        if u == 0:
            sum_val *= 1 / np.sqrt(2)
        else:
            sum_val *= 1

        dct_result[u] = np.sqrt(2 / N) * sum_val

    return dct_result

def dct(image):
    M, N = image.shape
    dct_result = np.zeros((M, N))

    for i in tqdm(range(M)):
        dct_result[i, :] = dct1d(image[i, :])

    for j in tqdm(range(N)):
        dct_result[:, j] = dct1d(dct_result[:, j])

    return dct_result

def idct1d(data):
    N, = data.shape
    u = np.arange(N)
    cu = np.ones(N)
    cu[0] = 1.0/np.sqrt(2)
    dct_result = np.zeros(N)

    for x in range(N):
        
        sum_val = np.sum(cu * data * np.cos((2 * x + 1) * u * np.pi / (2 * N)))

        dct_result[x] = np.sqrt(2 / N) * sum_val

    return dct_result

def idct(image):
    M, N = image.shape
    dct_result = np.zeros((M, N))

    for i in tqdm(range(M)):
        dct_result[i, :] = idct1d(image[i, :])

    for j in tqdm(range(N)):
        dct_result[:, j] = idct1d(dct_result[:, j])

    return dct_result


