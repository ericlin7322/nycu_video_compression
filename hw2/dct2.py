import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def dct2d(image, u, v):
    M, N = image.shape
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    cu = 1.0
    if u == 0:
        cu = 1.0/np.sqrt(2)

    cv = 1.0
    if v == 0:
        cv = 1.0/np.sqrt(2)

    dct_result = 2.0 / N * cu * cv * np.sum(image * np.cos((2 * X + 1) * u * np.pi / (2 * M)) * np.cos((2 * Y + 1) * v * np.pi / (2 * N)))

    return dct_result

def dct_worker(uv_pair):
    u, v, image = uv_pair
    return u, v, dct2d(image, u, v)

def dct(image, j):
    M, N = image.shape
    dct_result = np.zeros((M, N))

    num_processes = j  # Adjust this based on the available CPU cores
    pool = Pool(num_processes)
    uv_pairs = [(u, v, image) for u in range(M) for v in range(N)]
    results = list(tqdm(pool.imap(dct_worker, uv_pairs), total=len(uv_pairs)))
                   
    pool.close()
    pool.join()

    for u, v, coeff in results:
        dct_result[u, v] = coeff

    return dct_result

def idct2d(image, x, y):
    M, N = image.shape
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(u, v, indexing='ij')

    cu = np.ones((1,M))
    cu[0,0] = 1.0/np.sqrt(2)

    cv = np.ones((1,M))
    cv[0,0] = 1.0/np.sqrt(2)

    dct_result = 2.0 / N * np.sum(cu * cv * image * np.cos((2 * x + 1) * U * np.pi / (2 * M)) * np.cos((2 * y + 1) * V * np.pi / (2 * N)))

    return dct_result 

def idct_worker(uv_pair):
    u, v, image = uv_pair
    return u, v, idct2d(image, u, v)

def idct(image, j):
    M, N = image.shape
    dct_result = np.zeros((M, N))

    num_processes = j  # Adjust this based on the available CPU cores
    pool = Pool(num_processes)

    uv_pairs = [(u, v,image) for u in range(M) for v in range(N)]

    results = list(tqdm(pool.imap(idct_worker, uv_pairs), total=len(uv_pairs)))

    pool.close()
    pool.join()

    for u, v, coeff in results:
        dct_result[u, v] = coeff

    return dct_result