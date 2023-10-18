import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("lena_gray.png", image)

def dct2(image, u, v):
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

def worker(uv_pair):
    u, v = uv_pair
    return u, v, dct2(image, u, v)

def temp(image):
    M, N = image.shape
    dct_result = np.zeros((M, N))

    num_processes = 20  # Adjust this based on the available CPU cores
    pool = Pool(num_processes)
    uv_pairs = [(u, v) for u in range(M) for v in range(N)]
    # results = pool.map(worker, uv_pairs)
    results = list(tqdm(pool.imap(worker, uv_pairs), total=len(uv_pairs)))
                   
    pool.close()
    pool.join()

    # Fill the dct_result array with the results
    for u, v, coeff in results:
        dct_result[u, v] = coeff

    return dct_result

# dct_coeffs = temp(image)

import time

start_time = time.time()
dct_coeffs = temp(image)
end_time = time.time()
runtime_2d_dct = end_time - start_time

print(f"Runtime for 2D-DCT: {runtime_2d_dct:.6f} seconds")

cv2.imwrite("lena_dct.png", dct_coeffs)

def idct2(image, x, y):
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

def worker2(uv_pair):
    u, v = uv_pair
    return u, v, idct2(dct_coeffs, u, v)

def temp2(image):
    M, N = image.shape
    dct_result = np.zeros((M, N))

    # Create a pool of processes
    num_processes = 1  # Adjust this based on the available CPU cores
    pool = Pool(num_processes)

    # Generate a list of (u, v) pairs
    uv_pairs = [(u, v) for u in range(M) for v in range(N)]

    # Calculate DCT coefficients in parallel
    # results = pool.map(worker2, uv_pairs)
    results = list(tqdm(pool.imap(worker2, uv_pairs), total=len(uv_pairs)))

    pool.close()
    pool.join()

    # Fill the dct_result array with the results
    for u, v, coeff in results:
        dct_result[u, v] = coeff

    return dct_result


dct_coeffs2 = temp2(dct_coeffs)

cv2.imwrite("lena_idct.png", dct_coeffs2)
