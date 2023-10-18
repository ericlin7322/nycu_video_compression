import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("lena_gray.png", image)

M, N = image.shape

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

dct_coeffs = np.zeros((M, N))

import time

start_time = time.time()
for u in tqdm(range(M)):
    for v in range(N):
        dct_coeffs[u,v] = dct2(image,u,v)
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

dct_coeffs2 = np.zeros((M, N))

start_time = time.time()
for u in tqdm(range(M)):
    for v in range(N):
        dct_coeffs2[u,v] = idct2(image,u,v)
end_time = time.time()
runtime_2d_dct = end_time - start_time

print(f"Runtime for 2D-iDCT: {runtime_2d_dct:.6f} seconds")

cv2.imwrite("lena_idct.png", dct_coeffs2)
