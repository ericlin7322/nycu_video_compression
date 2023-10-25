import cv2
import psnr
import dct1
import dct2
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='multiprocess')
parser.add_argument("-j", type=int,default=1, help="choose how many core to multiprocess")
args = parser.parse_args()
j = args.j

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("lena_gray.png", image)

# two 1D-DCT to transform "lena.png" to DCT coefficients.
start_time = time.time()
dct1d_coeffs = dct1.dct(image)
end_time = time.time()
runtime_2d_dct = end_time - start_time
print(f"Runtime for 1D-DCT: {runtime_2d_dct:.6f} seconds")
dct1d_log = np.log(np.abs(dct1d_coeffs))
cv2.imshow("lena_dct_1d", dct1d_log)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("lena_dct_1d.png", dct1d_coeffs)

# 2D-DCT to transform "lena.png" to DCT coefficients.
start_time = time.time()
dct2d_coeffs = dct2.dct(image, j)
end_time = time.time()
runtime_2d_dct = end_time - start_time
print(f"Runtime for 2D-DCT: {runtime_2d_dct:.6f} seconds")
dct2d_log = np.log(np.abs(dct2d_coeffs))
cv2.imshow("lena_dct_2d", dct2d_log)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("lena_dct_2d.png", dct2d_coeffs)

# 2D-DCT's DCT coefficients reconstruct
start_time = time.time()
idct2d_coeffs = dct2.idct(dct2d_coeffs, j)
end_time = time.time()
runtime_2d_dct = end_time - start_time
print(f"Runtime for 2D-iDCT: {runtime_2d_dct:.6f} seconds")
psnr.psnr(image, idct2d_coeffs)
cv2.imwrite("lena_idct_2d.png", idct2d_coeffs)

# two 1D-DCT's DCT coefficients reconstruct
start_time = time.time()
idct1d_coeffs = dct2.idct(dct1d_coeffs, j)
end_time = time.time()
runtime_2d_dct = end_time - start_time
print(f"Runtime for 1D-iDCT: {runtime_2d_dct:.6f} seconds")
psnr.psnr(image, idct1d_coeffs)
cv2.imwrite("lena_idct_1d.png", idct1d_coeffs)
