import cv2
import psnr
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

dct_coeffs = cv2.dct(np.float32(image))
reconstructed_image = cv2.idct(dct_coeffs)

cv2.imwrite("lena_cv2_dct.png", dct_coeffs)
cv2.imwrite("lena_cv2_idct.png", reconstructed_image)
psnr(image, reconstructed_image)