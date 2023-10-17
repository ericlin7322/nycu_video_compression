import cv2

im = cv2.imread("lena.png")

# cv2.imread() store in B, G, R order

R = im[:, :, 2]
G = im[:, :, 1]
B = im[:, :, 0]
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.169 * R - 0.331 * G + 0.5 * B + 128
V = 0.5 * R - 0.419 * G - 0.081 * B + 128
Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

cv2.imwrite("R.png", R)
cv2.imwrite("G.png", G)
cv2.imwrite("B.png", B)
cv2.imwrite("Y.png", Y)
cv2.imwrite("U.png", U)
cv2.imwrite("V.png", V)
cv2.imwrite("Cb.png", Cb)
cv2.imwrite("Cr.png", Cr)
