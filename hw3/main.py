import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-s","--search", type=int, default=8)
args = parser.parse_args()

frame1 = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)

block_size = 8
search_range = args.search

motion_vectors = []

start_time = time.time()

for y in range(0, frame1.shape[0], block_size):
    for x in range(0, frame1.shape[1], block_size):
        block1 = frame1[y:y+block_size, x:x+block_size]

        min_mad = float('inf')
        best_match = None

        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                y2 = y + dy
                x2 = x + dx

                if y2 >= 0 and y2 + block_size <= frame2.shape[0] and x2 >= 0 and x2 + block_size <= frame2.shape[1]:
                    block2 = frame2[y2:y2+block_size, x2:x2+block_size]
                    mad = np.mean(np.abs(np.subtract(block1, block2)))

                    if mad < min_mad:
                        min_mad = mad
                        best_match = (dx, dy)

        motion_vectors.append(best_match)

reconstructed_frame = np.zeros_like(frame2)
for i, (dx, dy) in enumerate(motion_vectors):
    x = (i % (frame1.shape[1] // block_size)) * block_size
    y = (i // (frame1.shape[1] // block_size)) * block_size

    if dx is not None and dy is not None:
        block2 = frame2[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
        reconstructed_frame[y:y+block_size, x:x+block_size] = block2

residual_frame = cv2.absdiff(frame2, reconstructed_frame)
end_time = time.time()
runtime_2d_dct = end_time - start_time
print(f"Runtime for search range {args.search}: {runtime_2d_dct:.6f} seconds")

mse = np.mean((frame1 - reconstructed_frame) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)

print(f'search range {args.search} PSNR: {psnr}')

cv2.imwrite(f'reconstructed_frame_{args.search}.png', reconstructed_frame)
cv2.imwrite(f'residual_frame_{args.search}.png', residual_frame)