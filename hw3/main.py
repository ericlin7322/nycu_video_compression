import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s","--search", type=int, default=8)
args = parser.parse_args()

# Load two images (frames)
frame1 = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)

# Parameters
block_size = 8
search_range = args.search

# Create a list to store motion vectors
motion_vectors = []

# Iterate through the first frame in block-size steps
for y in range(0, frame1.shape[0], block_size):
    for x in range(0, frame1.shape[1], block_size):
        block1 = frame1[y:y+block_size, x:x+block_size]

        min_sad = float('inf')
        best_match = None

        # Search in the specified search range
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                y2 = y + dy
                x2 = x + dx

                # Ensure the search is within the frame bounds
                if y2 >= 0 and y2 + block_size <= frame2.shape[0] and x2 >= 0 and x2 + block_size <= frame2.shape[1]:
                    block2 = frame2[y2:y2+block_size, x2:x2+block_size]
                    sad = np.sum(np.abs(np.subtract(block1, block2)))

                    if sad < min_sad:
                        min_sad = sad
                        best_match = (dx, dy)

        motion_vectors.append(best_match)

# Perform motion compensation
reconstructed_frame = np.zeros_like(frame1)
for i, (dx, dy) in enumerate(motion_vectors):
    x = (i % (frame1.shape[1] // block_size)) * block_size
    y = (i // (frame1.shape[1] // block_size)) * block_size

    if dx is not None and dy is not None:
        block2 = frame2[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
        reconstructed_frame[y:y+block_size, x:x+block_size] = block2

# Calculate PSNR
mse = np.mean((frame1 - reconstructed_frame) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)

print(f'search range {args.search} PSNR: {psnr}')

# Save the reconstructed frame
cv2.imwrite(f'reconstructed_frame_{args.search}.png', reconstructed_frame)

# Display the original and reconstructed frames
# cv2.imshow('Original Frame', frame1)
# cv2.imshow('Reconstructed Frame', reconstructed_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()