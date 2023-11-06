import cv2
import numpy as np

# Load two images (frames)
frame1 = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)

# Parameters
block_size = 8

# Create a list to store motion vectors
motion_vectors = []

# Define the TSS search pattern
search_pattern = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

# Iterate through the first frame in block-size steps
for y in range(0, frame1.shape[0], block_size):
    for x in range(0, frame1.shape[1], block_size):
        block1 = frame1[y:y+block_size, x:x+block_size]

        min_sad = float('inf')
        best_match = None

        # Apply TSS search pattern
        for dx, dy in search_pattern:
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

print(f'Three-Step Search PSNR: {psnr}')

# Save the reconstructed frame
cv2.imwrite(f'reconstructed_frame_tss.png', reconstructed_frame)