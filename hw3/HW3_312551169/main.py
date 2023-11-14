import numpy as np
import cv2
import time

def three_step_search_block_matching(reference_frame, current_frame, block_size):
    height, width = reference_frame.shape

    reconstructed_frame = np.zeros_like(current_frame)
    residual_frame = np.zeros_like(current_frame)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            min_mad = float('inf')
            best_match = (0, 0)

            for step in [16, 8, 4]:
                for m in range(-step, step + 1, step):
                    for n in range(-step, step + 1, step):
                        if i + m < 0 or i + m + block_size >= height:
                            continue
                        if j + n < 0 or j + n + block_size >= width:
                            continue

                        block_current = current_frame[i:i + block_size, j:j + block_size]
                        block_reference = reference_frame[i + m:i + m + block_size, j + n:j + n + block_size]

                        mad = np.sum(np.abs(block_current - block_reference))

                        if mad < min_mad:
                            min_mad = mad
                            best_match = (m, n)


            i_start, j_start = i + best_match[0], j + best_match[1]
            block_reference = reference_frame[i_start:i_start + block_size, j_start:j_start + block_size]
            reconstructed_frame[i:i + block_size, j:j + block_size] = block_reference
            residual_frame[i:i + block_size, j:j + block_size] = current_frame[i:i + block_size, j:j + block_size] - block_reference


    return reconstructed_frame, residual_frame


def full_search_block_matching(reference_frame, current_frame, block_size, search_range):
    height, width = reference_frame.shape

    reconstructed_frame = np.zeros_like(current_frame)
    residual_frame = np.zeros_like(current_frame)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            min_mad = float('inf')
            best_match = (0, 0)

            block_current = current_frame[i:i + block_size, j:j + block_size]

            for m in range(max(0, i - search_range), min(height - block_size, i + search_range + 1)):
                for n in range(max(0, j - search_range), min(width - block_size, j + search_range + 1)):
                    
                    block_reference = reference_frame[m:m + block_size, n:n + block_size]

                    mad = np.mean(np.abs(block_current - block_reference))
                    if mad < min_mad:
                        min_mad = mad
                        best_match = (m-i, n-j)

            print(f"{i} {j}: {best_match}")
            i_start, j_start = i + best_match[0], j + best_match[1]
            block_reference = reference_frame[i_start:i_start + block_size, j_start:j_start + block_size]
            reconstructed_frame[i:i + block_size, j:j + block_size] = block_reference
            residual_frame[i:i + block_size, j:j + block_size] = current_frame[i:i + block_size, j:j + block_size] - block_reference

    return reconstructed_frame, residual_frame


reference_frame = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)
current_frame = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)
block_size = 8

# start_time = time.time()
# reconstructed_frame_tss, residual_frame_tss = three_step_search_block_matching(reference_frame, current_frame, block_size)
# end_time = time.time()
# runtime = end_time - start_time
# print(f"Runtime for search range tss: {runtime:.6f} seconds")
# mse = np.mean((current_frame - reconstructed_frame_tss) ** 2)
# psnr = 10 * np.log10((255 ** 2) / mse)
# print(f'search range tts PSNR: {psnr}')

start_time = time.time()
reconstructed_frame_8, residual_frame_8 = full_search_block_matching(reference_frame, current_frame, block_size, 8)
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime for search range 8: {runtime:.6f} seconds")
mse = np.mean((current_frame - reconstructed_frame_8) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)
print(f'search range 8 PSNR: {psnr}')

# start_time = time.time()
# reconstructed_frame_16, residual_frame_16 = full_search_block_matching(reference_frame, current_frame, block_size, 16)
# end_time = time.time()
# runtime = end_time - start_time
# print(f"Runtime for search range 16: {runtime:.6f} seconds")
# mse = np.mean((current_frame - reconstructed_frame_16) ** 2)
# psnr = 10 * np.log10((255 ** 2) / mse)
# print(f'search range 16 PSNR: {psnr}')

# start_time = time.time()
# reconstructed_frame_32, residual_frame_32 = full_search_block_matching(reference_frame, current_frame, block_size, 32)
# end_time = time.time()
# runtime = end_time - start_time
# print(f"Runtime for search range 32: {runtime:.6f} seconds")
# mse = np.mean((current_frame - reconstructed_frame_32) ** 2)
# psnr = 10 * np.log10((255 ** 2) / mse)
# print(f'search range 32 PSNR: {psnr}')

cv2.imwrite(f'reconstructed_frame_tss.png', reconstructed_frame_tss)
cv2.imwrite(f'reconstructed_frame_8.png', reconstructed_frame_8)
cv2.imwrite(f'reconstructed_frame_8.png', reconstructed_frame_16)
cv2.imwrite(f'reconstructed_frame_8.png', reconstructed_frame_32)
cv2.imwrite(f'residual_frame_tss.png', residual_frame_tss)
cv2.imwrite(f'residual_frame_8.png', residual_frame_8)
cv2.imwrite(f'residual_frame_16.png', residual_frame_16)
cv2.imwrite(f'residual_frame_32.png', residual_frame_32)