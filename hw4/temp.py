import cv2
import numpy as np

def run_length_encode(arr):
    rle = []
    count = 0

    for val in arr:
        if val == 0:
            count += 1
        else:
            rle.append((count, val))
            count = 0

    return rle

def run_length_decode(rle):
    decoded = []
    
    for count, value in rle:
        decoded.extend([0] * count)
        decoded.append(value)

    return np.array(decoded)

def block_dct_quantize(image_block, quant_dc=16, quant_ac=8):
    dct_block = cv2.dct(np.float32(image_block))
    
    # Quantization
    dct_block[0, 0] = np.round(dct_block[0, 0] / quant_dc)
    dct_block[1:, 1:] = np.round(dct_block[1:, 1:] / quant_ac)

    return dct_block

def block_idct_dequantize(dct_block, quant_dc=16, quant_ac=8):
    # Dequantization
    dct_block[0, 0] *= quant_dc
    dct_block[1:, 1:] *= quant_ac

    idct_block = cv2.idct(dct_block)
    
    return idct_block

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    encoded_blocks = []

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block = img[i:i+8, j:j+8]
            dct_quantized_block = block_dct_quantize(block)
            encoded_block = run_length_encode(dct_quantized_block.flatten())
            encoded_blocks.append(encoded_block)

    # Decode and reconstruct the image
    decoded_blocks = []

    for encoded_block in encoded_blocks:
        decoded_block = run_length_decode(encoded_block)
        decoded_block = block_idct_dequantize(decoded_block)
        decoded_blocks.append(decoded_block)

    decoded_image = np.zeros_like(img)

    index = 0
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            decoded_image[i:i+8, j:j+8] = decoded_blocks[index]
            index += 1

    return decoded_image.astype(np.uint8)

# Example usage
input_image_path = "lena.png"
output_image = process_image(input_image_path)
cv2.imshow("Original Image", cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE))
cv2.imshow("Reconstructed Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()