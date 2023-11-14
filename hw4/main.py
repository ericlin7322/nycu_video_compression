import numpy as np
import cv2

def RLE_encoding(img, bits=8):
    encoded = []
    count = 0
    prev = None
    
    rows, cols = img.shape
    fimg = []

    for i in range(rows + cols - 1):
        if i % 2 == 0:
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                fimg.append(img[j, i - j])
        else:
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                fimg.append(img[j, i - j])

    for pixel in fimg:
        if prev==None:
            prev = pixel
            count+=1
        else:
            if prev!=pixel:
                encoded.append((count, prev))
                prev=pixel
                count=1
            else:
                if count<(2**bits)-1:
                    count+=1
                else:
                    encoded.append((count, prev))
                    prev=pixel
                    count=1
    encoded.append((count, prev))
    return np.array(encoded)

def zigzag_to_2d_custom(zigzag_array, shape):
    rows, cols = shape
    result_matrix = np.zeros((rows, cols))
    i, j = 0, 0
    for k in range(len(zigzag_array)):
        result_matrix[i, j] = zigzag_array[k]
        if (i + j) % 2 == 0:
            if j == cols - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                j += 1
                i -= 1
        else:
            if i == rows - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i += 1
                j -= 1

    return result_matrix

def RLE_decode(encoded, shape):
    decoded=[]
    for rl in encoded:
        r,p = rl[0], rl[1]
        decoded.extend([p]*int(r))

    dimg = np.array(zigzag_to_2d_custom(decoded, shape)).reshape(shape)
    return dimg


def block_dct_quantize(image_block, quant_dc=16, quant_ac=8):
    dct_block = cv2.dct(np.float32(image_block))
    
    dct_block[0, 0] = np.round(dct_block[0, 0] / quant_dc)
    dct_block[1:, 1:] = np.round(dct_block[1:, 1:] / quant_ac)

    return dct_block

def block_idct_dequantize(dct_block, quant_dc=16, quant_ac=8):
    dct_block[0, 0] *= quant_dc
    dct_block[1:, 1:] *= quant_ac

    idct_block = cv2.idct(dct_block)
    
    return idct_block

def process_image(img):
    rows, cols = img.shape

    encoded_blocks = []

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block = img[i:i+8, j:j+8]
            dct_quantized_block = block_dct_quantize(block)
            encoded_block = RLE_encoding(dct_quantized_block)
            encoded_blocks.append(encoded_block)

    decoded_blocks = []

    for encoded_block in encoded_blocks:
        decoded_block = RLE_decode(encoded_block, (8,8))
        decoded_block = block_idct_dequantize(decoded_block)
        decoded_blocks.append(decoded_block)

    decoded_image = np.zeros_like(img)

    index = 0
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            decoded_image[i:i+8, j:j+8] = decoded_blocks[index]
            index += 1

    return decoded_image.astype(np.uint8)

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
output_image = process_image(img)
cv2.imwrite("origin.png", img)
cv2.imwrite("reconstruct.png", output_image)