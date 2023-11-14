import numpy as np

def img_to_zigzag_custom(img, shape):
    rows, cols = shape
    fimg = []

    for i in range(rows + cols - 1):
        if i % 2 == 0:
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                fimg.append(img[j, i - j])
        else:
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                fimg.append(img[j, i - j])

    return fimg

def zigzag_to_2d_custom(zigzag_array, rows, cols):
    result_matrix = np.zeros((rows, cols))

    i, j = 0, 0

    for k in range(len(zigzag_array)):
        
        result_matrix[i, j] = zigzag_array[k]
        print(i,j)
        if (i + j) % 2 == 0:  # Even sum indices
            
            if j == cols - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                j += 1
                i -= 1
        else:  # Odd sum indices
            if i == rows - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i += 1
                j -= 1

    return result_matrix

# Example usage
rows = 3
cols = 3

# Assuming dd is your zigzag-scanned 1D array
dd = [1, 2, 4, 7, 5, 3, 6, 8, 9]
ddd = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
ddd = np.array(ddd)
shape = (3,3)
# Convert zigzag array back to 2D

converted_array = zigzag_to_2d_custom(dd, rows, cols)
temp = img_to_zigzag_custom(ddd, shape)

print("Zigzag Scanned Array:")
print(dd)
print("\nConverted Array:")
print(temp)