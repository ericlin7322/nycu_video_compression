import numpy as np

def psnr(original, dct):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((original - dct) ** 2)

    # Calculate the maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate the PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    print(f"PSNR: {psnr:.5f} dB")