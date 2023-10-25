import numpy as np

def psnr(original, dct):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((original - dct) ** 2)

    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance.
        psnr = 100
        print(f"PSNR: {psnr:.5f} dB")
        return psnr

    # Calculate the maximum possible pixel value
    max_pixel_value = 255.0

    # Calculate the PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    print(f"PSNR: {psnr:.5f} dB")
    return psnr