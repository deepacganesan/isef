import numpy as np
bits = np.load("C:/Users/u28jk/Downloads/bits.npy")

print(bits.shape)
print(bits[0])        # first window, all 380 bits
print(bits[:5, :20])  # first 5 windows, first 20 bits
