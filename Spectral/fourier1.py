import matplotlib.pyplot as plt
import numpy as np

# 1D Fourier Collocation
# u' = cos(x) on [0,2*pi]

N = 8
dx = 2 * np.pi / N
x_N = np.array([j * dx for j in range(0,N)])

f = np.cos
f_N = f(x_N)

f_hat_N = np.fft.fft(f_N)

k = np.fft.fftfreq(N) * N

u_hat_N = np.zeros(N, dtype=complex)
u_hat_N[(k != 0)] = f_hat_N[(k != 0)] / (1j * k[(k != 0)])
u_hat_N[0] = 0

u_N = np.fft.ifft(u_hat_N)

p = plt.scatter(x_N, u_N)
plt.show()








