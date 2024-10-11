import numpy as np
import matplotlib.pyplot as plt

# Definisikan fungsi CL(AR)
def CL(AR):
    return (2 * np.pi * AR) / (1 + 2 / AR)

# Definisikan turunan fungsi CL(AR)
def dCL(AR):
    return (2 * np.pi * (AR**2 + 4 * AR)) / (AR + 2)**2

# Metode Newton-Raphson
def newton_raphson(initial_guess, tolerance, max_iterations):
    AR = initial_guess
    for _ in range(max_iterations):
        f_value = dCL(AR)
        if f_value == 0:  # Hindari pembagian dengan nol
            break
        AR_new = AR - CL(AR) / f_value
        if abs(AR_new - AR) < tolerance:
            return AR_new
        AR = AR_new
    return AR  # Kembalikan nilai AR jika tidak konvergen

# Parameter
initial_guess = 5.0
tolerance = 0.001
max_iterations = 100

# Mencari nilai optimal
optimal_AR = newton_raphson(initial_guess, tolerance, max_iterations)
print(f"Optimal AR: {optimal_AR:.4f}")

# Plotting
AR_values = np.linspace(1, 10, 100)
CL_values = CL(AR_values)

plt.figure(figsize=(10, 6))
plt.plot(AR_values, CL_values, label='$C_L(AR)$', color='blue')
plt.axvline(optimal_AR, color='red', linestyle='--', label=f'Optimal AR: {optimal_AR:.4f}')
plt.title('Koefisien Gaya Angkat $C_L$ terhadap Rasio Aspek Sayap (AR)')
plt.xlabel('Rasio Aspek Sayap (AR)')
plt.ylabel('$C_L(AR)$')
plt.legend()
plt.grid()
plt.show()
