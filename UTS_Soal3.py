import numpy as np
import matplotlib.pyplot as plt

# Definisikan fungsi performa P(x1, x2)
def P(x1, x2):
    return - (x1 - 10) ** 2 - (x2 - 5) ** 2 + 50

# Definisikan fungsi gradien dari P
def grad_P(x1, x2):
    dP_dx1 = -2 * (x1 - 10)
    dP_dx2 = -2 * (x2 - 5)
    return np.array([dP_dx1, dP_dx2])

# Metode Gradient Descent
def gradient_descent(initial_guess, learning_rate, tolerance, max_iterations):
    x = np.array(initial_guess)
    history = [x.copy()]  # Simpan setiap langkah untuk plotting
    for _ in range(max_iterations):
        grad = grad_P(x[0], x[1])
        x_new = x - learning_rate * grad
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x, np.array(history)

# Parameter
initial_guess = [8.0, 4.0]
learning_rate = 0.1
tolerance = 0.001
max_iterations = 1000

# Mencari nilai optimal
optimal_values, history = gradient_descent(initial_guess, learning_rate, tolerance, max_iterations)
print(f"Optimal x1: {optimal_values[0]:.4f}, Optimal x2: {optimal_values[1]:.4f}")

# Plotting
x1_values = np.linspace(0, 12, 100)
x2_values = np.linspace(0, 10, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = P(X1, X2)

plt.figure(figsize=(10, 6))
plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(label='Performa P(x1, x2)')
plt.plot(history[:, 0], history[:, 1], marker='o', color='red', label='Gradient Descent Path')
plt.scatter(optimal_values[0], optimal_values[1], color='white', s=100, label='Optimal Point')
plt.title('Optimalisasi Performa Pesawat')
plt.xlabel('Panjang Sayap (x1)')
plt.ylabel('Sudut Serang (x2)')
plt.legend()
plt.grid()
plt.show()
