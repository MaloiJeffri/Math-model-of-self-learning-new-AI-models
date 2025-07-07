import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Model parameters
gamma_1 = 0.5
gamma_2 = 0.3
gamma_q = 0.1
alpha = 0.05
beta = 0.1
lambda_reg = 10
xi_q = 0.2
omega_q = 2 * np.pi
theta_q = 0
epsilon_q_noise = 0.05
sigma = 1.0
omega = 1.0
U_max = 1.0

# Model functions
def F(P, phi, E):
    return gamma_1 * (P + phi) - gamma_2 * E

def U(P, t):
    return np.exp(-alpha * t) * np.sum(F(P, 0, 0))  # Упрощенная функция полезности

def G(U_val):
    return 1 / (1 + lambda_reg * (U_val - U_max)**2)

def epsilon_q(t):
    return xi_q * np.sin(omega_q * t + theta_q) + epsilon_q_noise * np.random.randn()

def model(t, state):
    P, phi = state

    # Calculation of external influence E
    E = sigma * np.sin(omega * t) + np.random.randn() * 0.1  # случайный шум для внешнего воздействия

    # Calculating the utility function
    U_val = U(P, t)

    # Suppression or enhancement of fluctuations depending on utility
    G_val = G(U_val)
    eps_q = epsilon_q(t)

    # Differential equations for P and phi
    dPdt = F(P, phi, E) + gamma_q * G_val * eps_q
    dphidt = alpha * (E - phi) + beta * np.random.randn()  # случайная ошибка восприятия

    return [dPdt, dphidt]

# Initial conditions and time interval
initial_conditions = [0.5, 0.5]  # Начальные состояния P и phi
time = np.linspace(0, 100, 10000)

# Solution of a system of equations
solution = integrate.solve_ivp(model, [0, 100], initial_conditions, t_eval=time)

# Visualisation of results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(solution.t, solution.y[0], label='P (когнитивный паттерн)')
plt.xlabel('Время')
plt.ylabel('P(t)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(solution.t, solution.y[1], label='Фазовый сдвиг (phi)', color='orange')
plt.xlabel('Время')
plt.ylabel('phi(t)')
plt.legend()

plt.tight_layout()
plt.show()
