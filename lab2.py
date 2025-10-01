import numpy as np
import matplotlib.pyplot as plt

# Налаштування параметрів
dt = 0.01            # крок інтегрування
T = 50.0             # загальний час моделювання
dmu_defuzz = 0.1     # безрозмірний крок інтегрування чисельника та знаменника
mu_min_rad = -np.pi/36
mu_max_rad =  np.pi/36

# переведемо радіани у градуси
rad2deg = 180.0/np.pi
mu_min = mu_min_rad * rad2deg
mu_max = mu_max_rad * rad2deg

n_terms = 6 # кількість термів (поділ на 6 частин для кута, швидкості і прискорення)

# Діапазони для вхідних значень
phi_range = (-180.0, 180.0)   # діапазон для кута
omega_range = (-50.0, 50.0)   # діапазон для швидкості

# Початкові значення
phi0_deg = -30.0   # початковий кут
omega0_deg = 1.0   # початкова швидкість

# Функції для побудови трикутних нечітких множин
def make_triangular_centers(rng, n):
    # робимо рівномірні "центри" термів у заданому діапазоні
    a, b = rng
    centers = np.linspace(a, b, n)
    width = (b - a) / (n - 1)   # відстань між центрами
    return centers, width*1.5   # ширина для трикутників

def triangular_mu(x, center, halfwidth):
    # обчислюємо, наскільки x "належить" до трикутної функції
    d = abs(x - center)
    if d >= halfwidth:
        return 0.0
    return 1.0 - d/halfwidth

# центри і ширини для кута та швидкості
phi_centers, phi_hw = make_triangular_centers(phi_range, n_terms)
omega_centers, omega_hw = make_triangular_centers(omega_range, n_terms)

# центри для вихідного прискорення
mu_centers = np.linspace(mu_min, mu_max, n_terms)
mu_hw = (mu_max - mu_min) / (n_terms - 1) * 1.5

# Правила (36 штук)
def normalize_centers(centers):
    # нормалізація значень у діапазон [-1, 1], щоб легше було комбінувати
    cmin, cmax = centers[0], centers[-1]
    return (centers - (cmin + cmax)/2.0) / ((cmax - cmin)/2.0)

norm_phi = normalize_centers(phi_centers)
norm_omega = normalize_centers(omega_centers)
norm_mu = normalize_centers(mu_centers)

# таблиця правил
rule_table = np.zeros((n_terms, n_terms), dtype=int)
for i in range(n_terms):
    for j in range(n_terms):
        # якщо і кут, і швидкість великі → даємо протилежне прискорення
        combined = -(norm_phi[i] + norm_omega[j]) / 2.0
        k = int(np.argmin(np.abs(norm_mu - combined))) # шукаємо найближчий терм
        rule_table[i, j] = k

# Нечіткий контролер
def fuzzy_control(phi, omega):
    # 1) fuzzify – рахуємо ступінь належності для всіх термів
    phi_mu = np.array([triangular_mu(phi, c, phi_hw) for c in phi_centers])
    omega_mu = np.array([triangular_mu(omega, c, omega_hw) for c in omega_centers])

    # 2) правила: беремо мінімум з phi та omega, дивимось у таблицю правил
    out_agg = np.zeros(n_terms)  # ступені для вихідних термів
    for i in range(n_terms):
        for j in range(n_terms):
            strength = min(phi_mu[i], omega_mu[j])
            out_idx = rule_table[i, j]
            out_agg[out_idx] = max(out_agg[out_idx], strength)

    # 3) дефазифікація – рахуємо середнє значення (центроїд)
    mu_vals = np.arange(mu_min, mu_max + 1e-9, dmu_defuzz)
    numerator = 0.0
    denominator = 0.0
    for m in mu_vals:
        membs = np.array([triangular_mu(m, c, mu_hw) for c in mu_centers])
        clipped = np.minimum(out_agg, membs)
        aggregated = np.max(clipped) if clipped.size else 0.0
        numerator += m * aggregated * dmu_defuzz
        denominator += aggregated * dmu_defuzz
    if denominator == 0.0:
        mu = 0.0
    else:
        mu = numerator / denominator
    mu = max(min(mu, mu_max), mu_min) # обмежуємо в межах
    return mu

# Основний цикл (метод Ейлера)
def run_simulation(phi0_deg, omega0_deg, dt, T):
    n_steps = int(np.ceil(T / dt)) + 1
    times = np.linspace(0.0, dt*(n_steps-1), n_steps)
    phi = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    mu_arr = np.zeros(n_steps)

    # стартові умови
    phi[0] = phi0_deg
    omega[0] = omega0_deg

    # крок за кроком рахуємо значення
    for k in range(n_steps-1):
        t = times[k]
        mu = fuzzy_control(phi[k], omega[k])    # керуюче прискорення
        mu_arr[k] = mu
        phi[k+1] = phi[k] + dt * omega[k]       # новий кут
        omega[k+1] = omega[k] + dt * mu         # нова швидкість
    mu_arr[-1] = fuzzy_control(phi[-1], omega[-1])
    return times, phi, omega, mu_arr

# Запуск і збереження результаті
if __name__ == "__main__":
    times, phi, omega, mu_arr = run_simulation(phi0_deg, omega0_deg, dt, T)

    # зберігаємо все у файл
    out_file = "results.txt"
    data = np.vstack([times, phi, omega, mu_arr]).T
    header = "time_s\tphi_deg\tomega_deg_per_s\tmu_deg_per_s2"
    np.savetxt(out_file, data, fmt="%.6f\t%.6f\t%.6f\t%.6f", header=header, comments='')
    print(f"Results saved to {out_file} ({data.shape[0]} rows).")

    # графіки
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(times, phi)
    axs[0].set_ylabel("phi")
    axs[0].grid(True)
    axs[1].plot(times, omega)
    axs[1].set_ylabel("omega")
    axs[1].grid(True)
    axs[2].plot(times, mu_arr)
    axs[2].set_ylabel("mu")
    axs[2].set_xlabel("time")
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()
