# Markov Weighted Fuzzy Time Series

# 1. Eksplorasi Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
DATA_URL = "https://docs.google.com/spreadsheets/d/1osgdVkDUDTR7krgvjQdGCCO0oNt11Gev/export?format=csv"
TRAIN_RATIO = 0.8
K_CLUSTERS = 5
FUZZINESS = 2
POPULATION_SIZE = 20
MAX_ITER = 100

np.random.seed(SEED)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif", "Liberation Serif"],
    "font.size": 12,
})

def to_numbers(nilai):
    if pd.isna(nilai):
        return np.nan
    if isinstance(nilai, str):
        nilai = nilai.replace(",", "").strip()
    return float(nilai)

def load_data(url):
    df = pd.read_csv(url)
    df.columns = ["Tanggal", "Harga"]
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=True)
    df["Harga"] = df["Harga"].apply(to_numbers)
    return df.dropna(subset=["Tanggal", "Harga"]).reset_index(drop=True)

def bagi_data(data, rasio_latih=0.8):
    data = np.asarray(data, dtype=float)
    titik_potong = int(rasio_latih * len(data))
    return data[:titik_potong], data[titik_potong:]

df = load_data(DATA_URL)
data = df["Harga"].to_numpy(dtype=float)
train_data, test_data = bagi_data(data, TRAIN_RATIO)

statistik_deskriptif = pd.Series({
    "jumlah_data": len(data),
    "jumlah_train": len(train_data),
    "jumlah_test": len(test_data),
    "minimum": np.min(data),
    "maksimum": np.max(data),
    "rata_rata": np.mean(data),
    "median": np.median(data),
    "standar_deviasi": np.std(data, ddof=1),
})
shapiro_stat, shapiro_pvalue = stats.shapiro(data)

#2. Fuzzy K-Medoids (FKM) dan Silhouette Coefficient (SC)
def fuzzy_kmedoids(data, k, fuzziness=2, max_iter=100):
    data = np.asarray(data, dtype=float)
    n = len(data)
    idx_medoid = np.random.choice(n, k, replace=False)
    medoid = data[idx_medoid]
    memberships = np.zeros((n, k), dtype=float)
    for _ in range(max_iter):
        jarak = np.abs(data[:, None] - medoid[None, :])

        for i in range(n):
            jarak_nol = np.where(jarak[i] == 0)[0]
            if len(jarak_nol):
                memberships[i] = 0
                memberships[i, jarak_nol[0]] = 1
                continue
            for j in range(k):
                rasio = (jarak[i, j] / jarak[i]) ** (2 / (fuzziness - 1))
                memberships[i, j] = 1 / np.sum(rasio)

        idx_medoid_baru = np.array([
            np.argmin(np.sum((memberships[:, j] ** fuzziness)[:, None] * np.abs(data[:, None] - data[None, :]), axis=0))
            for j in range(k)
        ])

        if np.array_equal(idx_medoid, idx_medoid_baru):
            break
        idx_medoid, medoid = idx_medoid_baru, data[idx_medoid_baru]

    urutan = np.argsort(medoid)
    return medoid[urutan], memberships[:, urutan]

def silhouette_coefficient(data, label):
    data = np.asarray(data, dtype=float)
    label = np.asarray(label)
    klaster = np.unique(label)
    if len(klaster) < 2:
        return np.nan

    skor = []
    for i, nilai in enumerate(data):
        satu_klaster = data[(label == label[i]) & (np.arange(len(data)) != i)]
        if len(satu_klaster) == 0:
            skor.append(0.0)
            continue
        a = np.mean(np.abs(nilai - satu_klaster))
        b = min(np.mean(np.abs(nilai - data[label == k])) for k in klaster if k != label[i])
        skor.append((b - a) / max(a, b))

    return float(np.mean(skor))


def build_intervals(medoids, data):
    # boundary[0..k] : batas antar interval, boundary[i] & boundary[i+1] membatasi A(i+1)
    C = np.sort(np.asarray(medoids, dtype=float))
    bound = np.empty(len(C) + 1)
    bound[1:-1] = (C[:-1] + C[1:]) / 2
    bound[0] = C[0] - abs(C[0] - np.min(data))
    bound[-1] = C[-1] + abs(np.max(data) - C[-1])
    return bound

np.random.seed(SEED)
sc_list = []
best_sc = 0

for k in range(21):
    medoids, keanggotaan = fuzzy_kmedoids(train_data, k + 1, FUZZINESS)
    label_klaster = np.argmax(keanggotaan, axis=1) + 1
    sc = silhouette_coefficient(train_data, label_klaster)
    sc_list.append(sc)
    if not np.isnan(sc) and sc > best_sc:
        best_sc = sc
        opt_medoids = medoids

boundary_fkm = build_intervals(opt_medoids, train_data)


# ## 3. Markov Weighted Fuzzy Time Series (MWFTS)
def fuzzify(data, boundary):
    # [boundary[i], boundary[i+1]] -> label A(i+1)
    boundary = np.asarray(boundary, dtype=float)
    n_state = len(boundary) - 1
    idx = np.clip(np.searchsorted(boundary[1:], data, side="left"), 0, n_state - 1)
    return [f"A{i + 1}" for i in idx]

def build_flr(label):
    return list(zip(label[:-1], label[1:]))

def build_weighted_markov_matrix(flr, n_state):
    freq = np.zeros((n_state, n_state), dtype=float)
    for asal, tujuan in flr:
        freq[int(asal[1:]) - 1, int(tujuan[1:]) - 1] += 1

    total_baris = freq.sum(axis=1, keepdims=True)
    return np.divide(
        freq, total_baris,
        out = np.full_like(freq, 1 / n_state),
        where = total_baris != 0,
    )

def mwfts_forecast(data, label, boundary, transisi):
    data = np.asarray(data, dtype=float)
    midpoint = (np.asarray(boundary[:-1], dtype=float) + np.asarray(boundary[1:], dtype=float)) / 2
    ramalan = [None]

    for t in range(len(data) - 1):
        state = int(label[t][1:]) - 1
        peluang = transisi[state]

        if np.count_nonzero(peluang) == 1:
            nilai = midpoint[np.argmax(peluang)]
        else:
            nilai = sum(
                (data[t] if j == state else midpoint[j]) * peluang[j]
                for j in range(len(midpoint))
            )

        koreksi = abs(data[t] - midpoint[state])
        nilai += koreksi if data[t] > midpoint[state] else -koreksi
        ramalan.append(float(nilai))

    return ramalan


def evaluate_forecast(aktual, ramalan):
    aktual = np.asarray(aktual, dtype=float)
    ramalan = np.asarray(ramalan[1:] if ramalan[0] is None else ramalan, dtype=float)
    aktual = aktual[-len(ramalan):]
    galat = aktual - ramalan
    bukan_nol = aktual != 0

    return {
        "MAE": float(np.mean(np.abs(galat))),
        "MAPE (%)": float(np.mean(np.abs(galat[bukan_nol] / aktual[bukan_nol])) * 100),
        "RMSE": float(np.sqrt(np.mean(galat ** 2))),
    }

def run_mwfts(data, boundary, nama_model=None, data_latih=None):
    boundary = np.asarray(boundary, dtype=float)
    sumber_latih = np.asarray(data if data_latih is None else data_latih, dtype=float)
    label_latih = fuzzify(sumber_latih, boundary)
    transisi = build_weighted_markov_matrix(build_flr(label_latih), len(boundary) - 1)

    if data_latih is None:
        ramalan = mwfts_forecast(data, label_latih, boundary, transisi)
        metrik = evaluate_forecast(data, ramalan)
    else:
        data_uji = np.concatenate(([sumber_latih[-1]], np.asarray(data, dtype=float)))
        label_uji = fuzzify(data_uji, boundary)
        ramalan = np.asarray(mwfts_forecast(data_uji, label_uji, boundary, transisi)[1:], dtype=float)
        print(ramalan)
        metrik = evaluate_forecast(data, ramalan.tolist())

    hasil = {**metrik, "Boundary": boundary, "Forecast": ramalan, "Transition Matrix": transisi}
    return {"Model": nama_model, **hasil} if nama_model is not None else hasil

# 4. Optimasi Batas Interval
def create_optimizer_population(boundary, data_latih, jumlah_populasi, seed=42):
    np.random.seed(seed)
    init = np.asarray(boundary, dtype=float)
    unique_val = np.sort(np.unique(data_latih))
    populasi = [init.copy()]

    for _ in range(jumlah_populasi - 1):
        candidate = init.copy()
        for i in range(1, len(candidate) - 1):
            adj = unique_val[
                (unique_val >= init[i] * 0.9) & (unique_val <= init[i] * 1.1)
            ]
            candidate[i] = np.random.choice(adj if len(adj) else unique_val)

        populasi.append(np.concatenate((
            [unique_val.min()], np.sort(candidate[1:-1]), [unique_val.max()],
        )))

    return np.asarray(populasi, dtype=float)

def repair_position(position, min_data, max_data):
    return np.sort(np.clip(np.asarray(position, dtype=float), min_data, max_data))

def make_boundary(position, min_data, max_data):
    return np.concatenate((
        [min_data],
        repair_position(position, min_data, max_data),
        [max_data],
    ))

def evaluate_boundary_rmse(data, boundary):
    try:
        rmse = run_mwfts(data, boundary)["RMSE"]
        return rmse if np.isfinite(rmse) else np.inf
    except Exception:
        return np.inf

def pso(data, particles, max_iter, w_max=0.9, w_min=0.4, c1=2, c2=2, seed=42):
    np.random.seed(seed)
    particles = np.asarray(particles, dtype=float)
    if particles.ndim == 1:
        particles = particles.reshape(1, -1)

    min_data, max_data = np.min(data), np.max(data)
    n_particles, n_boundaries = particles.shape
    optimized_dim = n_boundaries - 2
    positions = np.sort(particles[:, 1:-1], axis=1)
    velocities = np.random.uniform(-0.5, 0.5, size=(n_particles, optimized_dim))

    pbest_positions = positions.copy()
    pbest_rmse = np.array([
        evaluate_boundary_rmse(data, make_boundary(position, min_data, max_data))
        for position in positions
    ])

    best_index = int(np.argmin(pbest_rmse))
    gbest_position = pbest_positions[best_index].copy()
    gbest_rmse = float(pbest_rmse[best_index])
    rmse_history = [gbest_rmse]

    for iteration in range(max_iter):
        inertia = w_max - (iteration * (w_max - w_min)) / max_iter

        for i in range(n_particles):
            r1 = np.random.random(optimized_dim)
            r2 = np.random.random(optimized_dim)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )
            positions[i] = repair_position(positions[i] + velocities[i], min_data, max_data)
            boundary = make_boundary(positions[i], min_data, max_data)
            rmse = evaluate_boundary_rmse(data, boundary)

            if rmse < pbest_rmse[i]:
                pbest_rmse[i] = rmse
                pbest_positions[i] = positions[i].copy()
                if rmse < gbest_rmse:
                    gbest_rmse = float(rmse)
                    gbest_position = positions[i].copy()
        rmse_history.append(gbest_rmse)

    boundary = make_boundary(gbest_position, min_data, max_data)
    lb, ub = boundary[:-1].copy(), boundary[1:].copy()
    return boundary, gbest_rmse, lb, ub, rmse_history

def cmbo(data, population, max_iter, seed=42):
    np.random.seed(seed)
    population = np.asarray(population, dtype=float)
    if population.ndim == 1:
        population = population.reshape(1, -1)

    min_data, max_data = np.min(data), np.max(data)
    n_population, n_boundaries = population.shape
    optimized_dim = n_boundaries - 2
    n_mouse = n_population // 2
    n_cat = n_population - n_mouse
    positions = np.sort(population[:, 1:-1], axis=1)
    fitness = np.array([
        evaluate_boundary_rmse(data, make_boundary(position, min_data, max_data))
        for position in positions
    ])

    order = np.argsort(fitness)
    positions, fitness = positions[order], fitness[order]
    best_position = positions[0].copy()
    best_rmse = float(fitness[0])
    rmse_history = [best_rmse]

    for _ in range(max_iter):
        order = np.argsort(fitness)
        positions, fitness = positions[order], fitness[order]
        mice, mouse_fitness = positions[:n_mouse].copy(), fitness[:n_mouse].copy()
        cats, cat_fitness = positions[n_mouse:].copy(), fitness[n_mouse:].copy()

        for j in range(n_cat):
            target = np.random.randint(0, n_mouse)
            candidate = cats[j] + np.random.random(optimized_dim) * (mice[target] - cats[j])
            candidate = repair_position(candidate, min_data, max_data)
            candidate_fitness = evaluate_boundary_rmse(
                data,
                make_boundary(candidate, min_data, max_data),
            )
            if candidate_fitness < cat_fitness[j]:
                cats[j], cat_fitness[j] = candidate, candidate_fitness

        for i in range(n_mouse):
            heaven = np.zeros(optimized_dim)
            for d in range(optimized_dim):
                heaven[d] = positions[np.random.randint(0, n_population), d]
            heaven = repair_position(heaven, min_data, max_data)
            heaven_fitness = evaluate_boundary_rmse(
                data,
                make_boundary(heaven, min_data, max_data),
            )
            direction = np.sign(mouse_fitness[i] - heaven_fitness) 
            candidate = mice[i] + np.random.random(optimized_dim) * (heaven - mice[i]) * direction
            candidate = repair_position(candidate, min_data, max_data)
            candidate_fitness = evaluate_boundary_rmse(
                data,
                make_boundary(candidate, min_data, max_data),
            )
            if candidate_fitness < mouse_fitness[i]:
                mice[i], mouse_fitness[i] = candidate, candidate_fitness

        positions = np.vstack([mice, cats])
        fitness = np.concatenate([mouse_fitness, cat_fitness])
        best_index = int(np.argmin(fitness))

        if fitness[best_index] < best_rmse:
            best_position = positions[best_index].copy()
            best_rmse = float(fitness[best_index])
        rmse_history.append(best_rmse)

    boundary = make_boundary(best_position, min_data, max_data)
    lb, ub = boundary[:-1].copy(), boundary[1:].copy()
    return boundary, best_rmse, lb, ub, rmse_history

population = create_optimizer_population(boundary_fkm, train_data, POPULATION_SIZE, seed=SEED)

boundary_pso = pso(
    train_data, population, max_iter=MAX_ITER, seed=SEED,
)[0]

boundary_cmbo = cmbo(
    train_data, population, max_iter=MAX_ITER, seed=SEED,
)[0]

# ## 5. Hasil Optimal
def compare_models(data_latih, data_uji, kandidat_boundary):
    return [
        run_mwfts(data_uji, boundary, nama, data_latih)
        for nama, boundary in kandidat_boundary.items()
    ]

model_boundaries = {
    "MWFTS-FKM": boundary_fkm,
    "MWFTS-PSO": boundary_pso,
    "MWFTS-CMBO": boundary_cmbo,
}

results = compare_models(train_data, test_data, model_boundaries)

evaluate_model = pd.DataFrame([
    {
        "Model Optimal": hasil["Model"],
        "MAE": hasil["MAE"],
        "MAPE (%)": hasil["MAPE (%)"],
        "RMSE": hasil["RMSE"],
        "Boundary": np.round(hasil["Boundary"], 4).tolist(),
    }
    for hasil in results
])
print(evaluate_model)

fig, ax = plt.subplots(figsize=(12, 6))
time_axis = np.arange(1, len(test_data) + 1)
ax.plot(time_axis, test_data, "k-", label="Aktual", linewidth=1.6)

for hasil in results:
    ax.plot(time_axis, hasil["Forecast"], linewidth=1.2, label=hasil["Model"])

ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
fig.tight_layout()
plt.show()


