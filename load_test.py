import os
import time
import tracemalloc

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Chargement des artefacts
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"

imputer    = joblib.load(os.path.join(ARTIFACTS_DIR, "imputer.joblib"))
encoder    = joblib.load(os.path.join(ARTIFACTS_DIR, "ordinal_encoder.joblib"))
scaler     = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_if.joblib"))
iso_forest = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
model_math = joblib.load(os.path.join(ARTIFACTS_DIR, "catboost_math_score.joblib"))
model_lang = joblib.load(os.path.join(ARTIFACTS_DIR, "catboost_language_mean_score.joblib"))

EXPECTED_FEATURES = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]


# ---------------------------------------------------------------------------
# Pipeline de prédiction batch (Imputing → Scaling → IF → Prediction)
# ---------------------------------------------------------------------------
def predict_batch(X: pd.DataFrame) -> None:
    """Exécute le pipeline complet sur un batch d'entrées."""
    # 1. Imputation
    arr_imp = imputer.transform(X)
    X_imp = pd.DataFrame(arr_imp, columns=EXPECTED_FEATURES)
    for col in EXPECTED_FEATURES:
        X_imp[col] = X_imp[col].astype(str)

    # 2. Encodage ordinal + normalisation
    X_enc  = encoder.transform(X_imp)
    X_norm = scaler.transform(X_enc)

    # 3. Isolation Forest
    iso_forest.score_samples(X_norm)
    iso_forest.predict(X_norm)

    # 4. Prédictions CatBoost
    model_math.predict(X_imp)
    model_lang.predict(X_imp)


# ---------------------------------------------------------------------------
# Chargement du dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("data/StudentsPerformance.csv")
X_full = df[EXPECTED_FEATURES].copy()
n = len(X_full)

print(f"Dataset original : {n} lignes\n")

# ---------------------------------------------------------------------------
# Construction des tailles de test
# Étape 1 : chunks croissants du dataset  (20 %, 40 %, 60 %, 80 %, 100 %)
# Étape 2 : réplication artificielle      (2×, 3×, 4×, 5×)
# ---------------------------------------------------------------------------
fractions    = [0.2, 0.4, 0.6, 0.8, 1.0]
replications = [2, 3, 4, 5]

test_sizes = [max(1, int(n * f)) for f in fractions]
for rep in replications:
    test_sizes.append(n * rep)

# ---------------------------------------------------------------------------
# Mesure des performances
# ---------------------------------------------------------------------------
sample_sizes = []
exec_times   = []
peak_rams    = []
throughputs  = []

print(f"{'Échantillons':>12} | {'Temps (s)':>10} | {'Débit (pred/s)':>15} | {'RAM max (MB)':>12}")
print("-" * 60)

for size in test_sizes:
    if size <= n:
        X_test = X_full.iloc[:size].copy()
    else:
        rep    = size // n
        X_test = pd.concat([X_full] * rep, ignore_index=True)

    tracemalloc.start()
    t0 = time.time()
    predict_batch(X_test)
    elapsed = time.time() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb    = peak_ram / (1024 * 1024)
    throughput = len(X_test) / elapsed

    sample_sizes.append(len(X_test))
    exec_times.append(elapsed)
    peak_rams.append(peak_mb)
    throughputs.append(throughput)

    print(f"{len(X_test):>12} | {elapsed:>10.4f} | {throughput:>15.0f} | {peak_mb:>12.2f}")

# ---------------------------------------------------------------------------
# Test de charge (assertion sur la condition nominale)
# Charge nominale = dataset complet (100 %)
# ---------------------------------------------------------------------------
NOMINAL_SAMPLES    = n
NOMINAL_TIME_LIMIT = 5.0  # secondes

nominal_idx  = sample_sizes.index(NOMINAL_SAMPLES)
nominal_time = exec_times[nominal_idx]

print(f"\n--- Test de charge (charge nominale : {NOMINAL_SAMPLES} échantillons) ---")
assert nominal_time < NOMINAL_TIME_LIMIT, (
    f"ECHEC : {nominal_time:.4f}s > limite {NOMINAL_TIME_LIMIT}s"
)
print(f"SUCCES : {nominal_time:.4f}s < {NOMINAL_TIME_LIMIT}s (limite nominale)")

# ---------------------------------------------------------------------------
# Tracé des courbes de performance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Tests de performance — Pipeline Students Performance", fontsize=13)

# Latence
axes[0].plot(sample_sizes, exec_times, marker="o", color="steelblue")
axes[0].axvline(NOMINAL_SAMPLES, color="red",    linestyle="--", label="Charge nominale")
axes[0].axhline(NOMINAL_TIME_LIMIT, color="orange", linestyle="--",
                label=f"Limite {NOMINAL_TIME_LIMIT}s")
axes[0].set_title("Latence")
axes[0].set_xlabel("Nombre d'échantillons")
axes[0].set_ylabel("Temps d'exécution (s)")
axes[0].legend()
axes[0].grid(True)

# Empreinte mémoire
axes[1].plot(sample_sizes, peak_rams, marker="o", color="darkorange")
axes[1].axvline(NOMINAL_SAMPLES, color="red", linestyle="--", label="Charge nominale")
axes[1].set_title("Empreinte mémoire (RAM max)")
axes[1].set_xlabel("Nombre d'échantillons")
axes[1].set_ylabel("RAM max allouée (MB)")
axes[1].legend()
axes[1].grid(True)

# Débit
axes[2].plot(sample_sizes, throughputs, marker="o", color="seagreen")
axes[2].axvline(NOMINAL_SAMPLES, color="red", linestyle="--", label="Charge nominale")
axes[2].set_title("Débit")
axes[2].set_xlabel("Nombre d'échantillons")
axes[2].set_ylabel("Prédictions par seconde")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("performance_tests.png", dpi=150)
plt.show()
print("\nGraphiques sauvegardes dans performance_tests.png")
