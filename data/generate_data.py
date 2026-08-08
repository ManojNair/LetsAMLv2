"""Generate the synthetic datasets used throughout the AI-300 labs.

Creates:
  data/diabetes.csv        - 5,000 patient records for training (binary classification)
  data/diabetes-drift.csv  - 2,000 records with shifted feature distributions,
                             used in the model-monitoring lab to simulate data drift

Only the Python standard library is used, so the script runs anywhere.
Re-running it reproduces identical files (fixed random seed).
"""
import csv
import math
import random

random.seed(42)

HEADER = [
    "PatientID", "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
    "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age", "Diabetic",
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_row(pid, drift=False):
    # Drift shifts the population: older, heavier, higher glucose.
    glucose_mu = 118 if not drift else 138
    bmi_mu = 31 if not drift else 35.5
    age_lam = 0.045 if not drift else 0.028

    age = clamp(int(21 + random.expovariate(age_lam) * 0.8), 21, 89)
    pregnancies = random.choices(range(0, 15), weights=[18, 16, 14, 11, 9, 7, 6, 5, 4, 3, 3, 2, 1, 1, 1])[0]
    glucose = clamp(random.gauss(glucose_mu, 32), 44, 250)
    bp = clamp(random.gauss(71 + age * 0.12, 12), 40, 122)
    triceps = clamp(random.gauss(28, 12), 7, 92)
    insulin = clamp(random.gauss(120 + glucose * 0.9, 95), 14, 799)
    bmi = clamp(random.gauss(bmi_mu, 7.5), 18.0, 56.0)
    pedigree = clamp(random.expovariate(2.5), 0.07, 2.42)

    # Latent risk score drives the label so the data is genuinely learnable.
    risk = (
        (glucose - 110) * 0.042
        + (bmi - 30) * 0.09
        + (age - 40) * 0.02
        + pregnancies * 0.05
        + pedigree * 0.9
        - 1.1
        + random.gauss(0, 0.9)
    )
    diabetic = 1 if 1 / (1 + math.exp(-risk)) > 0.5 else 0

    return [
        pid, pregnancies, round(glucose), round(bp), round(triceps),
        round(insulin), round(bmi, 1), round(pedigree, 3), age, diabetic,
    ]


def write_csv(path, n, start_id, drift=False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(n):
            w.writerow(make_row(start_id + i, drift=drift))
    print(f"wrote {path} ({n} rows)")


if __name__ == "__main__":
    write_csv("diabetes.csv", 5000, 1000000)
    write_csv("diabetes-drift.csv", 2000, 2000000, drift=True)
