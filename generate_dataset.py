"""
Synthetic KPI Dataset Generator for AITU Faculty Performance System
Based on DP-AITU-61 regulations and KPI block structure
"""

import random
import numpy as np
import pandas as pd
import csv
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# 1. СПРАВОЧНИКИ
# ──────────────────────────────────────────────

DEPARTMENTS = [
    "School of Software Engineering",
    "School of Digital Public Administration",
    "School of Creative Industries",
    "School of Intelligent Systems",
    "School of Cybersecurity",
    "School of Artificial Intelligence and Data Science",
    "School of Computer Science",
]

# Казахские имена на английском (тестовые)
FIRST_NAMES_M = [
    "Aibek", "Nurlan", "Dastan", "Arman", "Yerlan", "Bekzat", "Rustem",
    "Azamat", "Timur", "Dias", "Almas", "Sanzhar", "Bauyrzhan", "Meiram",
    "Zhandos", "Adlet", "Talgat", "Serik", "Kanat", "Asset",
]
FIRST_NAMES_F = [
    "Aizat", "Dinara", "Gulnara", "Ainur", "Saule", "Meruert", "Aigerim",
    "Zhanna", "Madina", "Aliya", "Gulzat", "Kamila", "Nazerke", "Zarina",
    "Assel", "Sholpan", "Gaukhar", "Raushan", "Nurgul", "Dana",
]
LAST_NAMES = [
    "Akhmetov", "Bekova", "Nurgaliev", "Seitkali", "Dzhaksybekov",
    "Mukhambetova", "Kassymov", "Tulegenova", "Ospanov", "Abdrakhmanova",
    "Suleimenov", "Dzhakupova", "Rakhimov", "Kenzhebayeva", "Yesmagambetov",
    "Bizhanova", "Karimov", "Tokayeva", "Musayev", "Abenova",
    "Zhaksybekov", "Nurmagambetova", "Umarov", "Baisalova", "Askarov",
    "Kenzhebekova", "Seidaliev", "Zhumabayeva", "Khassanov", "Dairova",
    "Alimov", "Bekmuratova", "Tursunov", "Syzdykova", "Kulmanov",
    "Ergaliyeva", "Sharipov", "Akhmetova", "Issabekov", "Moldabayeva",
]

def generate_full_name(rng):
    gender = rng.choice(["M", "F"])
    first = rng.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
    last = rng.choice(LAST_NAMES)
    return f"{first} {last}"


# ──────────────────────────────────────────────
# 2. KPI-СХЕМЫ ПО РОЛЯМ
# Блоков макс 4 (block_1..block_4).
# Отсутствующий блок = 0.
# block_name → label для понимания
# ──────────────────────────────────────────────

# (name, max_score, is_director_block, base_bias)
KPI_SCHEMES = {
    "PROFESSOR": [
        ("research",     40, False, 0),
        ("teaching",     30, False, 0),
        ("service",      20, False, 0),
        ("director",     10, True,  5),   # небольшой директорский
    ],
    "ASSOCIATE_PROFESSOR": [
        ("research",     35, False, 0),
        ("teaching",     30, False, 0),
        ("service",      20, False, 0),
        ("director",     15, True,  5),
    ],
    "ASSISTANT_PROFESSOR": [
        ("research",     30, False, 0),
        ("teaching",     35, False, 0),
        ("service",      15, False, 0),
        ("director",     20, True,  8),
    ],
    "SENIOR_LECTURER": [
        ("teaching",     40, False, 0),
        ("director",     60, True,  10),
        (None,            0, False, 0),
        (None,            0, False, 0),
    ],
    "LECTURER": [
        ("academic",     25, False, 0),
        ("director",     75, True,  12),
        (None,            0, False, 0),
        (None,            0, False, 0),
    ],
    "ACM_HEAD_COACH": [
        ("training",     25, False, 0),
        ("competition",  25, False, 0),
        ("mentoring",    25, False, 0),
        ("activity",     25, False, 0),
    ],
    "ACM_COACH": [
        ("training",     30, False, 0),
        ("competition",  25, False, 0),
        ("mentoring",    25, False, 0),
        ("activity",     20, False, 0),
    ],
    "PE_TEACHER": [
        ("teaching",     50, False, 0),
        ("activity",     30, False, 0),
        ("service",      20, False, 0),
        (None,            0, False, 0),
    ],
    "PE_SENIOR_TEACHER": [
        ("teaching",     45, False, 0),
        ("activity",     30, False, 0),
        ("service",      15, False, 0),
        ("director",     10, True,  5),
    ],
    "DIRECTOR": [
        ("management",   40, False, 0),
        ("strategy",     30, False, 0),
        ("service",      20, False, 0),
        ("director",     10, True,  5),
    ],
}

# Роли которые не привязаны к департаменту
DEPT_INDEPENDENT_ROLES = {
    "ACM_HEAD_COACH", "ACM_COACH", "PE_TEACHER",
    "PE_SENIOR_TEACHER", "DIRECTOR",
}

# ──────────────────────────────────────────────
# 3. РАСПРЕДЕЛЕНИЕ СОТРУДНИКОВ
# (роль → кол-во на департамент или глобально)
# ──────────────────────────────────────────────

DEPT_ROLES_COUNT = {
    "PROFESSOR":           8,
    "ASSOCIATE_PROFESSOR": 12,
    "ASSISTANT_PROFESSOR": 16,
    "SENIOR_LECTURER":     20,
    "LECTURER":            24,
}

GLOBAL_ROLES_COUNT = {
    "ACM_HEAD_COACH":    6,
    "ACM_COACH":         14,
    "PE_TEACHER":        12,
    "PE_SENIOR_TEACHER":  8,
    "DIRECTOR":           7,   # по одному на департамент примерно
}

# ──────────────────────────────────────────────
# 4. ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ ПО РОЛЯМ
# (mu, sigma, min_clip, max_clip)
# ──────────────────────────────────────────────

PERF_PARAMS = {
    "PROFESSOR":           (0.75, 0.14, 0.30, 0.97),
    "ASSOCIATE_PROFESSOR": (0.72, 0.13, 0.28, 0.95),
    "ASSISTANT_PROFESSOR": (0.70, 0.12, 0.25, 0.95),
    "SENIOR_LECTURER":     (0.72, 0.10, 0.30, 0.95),
    "LECTURER":            (0.70, 0.10, 0.30, 0.95),
    "ACM_HEAD_COACH":      (0.76, 0.12, 0.40, 0.97),
    "ACM_COACH":           (0.65, 0.16, 0.25, 0.95),
    "PE_TEACHER":          (0.68, 0.11, 0.30, 0.95),
    "PE_SENIOR_TEACHER":   (0.73, 0.11, 0.35, 0.97),
    "DIRECTOR":            (0.78, 0.12, 0.40, 0.98),
}

YEARS = [2021, 2022, 2023, 2024, 2025]

# ──────────────────────────────────────────────
# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def generate_initial_performance(role, rng):
    mu, sigma, lo, hi = PERF_PARAMS[role]
    p = rng.normal(mu, sigma)
    return clamp(p, lo, hi)

def evolve_performance(p, rng):
    """Небольшой рост с вероятностью снижения."""
    delta = rng.uniform(-0.05, 0.08)
    return clamp(p + delta, 0.15, 0.99)

def generate_block_score(p, max_score, is_director, bias, rng):
    """
    p           — уровень производительности [0,1]
    max_score   — максимум блока
    is_director — признак директорского блока
    bias        — дополнительное смещение вверх для директорского
    """
    if max_score == 0:
        return 0.0

    base = p * max_score

    if is_director:
        # Директорский блок: смещение вверх + редкий провал
        base += bias
        if rng.random() < 0.13:          # ~13% провалов
            base -= rng.uniform(12, 30)
    else:
        # Обычный блок: симметричный шум
        base += rng.uniform(-5, 5)

    return round(clamp(base, 0, max_score), 1)

# ──────────────────────────────────────────────
# 6. ГЕНЕРАЦИЯ СПИСКА СОТРУДНИКОВ
# ──────────────────────────────────────────────

def build_teachers(rng):
    teachers = []
    teacher_id = 1

    # Сотрудники привязанные к департаментам
    for dept in DEPARTMENTS:
        for role, count in DEPT_ROLES_COUNT.items():
            for _ in range(count):
                perf_init = generate_initial_performance(role, rng)
                teachers.append({
                    "teacher_id":     teacher_id,
                    "full_name":      generate_full_name(rng),
                    "department":     dept,
                    "role":           role,
                    "experience_years": int(rng.choice(range(1, 26))),
                    "perf_init":      perf_init,
                })
                teacher_id += 1

    # Сотрудники без привязки к департаменту
    for role, count in GLOBAL_ROLES_COUNT.items():
        for _ in range(count):
            perf_init = generate_initial_performance(role, rng)
            teachers.append({
                "teacher_id":     teacher_id,
                "full_name":      generate_full_name(rng),
                "department":     "N/A",
                "role":           role,
                "experience_years": int(rng.choice(range(1, 26))),
                "perf_init":      perf_init,
            })
            teacher_id += 1

    return teachers

# ──────────────────────────────────────────────
# 7. ГЕНЕРАЦИЯ ДАТАСЕТА
# ──────────────────────────────────────────────

def build_dataset(teachers, rng):
    rows = []

    for t in teachers:
        role    = t["role"]
        scheme  = KPI_SCHEMES[role]          # list of 4 tuples
        p       = t["perf_init"]
        exp_base = t["experience_years"]

        for i, year in enumerate(YEARS):
            # Эволюция производительности год за годом
            if i > 0:
                p = evolve_performance(p, rng)
            p_clamped = clamp(p, 0.0, 1.0)

            # Генерируем 4 блока
            blocks = []
            for (bname, bmax, is_dir, bias) in scheme:
                score = generate_block_score(p_clamped, bmax, is_dir, bias, rng)
                blocks.append(score)

            b1, b2, b3, b4 = blocks
            total_raw = b1 + b2 + b3 + b4
            # Общий шум ±3
            noise = rng.uniform(-3, 3)
            total = round(clamp(total_raw + noise, 0, 100), 1)

            # Опыт увеличивается с годами
            exp = exp_base + i

            rows.append({
                "teacher_id":     t["teacher_id"],
                "full_name":      t["full_name"],
                "department":     t["department"],
                "role":           role,
                "year":           year,
                "experience_years": exp,
                "block_1":        b1,
                "block_2":        b2,
                "block_3":        b3,
                "block_4":        b4,
                "total_kpi":      total,
            })

    return rows

# ──────────────────────────────────────────────
# 8. ЗАПУСК
# ──────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    py_rng = random.Random(42)

    # Monkey-patch: generate_full_name использует python random
    import types
    def _gen_name(r):
        g = r.choice(["M", "F"])
        first = r.choice(FIRST_NAMES_M if g == "M" else FIRST_NAMES_F)
        last  = r.choice(LAST_NAMES)
        return f"{first} {last}"

    # Пересоздаём с py_rng
    def build_teachers_fixed():
        teachers = []
        tid = 1
        for dept in DEPARTMENTS:
            for role, count in DEPT_ROLES_COUNT.items():
                for _ in range(count):
                    pi = generate_initial_performance(role, rng)
                    teachers.append({
                        "teacher_id":      tid,
                        "full_name":       _gen_name(py_rng),
                        "department":      dept,
                        "role":            role,
                        "experience_years": py_rng.randint(1, 25),
                        "perf_init":       pi,
                    })
                    tid += 1
        for role, count in GLOBAL_ROLES_COUNT.items():
            for _ in range(count):
                pi = generate_initial_performance(role, rng)
                teachers.append({
                    "teacher_id":      tid,
                    "full_name":       _gen_name(py_rng),
                    "department":      "N/A",
                    "role":            role,
                    "experience_years": py_rng.randint(1, 25),
                    "perf_init":       pi,
                })
                tid += 1
        return teachers

    teachers = build_teachers_fixed()
    dataset  = build_dataset(teachers, rng)

    df = pd.DataFrame(dataset)

    print(f"Total records : {len(df)}")
    print(f"Unique teachers: {df['teacher_id'].nunique()}")
    print(f"\nRecords per role:\n{df.groupby('role')['teacher_id'].nunique().sort_values()}")
    print(f"\ntotal_kpi stats:\n{df['total_kpi'].describe().round(2)}")
    print(f"\nMean KPI per role:\n{df.groupby('role')['total_kpi'].mean().round(1).sort_values()}")

    output_path = Path(__file__).resolve().parent / "kpi_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
