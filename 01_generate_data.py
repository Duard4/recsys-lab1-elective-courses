# -*- coding: utf-8 -*-
"""Крок 1. Генерація синтетичного датасету для рекомендаційної системи елективних курсів.

Створює:
  data/raw/users_raw.csv, courses_raw.csv, interactions_raw.csv   — «сирі» дані з шумом
  data/ground_truth/*.csv                                           — приховані «істинні» мітки
                                                                     (використовуються лише для оцінювання)
Генерація повністю відтворювана (seed = 42).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from domain import *

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
TRUTH = ROOT / "data" / "ground_truth"

N_USERS, N_COURSES = 2000, 3000
WINDOW_START = pd.Timestamp("2025-01-01")
WINDOW_END = pd.Timestamp("2026-09-28 23:59:00")
K = len(AREA_CODES)
AIDX = {a: i for i, a in enumerate(AREA_CODES)}


def lc_first(s):
    """Перша літера малими, якщо друга — мала (щоб не псувати абревіатури)."""
    return s[0].lower() + s[1:] if len(s) > 1 and s[1].islower() else s


def rand_dates(n, start, end):
    s, e = pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9
    return pd.to_datetime(rng.integers(s, e, n), unit="s").normalize()


# =====================================================================================
# 1. КОРИСТУВАЧІ (студенти)
# =====================================================================================
def make_users():
    n = N_USERS
    ids = [f"U{i:04d}" for i in range(1, n + 1)]
    # запланована група активності
    grp = np.array(["new"] * int(.08 * n) + ["passive"] * int(.32 * n) +
                   ["regular"] * int(.48 * n) + ["active"] * (n - int(.08 * n) - int(.32 * n) - int(.48 * n)))
    rng.shuffle(grp)

    fac_area = rng.choice(AREA_CODES, n, p=np.array([.15, .13, .10, .10, .10, .10, .10, .12, .10]) / 1.0)
    dom = np.where(rng.random(n) < .60, fac_area, rng.choice(AREA_CODES, n))
    theta = np.zeros((n, K))
    for i in range(n):
        d = rng.dirichlet(np.full(K, .4))
        theta[i] = .6 * d
        theta[i, AIDX[dom[i]]] += .4 + .0
    theta /= theta.sum(1, keepdims=True)   # інтереси користувача до тематичних областей

    faculty = [AREAS[a]["faculty"] for a in fac_area]
    level = np.where(rng.random(n) < .78, "бакалавр", "магістр")
    year = np.where(level == "бакалавр", rng.integers(1, 5, n), rng.integers(1, 3, n))
    age = np.where(level == "бакалавр", 17 + year + rng.normal(.7, .8, n), 21 + year + rng.normal(.7, 1.1, n))
    mature = rng.random(n) < .02
    age = np.where(mature, rng.integers(27, 46, n), age)
    age = np.clip(np.round(age), 17, 45).astype(int)
    p_male = np.array([{"IT": .68, "ENG": .72}.get(a, .35) for a in fac_area])
    r = rng.random(n)
    gender = np.where(r < .02, "інша", np.where(r < .02 + (1 - .02) * p_male, "чоловіча", "жіноча"))
    gpa = np.round(np.clip(rng.normal(80, 9, n) + np.where(level == "магістр", 3, 0), 60, 100), 1)
    pref_fmt = rng.choice(["онлайн", "офлайн", "гібридний"], n, p=[.35, .40, .25])
    city = rng.choice(CITIES, n, p=CITY_W)

    career, interests = [], []
    for i in range(n):
        if rng.random() < .7:
            career.append(rng.choice(AREAS[dom[i]]["careers"]))
        else:
            career.append(rng.choice(GENERIC_CAREERS))
        k = rng.integers(2, 5)
        tags = []
        while len(tags) < k:
            a = dom[i] if rng.random() < .65 else rng.choice(AREA_CODES, p=theta[i])
            t = rng.choice(AREAS[a]["tags"])
            if t not in tags:
                tags.append(t)
        interests.append(";".join(tags))

    reg = rand_dates(n, "2022-09-01", "2024-12-31")
    new_mask = grp == "new"
    reg = pd.Series(reg)
    reg[new_mask] = rand_dates(new_mask.sum(), "2026-09-01", "2026-09-28")

    users = pd.DataFrame({
        "user_id": ids, "age": age, "gender": gender, "city": city, "faculty": faculty,
        "degree_level": level, "study_year": year, "gpa": gpa, "preferred_format": pref_fmt,
        "career_goal": career, "interests": interests, "registration_date": reg.dt.strftime("%Y-%m-%d"),
    })
    truth = pd.DataFrame({"user_id": ids, "dominant_area": dom, "activity_group": grp})
    return users, truth, theta, pref_fmt, gpa


# =====================================================================================
# 2. ОБ'ЄКТИ ІНТЕРЕСУ (елективні курси)
# =====================================================================================
def make_courses():
    n = N_COURSES
    ids = [f"C{i:04d}" for i in range(1, n + 1)]
    tier = np.array(["new"] * int(.06 * n) + ["rare"] * int(.34 * n) + ["regular"] * int(.45 * n) +
                    ["popular"] * (n - int(.06 * n) - int(.34 * n) - int(.45 * n)))
    rng.shuffle(tier)
    area_p = np.array([.14, .13, .11, .11, .11, .10, .10, .12, .08])
    area = rng.choice(AREA_CODES, n, p=area_p / area_p.sum())

    pools = {a: [f"{rng.choice(INSTR_SURN)} {rng.choice(INITIALS)}" for _ in range(14)] for a in AREA_CODES}

    rows, area2, base_topic = [], [], []
    for i in range(n):
        a = area[i]
        topics = AREAS[a]["topics"]
        t1_i = rng.integers(len(topics))
        t1, kw1 = topics[t1_i]
        combo = rng.random() < .55
        a2 = None
        kws = list(rng.permutation(kw1))
        if combo:
            if rng.random() < .28:   # міждисциплінарний курс
                a2 = rng.choice([x for x in AREA_CODES if x != a])
                t2, kw2 = AREAS[a2]["topics"][rng.integers(10)]
            else:
                t2_i = rng.choice([j for j in range(len(topics)) if j != t1_i])
                t2, kw2 = topics[t2_i]
            kw2 = list(rng.permutation(kw2))
            kws = kws[:2] + kw2[:2]
            title = f"{t1} та {lc_first(t2)}{rng.choice(QUALIFIERS)}"
            topic_txt = f"{lc_first(t1)} та {lc_first(t2)}"
        else:
            title = f"{t1}{rng.choice(QUALIFIERS)}"
            topic_txt = lc_first(t1)
        area2.append(a2)
        base_topic.append(topic_txt)
        diff_p = np.array([.15, .28, .30, .20, .07])
        if a in ("IT", "ENG"):
            diff_p = np.array([.06, .18, .32, .29, .15])
        difficulty = int(rng.choice([1, 2, 3, 4, 5], p=diff_p))
        en_p = {"LNG": .45, "IT": .30}.get(a, .15)
        language = rng.choice(["українська", "англійська", "польська"], p=[.98 - en_p, en_p, .02])
        fmt = rng.choice(["онлайн", "офлайн", "гібридний"], p=[.34, .38, .28])
        credits = int(rng.choice([3, 4, 5, 6], p=[.25, .30, .25, .20]))
        assess = rng.choice(["залік", "іспит", "проєкт"], p=[.45, .25, .30])
        desc = " ".join([
            rng.choice(INTRO).format(t=topic_txt),
            rng.choice(CONTENT).format(k1=kws[0], k2=kws[1], k3=kws[2], k4=kws[3]),
            rng.choice(OUTCOME).format(k1=kws[0], k2=kws[1], k3=kws[2], k4=kws[3]),
            LEVEL_LOW if difficulty <= 2 else (LEVEL_HIGH if difficulty >= 4 else ""),
            f"Формат: {fmt}. Оцінювання: {assess}.",
        ]).replace("  ", " ").strip()
        rows.append(dict(
            course_id=ids[i], title=title, description=desc, category=AREAS[a]["name"],
            department=AREAS[a]["faculty"], credits=credits, difficulty=difficulty, language=language,
            format=fmt, semester=rng.choice(["осінній", "весняний", "обидва"], p=[.4, .4, .2]),
            instructor=rng.choice(pools[a]), weekly_hours=int(np.clip(credits + rng.integers(-1, 3), 2, 9)),
            max_students=int(rng.choice([20, 30, 40, 60, 80, 100, 150])), assessment=assess,
            keywords=";".join(kws),
        ))
    courses = pd.DataFrame(rows)
    # унікальні назви
    dup = courses.groupby("title").cumcount()
    courses.loc[dup > 0, "title"] = courses.loc[dup > 0, "title"] + " (група " + (dup[dup > 0] + 1).astype(str) + ")"
    pub = pd.Series(rand_dates(n, "2021-09-01", "2024-12-31"))
    newm = tier == "new"
    pub[newm] = rand_dates(newm.sum(), "2026-09-01", "2026-09-28")
    courses["publication_date"] = pub.dt.strftime("%Y-%m-%d")

    # приховані параметри: популярність і якість
    pop = np.zeros(n)
    pop[tier == "popular"] = rng.lognormal(2.2, .30, (tier == "popular").sum())     # ≈ 9
    pop[tier == "regular"] = rng.lognormal(.15, .45, (tier == "regular").sum())     # ≈ 1.2
    pop[tier == "rare"] = rng.lognormal(-1.9, .4, (tier == "rare").sum())           # ≈ 0.16
    q = np.zeros(n)
    q[tier == "popular"] = rng.normal(4.25, .25, (tier == "popular").sum())
    q[tier == "regular"] = rng.normal(3.70, .40, (tier == "regular").sum())
    rare_idx = np.where(tier == "rare")[0]
    gem = rng.random(len(rare_idx)) < .30                                           # «приховані перлини»
    q[rare_idx] = np.where(gem, rng.normal(4.55, .20, len(rare_idx)), rng.normal(3.35, .50, len(rare_idx)))
    q[tier == "new"] = rng.normal(3.7, .4, (tier == "new").sum())
    q = np.clip(q, 2, 5)
    gem_flag = np.zeros(n, bool)
    gem_flag[rare_idx] = gem
    truth = pd.DataFrame({"course_id": ids, "area": area, "secondary_area": area2, "popularity_group": tier,
                          "popularity_weight": np.round(pop, 4), "quality": np.round(q, 3), "hidden_gem": gem_flag})
    return courses, truth, pop, q, area, area2, base_topic


# =====================================================================================
# 3. ВЗАЄМОДІЇ
# =====================================================================================
def make_interactions(users, utruth, ctruth, courses, theta, pref_fmt, gpa, pop, q, area, area2, base_topic):
    nU, nC = N_USERS, N_COURSES
    a_idx = np.array([AIDX[a] for a in area])
    a2_idx = np.array([AIDX[a] if a else -1 for a in area2])
    S = 6.0 * theta[:, a_idx]                                     # відповідність тематиці
    has2 = a2_idx >= 0
    S[:, has2] += 2.0 * theta[:, a2_idx[has2]]
    cfmt = courses["format"].to_numpy(dtype=object)
    pf = np.asarray(pref_fmt, dtype=object)
    S += 0.5 * (pf[:, None] == cfmt[None, :]) + 0.25 * ((pf[:, None] == "гібридний") | (cfmt[None, :] == "гібридний"))
    pref_d = 1.5 + (gpa - 60) / 40 * 3
    S -= 0.3 * np.abs(courses["difficulty"].to_numpy(dtype=float)[None, :] - pref_d[:, None])
    with np.errstate(divide="ignore"):
        logpop = np.log(pop)                                       # -inf для нових курсів

    grp = utruth["activity_group"].to_numpy(dtype=object)
    n_pairs = np.zeros(nU, int)
    for i, g in enumerate(grp):
        if g == "passive":
            n_pairs[i] = rng.integers(1, 7)
        elif g == "regular":
            n_pairs[i] = rng.integers(8, 36)
        elif g == "active":
            n_pairs[i] = int(np.clip(50 + rng.gamma(2.5, 20), 50, 300))

    pairs = {}
    for u in range(nU):
        if n_pairs[u] == 0:
            continue
        keys = logpop + S[u] + rng.gumbel(size=nC)
        top = np.argpartition(-keys, n_pairs[u])[:n_pairs[u]]
        pairs[u] = set(top.tolist())

    # гарантуємо хоча б одну взаємодію в кожного «рідкісного» та «звичайного» курсу
    covered = set().union(*pairs.values())
    active_users = np.array(list(pairs.keys()))
    for c in np.where(np.isin(ctruth["popularity_group"].to_numpy(dtype=object), ["rare", "regular"]))[0]:
        if c not in covered:
            cand = active_users[np.argsort(-S[active_users, c])[:25]]
            pairs[int(rng.choice(cand))].add(int(c))

    diffs = courses["difficulty"].values
    rows = []
    span = (WINDOW_END - WINDOW_START).total_seconds()
    dev_main = rng.choice(["мобільний", "комп'ютер", "планшет"], nU, p=[.52, .40, .08])
    for u, cs in pairs.items():
        for c in cs:
            a = theta[u, a_idx[c]] + (0.4 * theta[u, a2_idx[c]] if a2_idx[c] >= 0 else 0)
            a = min(a, 1.0)
            t0 = WINDOW_START + pd.Timedelta(seconds=float(rng.random() * (span - 40 * 86400)))
            n_click = 1 + rng.poisson(.25 + .9 * a)
            offs = np.sort(np.concatenate([[0], np.cumsum(rng.exponential(2 * 86400, n_click - 1))]))
            evs = []

            def cap(t):
                return min(t, WINDOW_END - pd.Timedelta(minutes=float(rng.integers(1, 600))))

            for o in offs:
                evs.append(("click", cap(t0 + pd.Timedelta(seconds=float(o))), None, None))
            fav = rng.random() < .05 + .30 * a
            enr = rng.random() < .04 + .28 * a
            t_enr = t0 + pd.Timedelta(seconds=float(rng.exponential(3 * 86400)))
            if fav:
                evs.append(("favorite", cap(t0 + pd.Timedelta(seconds=float(rng.exponential(6 * 3600)))), None, None))
            if enr:
                evs.append(("enroll", cap(t_enr), None, None))
            s = q[c] + .8 * (a - .35) + rng.normal(0, .55)
            t_after = t_enr if enr else t0
            if rng.random() < (.70 if enr else .03):
                delay = rng.uniform(20, 120) * 86400 if enr else rng.uniform(1, 10) * 86400
                evs.append(("rating", cap(t_after + pd.Timedelta(seconds=float(delay))), int(np.clip(round(s), 1, 5)), None))
            if rng.random() < (.25 if enr else .03):
                pool = COMMENTS_POS if s >= 3.6 else (COMMENTS_NEG if s < 2.6 else COMMENTS_NEU)
                txt = rng.choice(pool).format(t=base_topic[c])
                delay = rng.uniform(15, 130) * 86400 if enr else rng.uniform(1, 12) * 86400
                evs.append(("comment", cap(t_after + pd.Timedelta(seconds=float(delay))), None, txt))
            for typ, t, rt, cm in evs:
                dwell = int(np.clip(rng.lognormal(np.log(75), .8), 5, 1800)) if typ == "click" else None
                dev = dev_main[u] if rng.random() < .7 else rng.choice(["мобільний", "комп'ютер", "планшет"])
                rows.append((users.user_id[u], courses.course_id[c], typ, t, rt, cm, dwell, dev))
    inter = pd.DataFrame(rows, columns=["user_id", "course_id", "interaction_type", "timestamp", "rating",
                                        "comment_text", "dwell_time_sec", "device"])
    inter["timestamp"] = pd.to_datetime(inter["timestamp"]).dt.floor("s").dt.strftime("%Y-%m-%d %H:%M:%S")
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    inter.insert(0, "interaction_id", np.arange(1, len(inter) + 1))
    return inter


# =====================================================================================
# 4. ШУМ: пропуски (≤ 5 %) та дублікати (≤ 2 %)
# =====================================================================================
def add_missing(df, rates, mask_rows=None):
    df = df.copy()
    for col, p in rates.items():
        cand = df.index if mask_rows is None else df.index[mask_rows(df, col)]
        k = int(round(p * len(df)))
        idx = rng.choice(cand, min(k, len(cand)), replace=False)
        df[col] = df[col].astype("object")
        df.loc[idx, col] = np.nan
    return df


def add_duplicates(df, frac):
    k = int(round(frac * len(df)))
    dup = df.sample(k, random_state=SEED)
    out = pd.concat([df, dup]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out


def main():
    users, utruth, theta, pref_fmt, gpa = make_users()
    courses, ctruth, pop, q, area, area2, base_topic = make_courses()
    inter = make_interactions(users, utruth, ctruth, courses, theta, pref_fmt, gpa, pop, q, area, area2, base_topic)

    users_n = add_missing(users, {"age": .03, "gender": .02, "city": .03, "gpa": .048, "study_year": .02,
                                  "preferred_format": .04, "career_goal": .04, "interests": .03})
    users_n = add_duplicates(users_n, .015)
    courses_n = add_missing(courses, {"description": .02, "instructor": .03, "language": .015, "difficulty": .03,
                                      "credits": .02, "weekly_hours": .04, "keywords": .03, "max_students": .048})
    courses_n = add_duplicates(courses_n, .015)
    inter_n = add_missing(inter, {"timestamp": .01, "device": .03})
    click_only = lambda d, c: d["interaction_type"] == "click"
    inter_n = add_missing(inter_n, {"dwell_time_sec": .02 * len(inter) / (inter.interaction_type == "click").sum()},
                          mask_rows=click_only)
    inter_n = add_duplicates(inter_n, .015)

    users_n.to_csv(RAW / "users_raw.csv", index=False, encoding="utf-8-sig")
    courses_n.to_csv(RAW / "courses_raw.csv", index=False, encoding="utf-8-sig")
    inter_n.to_csv(RAW / "interactions_raw.csv", index=False, encoding="utf-8-sig")
    utruth.to_csv(TRUTH / "users_truth.csv", index=False, encoding="utf-8-sig")
    ctruth.to_csv(TRUTH / "courses_truth.csv", index=False, encoding="utf-8-sig")

    print("Користувачі:", users_n.shape, "| Курси:", courses_n.shape, "| Взаємодії:", inter_n.shape)
    print(inter.interaction_type.value_counts().to_string())
    print("унікальних пар user-course:", inter.groupby(["user_id", "course_id"]).ngroups)


if __name__ == "__main__":
    main()
