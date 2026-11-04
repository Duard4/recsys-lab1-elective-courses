# -*- coding: utf-8 -*-
"""Крок 2. Попередня обробка даних (users, courses, interactions).

Вхід : data/raw/*.csv
Вихід: data/processed/users_clean.csv, courses_clean.csv, interactions_clean.csv,
       data/processed/strength_matrix.npz (користувач × курс, «сила» взаємодії),
       results/preprocessing_log.json, results/prep_examples_*.csv
"""
import json
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

ROOT = Path(__file__).parent
RAW, PROC, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"
ENC = "utf-8-sig"
REF_DATE = pd.Timestamp("2026-09-30")
log = {}

users_raw = pd.read_csv(RAW / "users_raw.csv", encoding=ENC)
courses_raw = pd.read_csv(RAW / "courses_raw.csv", encoding=ENC)
inter_raw = pd.read_csv(RAW / "interactions_raw.csv", encoding=ENC)
log["raw_rows"] = dict(users=len(users_raw), courses=len(courses_raw), interactions=len(inter_raw))
log["raw_missing_pct"] = {
    "users": (users_raw.isna().mean() * 100).round(2).loc[lambda s: s > 0].to_dict(),
    "courses": (courses_raw.isna().mean() * 100).round(2).loc[lambda s: s > 0].to_dict(),
    "interactions": (inter_raw.isna().mean() * 100).round(2).loc[lambda s: s > 0].to_dict(),
}
log["raw_missing_cells_pct_users"] = round(float(users_raw.isna().sum().sum() / users_raw.size * 100), 2)
log["raw_missing_cells_pct_courses"] = round(float(courses_raw.isna().sum().sum() / courses_raw.size * 100), 2)

# ------------------------------------------------------------------------------------------
# 1. КОРИСТУВАЧІ
# ------------------------------------------------------------------------------------------
u = users_raw.copy()
log["users_exact_duplicates"] = int(u.duplicated().sum())
u = u.drop_duplicates().drop_duplicates("user_id", keep="first").reset_index(drop=True)
u["registration_date"] = pd.to_datetime(u["registration_date"])

# валідація діапазонів (некоректні → NaN, далі імпутація)
bad_age = ~u["age"].between(16, 60) & u["age"].notna()
u.loc[bad_age, "age"] = np.nan
u["gpa"] = u["gpa"].clip(0, 100)

# імпутація: числові — медіана у відповідній підгрупі, категоріальні — окрема категорія «невідомо»
u["age_missing"] = u["age"].isna().astype(int)
u["gpa_missing"] = u["gpa"].isna().astype(int)
u["study_year"] = u["study_year"].fillna(u.groupby("degree_level")["study_year"].transform("median"))
u["age"] = u["age"].fillna(u.groupby(["degree_level", "study_year"])["age"].transform("median")).fillna(u["age"].median())
u["gpa"] = u["gpa"].fillna(u.groupby("faculty")["gpa"].transform("median")).fillna(u["gpa"].median())
for col in ["gender", "city", "career_goal", "preferred_format"]:
    u[col] = u[col].fillna("невідомо")
u["interests"] = u["interests"].fillna("")
u["age"] = u["age"].round().astype(int)
u["study_year"] = u["study_year"].round().astype(int)
u["tenure_days"] = (REF_DATE - u["registration_date"]).dt.days
# вінсоризація віку (викиди, напр. студенти-«дорослі»): 1–99 перцентилі
lo, hi = u["age"].quantile([.01, .99])
u["age_w"] = u["age"].clip(lo, hi)
u["n_interests"] = u["interests"].apply(lambda s: len([t for t in s.split(";") if t]))
u["registration_date"] = u["registration_date"].dt.strftime("%Y-%m-%d")
log["users_after_dedup"] = len(u)

# ------------------------------------------------------------------------------------------
# 2. КУРСИ
# ------------------------------------------------------------------------------------------
c = courses_raw.copy()
log["courses_exact_duplicates"] = int(c.duplicated().sum())
c = c.drop_duplicates().drop_duplicates("course_id", keep="first").reset_index(drop=True)

c["difficulty"] = c["difficulty"].fillna(c.groupby("category")["difficulty"].transform("median")).round().astype(int)
c["credits"] = c["credits"].fillna(c["credits"].mode()[0]).astype(int)
# навантаження виводимо з кредитів (зв'язок credits ↔ weekly_hours)
c["weekly_hours"] = c["weekly_hours"].fillna(c.groupby("credits")["weekly_hours"].transform("median")).round().astype(int)
c["max_students"] = c["max_students"].fillna(c["max_students"].median()).astype(int)
c["language"] = c["language"].fillna(c["language"].mode()[0])
c["instructor"] = c["instructor"].fillna("невідомо")
c["keywords"] = c["keywords"].fillna("")
c["description_missing"] = c["description"].isna().astype(int)
c["description"] = c["description"].fillna("")
c["publication_date"] = pd.to_datetime(c["publication_date"])
c["age_days"] = (REF_DATE - c["publication_date"]).dt.days
c["publication_date"] = c["publication_date"].dt.strftime("%Y-%m-%d")

# очищення тексту для TF-IDF
STOP = set("""і й та а але або чи що щоб як у в на з із до по за від для про при над під між без через
також це цей ця ці той така такі його її їх їм він вона вони ми ви я ти бути є буде були було
наприклад тощо усі всі кожен один одна одне основні особлива увага приділяється студенти студентів студентам
курс курсу курсі курсом дисципліна дисципліни дисципліною межах програма формат оцінювання залік іспит проєкт
онлайн офлайн гібридний потрібні потрібно підготовки попередньої знання базові зі суміжних""".split())


def clean_text(s):
    s = re.sub(r"[^\w\s'’-]|\d", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in STOP and len(w) > 2)


c["text_clean"] = (c["title"].fillna("") + " " + c["description"]).apply(clean_text)
log["courses_after_dedup"] = len(c)

# ------------------------------------------------------------------------------------------
# 3. ВЗАЄМОДІЇ
# ------------------------------------------------------------------------------------------
it = inter_raw.copy()
log["interactions_exact_duplicates"] = int(it.duplicated().sum())
it = it.drop_duplicates().drop_duplicates("interaction_id").reset_index(drop=True)
it["timestamp"] = pd.to_datetime(it["timestamp"], format="ISO8601")
# цілісність: лише відомі user_id / course_id
ok = it["user_id"].isin(u["user_id"]) & it["course_id"].isin(c["course_id"])
log["interactions_orphans_removed"] = int((~ok).sum())
it = it[ok].copy()
# логічні перевірки
it = it[it["rating"].isna() | it["rating"].between(1, 5)]
# пропуски: timestamp → медіана часу взаємодій користувача; device → мода користувача; dwell → медіана курсу
log["timestamp_missing"] = int(it["timestamp"].isna().sum())
med_ts = it.groupby("user_id")["timestamp"].transform("median")
it["timestamp"] = it["timestamp"].fillna(med_ts).fillna(it["timestamp"].median())
log["device_missing"] = int(it["device"].isna().sum())
mode_dev = it.groupby("user_id")["device"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
it["device"] = it["device"].fillna(it["user_id"].map(mode_dev)).fillna("невідомо")
clicks = it["interaction_type"] == "click"
log["dwell_missing_clicks"] = int(it.loc[clicks, "dwell_time_sec"].isna().sum())
med_dw = it[clicks].groupby("course_id")["dwell_time_sec"].transform("median")
it.loc[clicks, "dwell_time_sec"] = it.loc[clicks, "dwell_time_sec"].fillna(med_dw).fillna(it["dwell_time_sec"].median())
it = it.sort_values("timestamp").reset_index(drop=True)

# «сила» взаємодії (неявний зворотний зв'язок)
W = {"favorite": 3.0, "enroll": 5.0, "comment": 1.0}
pair = it.groupby(["user_id", "course_id"])
n_click = it[it.interaction_type == "click"].groupby(["user_id", "course_id"]).size()
parts = {"click": np.log1p(n_click)}
for t, w in W.items():
    parts[t] = it[it.interaction_type == t].groupby(["user_id", "course_id"]).size().clip(upper=1) * w
rt = it[it.interaction_type == "rating"].groupby(["user_id", "course_id"])["rating"].mean()
parts["rating"] = rt / 5 * 2.0
strength = pd.concat(parts, axis=1).fillna(0)
strength["strength"] = strength.sum(axis=1)
strength["n_events"] = pair.size().reindex(strength.index)
strength["rating"] = rt.reindex(strength.index)
strength = strength.reset_index()

uidx = {k: i for i, k in enumerate(u["user_id"])}
cidx = {k: i for i, k in enumerate(c["course_id"])}
rows = strength["user_id"].map(uidx).to_numpy()
cols = strength["course_id"].map(cidx).to_numpy()
X = sp.csr_matrix((strength["strength"].to_numpy(), (rows, cols)), shape=(len(u), len(c)))
sp.save_npz(PROC / "strength_matrix.npz", X)
mask = strength["rating"].notna().to_numpy()
R = sp.csr_matrix((strength.loc[mask, "rating"].to_numpy(), (rows[mask], cols[mask])), shape=X.shape)
sp.save_npz(PROC / "rating_matrix.npz", R)

# ------------------------------------------------------------------------------------------
# 4. АГРЕГАТИ ДЛЯ ОПИСУ КЛАСТЕРІВ
# ------------------------------------------------------------------------------------------
ev = it.groupby("user_id").size()
u["n_events"] = u["user_id"].map(ev).fillna(0).astype(int)
u["n_courses"] = u["user_id"].map(strength.groupby("user_id").size()).fillna(0).astype(int)
n_enr = it[it.interaction_type == "enroll"].groupby("user_id").size()
u["n_enroll"] = u["user_id"].map(n_enr).fillna(0).astype(int)
ur = it[it.interaction_type == "rating"].groupby("user_id")["rating"].mean()
u["avg_rating_given"] = u["user_id"].map(ur)

cev = it.groupby("course_id").size()
c["n_events"] = c["course_id"].map(cev).fillna(0).astype(int)
c["n_users"] = c["course_id"].map(strength.groupby("course_id").size()).fillna(0).astype(int)
c["n_enroll"] = c["course_id"].map(it[it.interaction_type == "enroll"].groupby("course_id").size()).fillna(0).astype(int)
c["n_favorite"] = c["course_id"].map(it[it.interaction_type == "favorite"].groupby("course_id").size()).fillna(0).astype(int)
c["n_comments"] = c["course_id"].map(it[it.interaction_type == "comment"].groupby("course_id").size()).fillna(0).astype(int)
cr = it[it.interaction_type == "rating"].groupby("course_id")["rating"].agg(["mean", "count"])
c["avg_rating"] = c["course_id"].map(cr["mean"])
c["n_ratings"] = c["course_id"].map(cr["count"]).fillna(0).astype(int)

# спостережувані групи активності / популярності (для звіту)
def user_group(r):
    if r["n_events"] == 0: return "нові"
    return None
q_lo, q_hi = u.loc[u.n_events > 0, "n_events"].quantile([.35, .85])
u["activity_group"] = np.where(u.n_events == 0, "нові (0)",
                        np.where(u.n_events <= q_lo, "пасивні", np.where(u.n_events >= q_hi, "активні", "звичайні")))
cq_lo, cq_hi = c.loc[c.n_events > 0, "n_events"].quantile([.35, .85])
c["popularity_group"] = np.where(c.n_events == 0, "нові (0)",
                          np.where(c.n_events <= cq_lo, "рідкісні", np.where(c.n_events >= cq_hi, "популярні", "звичайні")))
log["activity_thresholds"] = dict(passive_max=float(q_lo), active_min=float(q_hi))
log["popularity_thresholds"] = dict(rare_max=float(cq_lo), popular_min=float(cq_hi))

# ------------------------------------------------------------------------------------------
# 5. ЗБЕРЕЖЕННЯ + ПРИКЛАДИ «ДО / ПІСЛЯ»
# ------------------------------------------------------------------------------------------
u.to_csv(PROC / "users_clean.csv", index=False, encoding=ENC)
c.to_csv(PROC / "courses_clean.csv", index=False, encoding=ENC)
it.assign(timestamp=it["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(PROC / "interactions_clean.csv", index=False, encoding=ENC)
strength.to_csv(PROC / "user_course_strength.csv", index=False, encoding=ENC)

pick_u = pd.concat([users_raw[users_raw[col].isna() & users_raw.drop(columns=col).notna().all(axis=1)].drop_duplicates("user_id").head(1)
                    for col in ["age", "gpa", "gender", "interests"]])
miss_u = pick_u
ex_u = pd.concat([miss_u.assign(стан="до"), u[u.user_id.isin(miss_u.user_id)][miss_u.columns].assign(стан="після")])
ex_u["ord"] = ex_u["стан"].map({"до": 0, "після": 1})
ex_u = ex_u.sort_values(["user_id", "ord"]).drop(columns="ord")
ex_u.to_csv(RES / "prep_examples_users.csv", index=False, encoding=ENC)
cols_c = ["course_id", "title", "credits", "difficulty", "weekly_hours", "instructor", "max_students", "keywords"]
miss_c = pd.concat([courses_raw[courses_raw[col].isna() & courses_raw[[x for x in cols_c if x != col]].notna().all(axis=1)].drop_duplicates("course_id").head(1)
                    for col in ["difficulty", "weekly_hours", "instructor", "keywords"]])
ex_c = pd.concat([miss_c[cols_c].assign(стан="до"), c[c.course_id.isin(miss_c.course_id)][cols_c].assign(стан="після")])
ex_c["ord"] = ex_c["стан"].map({"до": 0, "після": 1})
ex_c = ex_c.sort_values(["course_id", "ord"]).drop(columns="ord")
ex_c.to_csv(RES / "prep_examples_courses.csv", index=False, encoding=ENC)
sample_txt = c[c.description != ""].sample(3, random_state=1)[["course_id", "title", "description", "text_clean"]]
sample_txt.to_csv(RES / "prep_examples_text.csv", index=False, encoding=ENC)

log["final"] = dict(users=len(u), courses=len(c), interactions=len(it), pairs=int(X.nnz),
                    users_with_interactions=int((u.n_events > 0).sum()), courses_with_interactions=int((c.n_events > 0).sum()),
                    ratings_pairs=int(R.nnz), matrix_density_pct=round(X.nnz / X.shape[0] / X.shape[1] * 100, 3))
log["interaction_types"] = it["interaction_type"].value_counts().to_dict()
log["user_groups"] = u["activity_group"].value_counts().to_dict()
log["course_groups"] = c["popularity_group"].value_counts().to_dict()
json.dump(log, open(RES / "preprocessing_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
print(json.dumps(log, ensure_ascii=False, indent=1, default=float))
