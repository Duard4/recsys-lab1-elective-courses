# -*- coding: utf-8 -*-
"""Крок 3. 5 метрик схожості користувачів → агломеративна кластеризація → характеристики кластерів → ТОП-10 «ядро» кожного кластера."""
from collections import Counter
from recsys_lib import *

u, c, it, X, R, ut, ct = load_clean()
n = len(u)
truth = ut.loc[u.user_id, "dominant_area"].to_numpy()          # прихована мітка — лише для оцінювання
NUM_G, NUM_E = ["age", "gpa", "study_year"], ["age_w", "gpa", "study_year"]
CAT = ["gender", "city", "faculty", "degree_level", "preferred_format", "career_goal"]
MIN_COURSES = 8     # метрики на взаємодіях — лише для користувачів з ≥ 8 курсів; решті кластер призначається за kNN

# --- метрики: U1–U2 за анкетою, U3 за заявленими інтересами, U4–U5 за взаємодіями
B, _ = sets_to_binary(u["interests"])
has = np.asarray(B.sum(1)).ravel() > 0
el = np.where(u["n_courses"] >= MIN_COURSES)[0]
Xe = X[el]
cat_dummies = pd.get_dummies(c["category"])
areas = list(cat_dummies.columns)
metrics = {
    "U1": dict(name="Говера (анкета)", D=gower_distance(u, NUM_G, CAT), idx=np.arange(n)),
    "U2": dict(name="Евклідова (анкета)", D=euclidean_distance(u, NUM_E, CAT), idx=np.arange(n)),
    "U3": dict(name="Жаккара (теги інтересів)", D=jaccard_distance_binary(B[has]), idx=np.where(has)[0]),
    "U4": dict(name="Косинусна (взаємодії)", D=cosine_distance_dense(idf_weight(Xe)), idx=el),
    "U5": dict(name="Єнсена–Шеннона (профіль тем)", D=js_distance(np.asarray(Xe @ cat_dummies.to_numpy(float)) + 0.01), idx=el),
}
# домінантна тема взаємодій (для опису кластерів)
u["area_by_interactions"] = pd.Series([None] * n, dtype=object)
any_ = np.where(u["n_courses"] > 0)[0]
u.loc[any_, "area_by_interactions"] = [areas[i] for i in np.asarray(X[any_] @ cat_dummies.to_numpy(float)).argmax(1)]

# резервна метрика для користувачів без даних: 0.5·Говер + 0.5·Жаккар(інтереси)
D_fallback = 0.5 * metrics["U1"]["D"] + 0.5 * jaccard_distance_binary(B)
# незалежна ознака: факультет не входить до U3–U5 (але входить до U1–U2)
fac = pd.factorize(u["faculty"])[0]
summary, labels, labels_e = cluster_metrics(metrics, n, D_fallback, truth, indep={m: fac for m in ("U3", "U4", "U5")})
summary.to_csv(RES / "users_summary.csv", index=False, encoding=ENC)
pd.DataFrame({"user_id": u.user_id, **{f"cluster_{m}": labels[m] for m in metrics}}).to_csv(RES / "users_cluster_labels.csv", index=False, encoding=ENC)

# --- характеристики кластерів та ТОП-10
def profile(lab):
    df = u.assign(cl=lab)
    rows = []
    for cl in sorted(set(lab)):
        s = df[df.cl == cl]
        tags = Counter(t for x in s["interests"] for t in x.split(";") if t)
        ai, fc = s["area_by_interactions"].value_counts(normalize=True), s["faculty"].value_counts(normalize=True)
        rows.append(dict(cluster=cl, size=len(s), mean_age=round(s.age.mean(), 1), mean_gpa=round(s.gpa.mean(), 1),
                         share_master_pct=round(100 * (s.degree_level == "магістр").mean()), share_male_pct=round(100 * (s.gender == "чоловіча").mean()),
                         top_faculty=fc.index[0].replace("Факультет ", "").replace("факультет", "").strip(), top_faculty_pct=round(100 * fc.iloc[0]),
                         top_interest_tags=", ".join(t for t, _ in tags.most_common(3)),
                         top_area_by_interactions=ai.index[0] if len(ai) else "—", top_area_pct=round(100 * ai.iloc[0]) if len(ai) else np.nan,
                         mean_events=round(s.n_events.mean(), 1)))
    return pd.DataFrame(rows)

ids = u["user_id"].to_numpy()
for mid, m in metrics.items():
    profile(labels[mid]).to_csv(RES / f"users_profile_{mid}.csv", index=False, encoding=ENC)
    top, mats = top_core(m["D"], labels_e[mid], ids[m["idx"]], 10)
    top.merge(u[["user_id", "faculty", "age", "gpa", "interests", "n_events", "activity_group"]], left_on="id", right_on="user_id").drop(columns="user_id") \
        .sort_values(["cluster", "rank"]).to_csv(RES / f"users_top10_{mid}.csv", index=False, encoding=ENC)
    plot_top10_heatmaps(mats, f"{mid}. {m['name']}: схожість ТОП-10 користувачів кожного кластера", FIG / f"users_top10_{mid}.png")

short = {a: a.split(" та ")[0].split(",")[0][:22] for a in areas}
composition_heatmap(metrics, labels, areas + ["немає взаємодій"], u["area_by_interactions"].fillna("немає взаємодій"), short,
                    "Склад кластерів користувачів за домінантною темою їхніх взаємодій (частка кластера, %)", FIG / "users_composition.png")
print("done users")
