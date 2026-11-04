# -*- coding: utf-8 -*-
"""Крок 4. 5 метрик схожості курсів → агломеративна кластеризація → характеристики кластерів → ТОП-10 «ядро» кожного кластера."""
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from recsys_lib import *

u, c, it, X, R, ut, ct = load_clean()
n = len(c)
truth = ct.loc[c.course_id, "area"].to_numpy()                  # прихована мітка — лише для оцінювання
XC = X.T.tocsr()                                                 # курси × користувачі (сила взаємодії)
NUM_G = ["credits", "difficulty", "weekly_hours", "max_students"]
CAT = ["category", "language", "format", "semester", "assessment"]
MIN_USERS = 10      # метрики на взаємодіях — лише для курсів з ≥ 10 користувачів; решті кластер призначається за TF-IDF (kNN)

# --- метрики: C1 за текстом, C2 за атрибутами, C3 за ключовими словами, C4–C5 за взаємодіями
T = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True).fit_transform(c["text_clean"])
assert (np.asarray(T.sum(1)).ravel() > 0).all()                  # кожен курс має непорожній TF-IDF-вектор → резервна метрика C1 визначена для всіх
Bk, _ = sets_to_binary(c["keywords"])
has_kw = np.asarray(Bk.sum(1)).ravel() > 0
el = np.where(c["n_users"] >= MIN_USERS)[0]
metrics = {
    "C1": dict(name="Косинусна (TF-IDF описів)", D=cosine_distance_dense(T), idx=np.arange(n)),
    "C2": dict(name="Говера (атрибути)", D=gower_distance(c, NUM_G, CAT), idx=np.arange(n)),
    "C3": dict(name="Жаккара (ключові слова)", D=jaccard_distance_binary(Bk[has_kw]), idx=np.where(has_kw)[0]),
    "C4": dict(name="Косинусна (взаємодії)", D=cosine_distance_dense(idf_weight(XC[el])), idx=el),
    "C5": dict(name="Жаккара (спільні користувачі)", D=jaccard_distance_binary(XC[el] > 0), idx=el),
}
D_fallback = metrics["C1"]["D"]                                  # для курсів без даних — текстова схожість (C1)
# незалежна ознака: category входить лише до C2 (Говер), тому для C1, C3–C5 ARI/NMI відносно категорії є незалежною перевіркою
cat_codes = pd.factorize(c["category"])[0]
summary, labels, labels_e = cluster_metrics(metrics, n, D_fallback, truth, indep={m: cat_codes for m in ("C1", "C3", "C4", "C5")})
summary.to_csv(RES / "courses_summary.csv", index=False, encoding=ENC)
pd.DataFrame({"course_id": c.course_id, **{f"cluster_{m}": labels[m] for m in metrics}}).to_csv(RES / "courses_cluster_labels.csv", index=False, encoding=ENC)

# --- характеристики кластерів та ТОП-10
def profile(lab):
    df = c.assign(cl=lab)
    rows = []
    for cl in sorted(set(lab)):
        s = df[df.cl == cl]
        cat, pg = s["category"].value_counts(normalize=True), s["popularity_group"].value_counts(normalize=True)
        kws = Counter(t for x in s["keywords"] for t in x.split(";") if t)
        rows.append(dict(cluster=cl, size=len(s), top_category=cat.index[0], top_category_pct=round(100 * cat.iloc[0]),
                         mean_difficulty=round(s.difficulty.mean(), 2), mean_credits=round(s.credits.mean(), 2),
                         top_keywords=", ".join(k for k, _ in kws.most_common(4)), mean_users=round(s.n_users.mean(), 1),
                         mean_rating=round(s.avg_rating.mean(), 2), share_popular_pct=round(100 * pg.get("популярні", 0)),
                         share_rare_pct=round(100 * pg.get("рідкісні", 0)), share_new_pct=round(100 * pg.get("нові (0)", 0))))
    return pd.DataFrame(rows)

ids = c["course_id"].to_numpy()
for mid, m in metrics.items():
    profile(labels[mid]).to_csv(RES / f"courses_profile_{mid}.csv", index=False, encoding=ENC)
    top, mats = top_core(m["D"], labels_e[mid], ids[m["idx"]], 10)
    top.merge(c[["course_id", "title", "category", "n_users", "avg_rating", "popularity_group"]], left_on="id", right_on="course_id").drop(columns="course_id") \
        .sort_values(["cluster", "rank"]).to_csv(RES / f"courses_top10_{mid}.csv", index=False, encoding=ENC)
    plot_top10_heatmaps(mats, f"{mid}. {m['name']}: схожість ТОП-10 курсів кожного кластера", FIG / f"courses_top10_{mid}.png")

cats = sorted(c["category"].unique())
short = {a: a.split(" та ")[0].split(",")[0][:22] for a in cats}
composition_heatmap(metrics, labels, cats, c["category"], short, "Склад кластерів курсів за тематичною категорією (частка кластера, %)", FIG / "courses_composition.png")

# --- діагностика нічиїх у ТОП-10 (Жаккар за ключовими словами): скільки курсів мають однаковий набір ключових слів
ks = c.keywords[c.keywords != ""].apply(lambda s: tuple(sorted(s.split(";"))))
print("унікальних наборів ключових слів:", ks.nunique(), "із", len(ks), "курсів")
print("done courses")
