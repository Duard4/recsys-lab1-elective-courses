# -*- coding: utf-8 -*-
"""Крок 5. Підсумкова діаграма «силует vs ARI/NMI» та статистика датасету для звіту (results/report_data.json)."""
import json
from recsys_lib import *

u, c, it, X, R, ut, ct = load_clean()
su = pd.read_csv(RES / "users_summary.csv", encoding=ENC)
sc = pd.read_csv(RES / "courses_summary.csv", encoding=ENC)

# --- підсумкова діаграма: внутрішня якість (силует) та відповідність прихованій темі (ARI) для кожної метрики
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
for ax, df, title in zip(axes, [su, sc], ["Користувачі", "Курси"]):
    x = np.arange(len(df))
    ax.bar(x - .2, df["silhouette"], .4, color=PALETTE[0], label="силует (внутрішня якість)")
    ax.bar(x + .2, df["ARI_hidden"].clip(lower=0), .4, color=PALETTE[1], label="ARI з прихованою темою")
    for i, r in df.iterrows():
        ax.text(i - .2, r.silhouette + .015, f"{r.silhouette:.2f}".replace(".", ","), ha="center", fontsize=7)
        ax.text(i + .2, max(r.ARI_hidden, 0) + .015, f"{r.ARI_hidden:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df["metric_id"]); ax.set_title(title); ax.set_ylim(0, 0.85)
axes[0].legend(frameon=False, fontsize=8, loc="upper left"); axes[0].set_ylabel("значення показника")
fig.tight_layout(); fig.savefig(FIG / "quality_summary.png", bbox_inches="tight"); plt.close(fig)

# --- статистика датасету
S = {}
ev = c["n_events"].sort_values(ascending=False).to_numpy()
S["top10pct_courses_share_events"] = round(float(ev[:int(.1 * len(ev))].sum() / ev.sum()), 3)
gini = lambda a: float((2 * np.arange(1, len(a) + 1) - len(a) - 1).dot(np.sort(a)) / (len(a) * np.sort(a).sum()))
S["gini_courses"], S["gini_users"] = round(gini(ev), 3), round(gini(u["n_events"].to_numpy()), 3)
S["course_groups"] = c.groupby("popularity_group").agg(courses=("course_id", "size"), mean_events=("n_events", "mean"), mean_rating=("avg_rating", "mean")).round(2).reset_index().to_dict("records")
S["user_groups"] = u.groupby("activity_group").agg(users=("user_id", "size"), mean_events=("n_events", "mean"), mean_courses=("n_courses", "mean")).round(1).reset_index().to_dict("records")
rare = c[(c.popularity_group == "рідкісні") & (c.n_ratings >= 1)]
pop = c[(c.popularity_group == "популярні") & (c.n_ratings >= 1)]
S["rare_rating_ge4_pct"] = round(100 * float((rare.avg_rating >= 4).mean()), 1)
S["popular_rating_ge4_pct"] = round(100 * float((pop.avg_rating >= 4).mean()), 1)
S["interaction_types"] = it["interaction_type"].value_counts().to_dict()
S["unique_pairs"], S["density_pct"] = int(X.nnz), round(100 * X.nnz / (X.shape[0] * X.shape[1]), 2)
S["users_desc"] = dict(age_mean=round(u.age.mean(), 1), gpa_mean=round(u.gpa.mean(), 1), level=u.degree_level.value_counts().to_dict(), gender=u.gender.value_counts().to_dict())
S["courses_desc"] = dict(category=c.category.value_counts().to_dict(), language=c.language.value_counts().to_dict(), format=c.format.value_counts().to_dict(),
                         words_mean=round(float(c.description.str.split().str.len().mean()), 0))
ks = c.keywords[c.keywords != ""].apply(lambda s: tuple(sorted(s.split(";"))))
S["keyword_sets_unique"], S["keyword_sets_courses"] = int(ks.nunique()), int(len(ks))
t3 = pd.read_csv(RES / "courses_top10_C3.csv", encoding=ENC)
S["c3_clusters_with_top10_sim_1"] = int((t3.drop_duplicates("cluster")["top10_mean_pairwise_similarity"] >= 0.9999).sum())
S["c3_clusters"] = int(t3["cluster"].nunique())
log = json.load(open(RES / "preprocessing_log.json", encoding="utf-8"))
json.dump(dict(stats=S, log=log), open(RES / "report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(S, ensure_ascii=False, default=float)[:1500]); print(log.keys())

# --- зведення для звіту: таблиці результатів та характеристики кластерів репрезентативних метрик
rep = dict(users_summary=su.where(su.notna(), None).to_dict("records"), courses_summary=sc.where(sc.notna(), None).to_dict("records"), clusters={})
pc_ = lambda x: f"{x:.0f}".replace(".", ",")
for mid in ["U3", "U5", "C1", "C4"]:
    isu = mid.startswith("U")
    prof = pd.read_csv(RES / f"{'users' if isu else 'courses'}_profile_{mid}.csv", encoding=ENC)
    top = pd.read_csv(RES / f"{'users' if isu else 'courses'}_top10_{mid}.csv", encoding=ENC)
    rows = []
    for _, p in prof.iterrows():
        t = top[top.cluster == p.cluster].sort_values("rank")
        coh = float(t.top10_mean_pairwise_similarity.iloc[0])
        if isu:
            desc = (f"факультет: {p.top_faculty} ({p.top_faculty_pct:.0f} %); теги: {p.top_interest_tags}; тема взаємодій: {p.top_area_by_interactions} ({p.top_area_pct:.0f} %); "
                    f"вік {p.mean_age}, GPA {p.mean_gpa}, магістрів {p.share_master_pct:.0f} %, чоловіків {p.share_male_pct:.0f} %; подій у середньому {p.mean_events}")
            ex = ", ".join(t.id.head(3))
        else:
            desc = (f"{p.top_category} ({p.top_category_pct:.0f} %); ключові слова: {p.top_keywords}; складність {p.mean_difficulty}, рейтинг {p.mean_rating}; "
                    f"популярних {p.share_popular_pct:.0f} %, рідкісних {p.share_rare_pct:.0f} %, нових {p.share_new_pct:.0f} %")
            ex = "; ".join(f"{a} «{b}»" for a, b in zip(t.id.head(3), t.title.head(3)))
        rows.append(dict(cl=int(p.cluster), size=int(p["size"]), desc=desc, top3=ex, cohesion=round(coh, 2)))
    rep["clusters"][mid] = rows
d = json.load(open(RES / "report_data.json", encoding="utf-8"))
d["rep"] = rep
json.dump(d, open(RES / "report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("report_data.json updated")
