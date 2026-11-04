# -*- coding: utf-8 -*-
"""Спільна бібліотека: метрики схожості, ієрархічна кластеризація, ТОП-10, візуалізація."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
PROC, RES, FIG, TRUTH = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures", ROOT / "data" / "ground_truth"
ENC = "utf-8-sig"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 150, "axes.titlesize": 10, "axes.labelsize": 9})
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D", "#EECA3B", "#79706E", "#FF9DA6", "#2F4B7C", "#A0CBE8"]


def load_clean():
    u = pd.read_csv(PROC / "users_clean.csv", encoding=ENC)
    c = pd.read_csv(PROC / "courses_clean.csv", encoding=ENC)
    it = pd.read_csv(PROC / "interactions_clean.csv", encoding=ENC)
    u["interests"] = u["interests"].fillna("")
    for col in ("keywords", "description", "text_clean"):
        c[col] = c[col].fillna("")
    X = sp.load_npz(PROC / "strength_matrix.npz").tocsr()
    R = sp.load_npz(PROC / "rating_matrix.npz").tocsr()
    ut = pd.read_csv(TRUTH / "users_truth.csv", encoding=ENC).set_index("user_id")
    ct = pd.read_csv(TRUTH / "courses_truth.csv", encoding=ENC).set_index("course_id")
    return u, c, it, X, R, ut, ct


# ------------------------------------------------------------------------------------------
# МЕТРИКИ СХОЖОСТІ (усі повертають МАТРИЦЮ ВІДСТАНЕЙ D ∈ [0, 1], симетричну, з нульовою діагоналлю)
# схожість S = 1 − D
# ------------------------------------------------------------------------------------------
def _finalize(D):
    D = np.clip((D + D.T) / 2, 0, None)
    np.fill_diagonal(D, 0.0)
    return D


def gower_distance(df, num_cols, cat_cols, unknown="невідомо"):
    """Відстань Говера для змішаних даних. Значення «невідомо» / NaN не враховуються в парі."""
    n = len(df)
    num_sum, w_sum = np.zeros((n, n)), np.zeros((n, n))
    for col in num_cols:
        x = df[col].to_numpy(float)
        rng_ = np.nanmax(x) - np.nanmin(x) or 1.0
        valid = ~np.isnan(x)
        d = np.abs(x[:, None] - x[None, :]) / rng_
        w = (valid[:, None] & valid[None, :]).astype(float)
        num_sum += np.nan_to_num(d) * w
        w_sum += w
    for col in cat_cols:
        x = df[col].astype(object).to_numpy()
        valid = (x != unknown) & pd.notna(x)
        d = (x[:, None] != x[None, :]).astype(float)
        w = (valid[:, None] & valid[None, :]).astype(float)
        num_sum += d * w
        w_sum += w
    return _finalize(num_sum / np.maximum(w_sum, 1e-9))


def euclidean_distance(df, num_cols, cat_cols):
    """Евклідова відстань на стандартизованих числових ознаках + one-hot категоріальних (вага 1/√2)."""
    Z = StandardScaler().fit_transform(df[num_cols].to_numpy(float))
    oh = pd.get_dummies(df[cat_cols].astype(str)).to_numpy(float) / np.sqrt(2)
    F = np.hstack([Z, oh])
    sq = (F ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * F @ F.T, 0)
    D = np.sqrt(D2)
    return _finalize(D / D.max())


def jaccard_distance_binary(B):
    """Відстань Жаккара для бінарної розрідженої матриці B (рядки — об'єкти)."""
    B = sp.csr_matrix(B, dtype=np.float32)
    inter = (B @ B.T).toarray()
    sz = np.asarray(B.sum(1)).ravel()
    union = sz[:, None] + sz[None, :] - inter
    return _finalize(1 - inter / np.maximum(union, 1e-9))


def sets_to_binary(series, sep=";"):
    """Стовпець із «тег1;тег2» → бінарна матриця (multi-hot)."""
    sets = series.fillna("").apply(lambda s: [t for t in s.split(sep) if t])
    vocab = {t: i for i, t in enumerate(sorted({t for l in sets for t in l}))}
    rows = [i for i, l in enumerate(sets) for _ in l]
    cols = [vocab[t] for l in sets for t in l]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vocab))), vocab


def cosine_distance_dense(A):
    """Косинусна відстань між рядками матриці (розріджена або щільна)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    nrm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
    nrm[nrm == 0] = 1
    A = sp.diags(1 / nrm) @ A
    return _finalize(1 - (A @ A.T).toarray())


def idf_weight(X, axis_users=True):
    """IDF-зважування: рідкісні спільні об'єкти дають більший внесок, ніж популярні."""
    Xb = X.copy()
    df_ = np.asarray((Xb > 0).sum(0)).ravel()
    idf = np.log((Xb.shape[0] + 1) / (df_ + 1)) + 1
    return Xb @ sp.diags(idf)


def js_distance(P, eps=1e-12):
    """Відстань Єнсена–Шеннона (sqrt JS-дивергенції, основа 2) між рядками-розподілами P."""
    P = P / P.sum(1, keepdims=True)
    H = lambda M: -(M * np.log2(np.where(M > 0, M, 1))).sum(-1)
    Hp = H(P)
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n):
        M = (P[i][None, :] + P) / 2
        D[i] = H(M) - (Hp[i] + Hp) / 2
    return _finalize(np.sqrt(np.clip(D, 0, 1)))


# ------------------------------------------------------------------------------------------
# КЛАСТЕРИЗАЦІЯ: агломеративна ієрархічна (scipy.linkage на матриці відстаней)
# ------------------------------------------------------------------------------------------
def merge_small(D, lab, min_size):
    """Поглинання кластерів-«викидів»: кластер менший за min_size приєднується до найближчого
    (за середньою відстанню між членами) кластера. Повторюється, доки всі кластери ≥ min_size."""
    lab = lab.copy()
    while True:
        ids, sizes = np.unique(lab, return_counts=True)
        if len(ids) <= 2 or sizes.min() >= min_size:
            break
        small = ids[sizes.argmin()]
        ms = lab == small
        best, bd = None, np.inf
        for o in ids:
            if o == small:
                continue
            d = D[np.ix_(ms, lab == o)].mean()
            if d < bd:
                best, bd = o, d
        lab[ms] = best
    return pd.factorize(lab, sort=True)[0]


def hierarchical(D, ks=range(5, 13), methods=("average", "complete"), min_frac=0.02):
    """Єдиний протокол агломеративної ієрархічної кластеризації на матриці відстаней D:
    перебираються два зв'язки (average, complete) та k = 5…12; обирається розбиття з найбільшим силуетом серед
    тих, де найменший кластер ≥ 2 % вибірки (якщо таких немає — найкраще за силуетом, а кластери-«викиди» < 2 %
    поглинаються найближчими).  Повертає (labels, k, method, info)."""
    n = len(D)
    cond = squareform(D, checks=False)
    rows, Zs = [], {}
    for method in methods:
        Zs[method] = linkage(cond, method=method)
        for k in ks:
            lab = fcluster(Zs[method], k, "maxclust")
            if len(np.unique(lab)) < k:
                continue
            sizes = np.bincount(lab)[1:]
            rows.append(dict(method=method, k=k, silhouette=silhouette_score(D, lab, metric="precomputed"),
                             ok=bool(sizes.min() >= min_frac * n)))
    q = pd.DataFrame(rows)
    cand = q[q.ok] if q.ok.any() else q
    best = cand.sort_values("silhouette", ascending=False).iloc[0]
    lab = fcluster(Zs[best.method], int(best.k), "maxclust") - 1
    merged = not best.ok
    if merged:
        lab = merge_small(D, lab, int(np.ceil(min_frac * n)))
    info = dict(k_cut=int(best.k), silhouette=float(silhouette_score(D, lab, metric="precomputed")), merged_outliers=merged)
    return lab, len(np.unique(lab)), best.method, info


def assign_cold_start(labels_eligible, elig_idx, n_total, D_fallback, knn=10):
    """Об'єктам без даних для метрики призначаємо кластер за kNN на резервній метриці (Говер / TF-IDF)."""
    lab = np.full(n_total, -1)
    lab[elig_idx] = labels_eligible
    rest = np.where(lab < 0)[0]
    for i in rest:
        d = D_fallback[i, elig_idx]
        nn = np.argsort(d)[:knn]
        lab[i] = np.bincount(labels_eligible[nn]).argmax()
    return lab


def top_core(D, labels, ids, n_top=10):
    """ТОП-n «ядро» кластера: об'єкти з найменшою середньою відстанню до решти членів кластеру.
    Повертає список DataFrame (по кластерах) та 10×10 матриці схожості."""
    out, mats = [], {}
    S = 1 - D
    for cl in sorted(set(labels)):
        idx = np.where(labels == cl)[0]
        sub = S[np.ix_(idx, idx)]
        mean_sim = (sub.sum(1) - 1) / max(len(idx) - 1, 1)
        order = np.argsort(-mean_sim, kind="stable")[:n_top]
        top_idx = idx[order]
        m = S[np.ix_(top_idx, top_idx)]
        pair_mean = (m.sum() - len(top_idx)) / max(len(top_idx) * (len(top_idx) - 1), 1)
        for r, (j, ms) in enumerate(zip(top_idx, mean_sim[order]), 1):
            out.append(dict(cluster=cl, rank=r, id=ids[j], mean_similarity_to_cluster=round(float(ms), 4),
                            top10_mean_pairwise_similarity=round(float(pair_mean), 4)))
        mats[cl] = (ids[top_idx], m)
    return pd.DataFrame(out), mats


def plot_top10_heatmaps(mats, title, path, ncols=5):
    k = len(mats)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (cl, (ids, m)) in zip(axes.ravel(), mats.items()):
        ax.axis("on")
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"Кластер {cl}  (n={len(ids)})", fontsize=9)
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels([i[1:] for i in ids], rotation=90, fontsize=6)
        ax.set_yticklabels([i[1:] for i in ids], fontsize=6)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="схожість")
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ari_nmi(labels, truth):
    return adjusted_rand_score(truth, labels), normalized_mutual_info_score(truth, labels)


def cluster_metrics(metrics, n, D_fallback, truth, indep=None, knn=10):
    """Єдиний цикл для всіх метрик: кластеризація за протоколом hierarchical(), кластер для об'єктів без даних
    (kNN на резервній метриці), показники якості. indep[id] — спостережувана ознака, яка НЕ входила до метрики
    (для незалежної перевірки; NMI між кластерами та цією ознакою).  Повертає (summary DataFrame, labels dict, lab_e dict)."""
    indep = indep or {}
    rows, labels_all, labels_e = [], {}, {}
    for mid, m in metrics.items():
        lab_e, k, method, info = hierarchical(m["D"])
        lab = assign_cold_start(lab_e, m["idx"], n, D_fallback, knn) if len(m["idx"]) < n else lab_e
        labels_all[mid], labels_e[mid] = lab, lab_e
        sizes = np.bincount(lab_e)
        off = m["D"][np.triu_indices(len(m["D"]), 1)]
        ari, nmi = ari_nmi(lab_e, truth[m["idx"]])
        nmi_ind = normalized_mutual_info_score(indep[mid][m["idx"]], lab_e) if mid in indep else np.nan
        rows.append(dict(metric_id=mid, metric=m["name"], eligible=len(m["idx"]), linkage=method, k=k, outliers_merged=info["merged_outliers"],
                         silhouette=round(info["silhouette"], 3), min_cluster=int(sizes.min()), max_cluster=int(sizes.max()),
                         share_pairs_no_overlap=round(float((off >= 0.9999).mean()), 3), ARI_hidden=round(ari, 3), NMI_hidden=round(nmi, 3),
                         NMI_unused_feature=round(nmi_ind, 3) if nmi_ind == nmi_ind else np.nan))
        print(mid, m["name"], method, k, "sil %.3f" % info["silhouette"], sorted(sizes, reverse=True), "ARI %.3f" % ari)
    return pd.DataFrame(rows), labels_all, labels_e


def composition_heatmap(metrics, labels, cats, cat_values, short, title, path):
    """Склад кластерів за категорією (частка об'єктів кластера, %): по панелі на метрику."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (mid, m) in zip(axes.ravel(), metrics.items()):
        ax.axis("on")
        tab = pd.crosstab(labels[mid], cat_values, normalize="index").reindex(columns=cats, fill_value=0)
        ax.imshow(tab.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(tab))); ax.set_yticklabels([f"кл. {i}" for i in tab.index], fontsize=7)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels([short.get(a, a) for a in cats], rotation=60, ha="right", fontsize=6.5)
        ax.set_title(f"{mid}. {m['name']}", fontsize=9)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                if tab.values[i, j] >= .15:
                    ax.text(j, i, f"{tab.values[i, j] * 100:.0f}", ha="center", va="center", fontsize=6, color="white" if tab.values[i, j] > .55 else "black")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
