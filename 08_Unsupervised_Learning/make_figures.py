"""Regenerates every image used in the Module 8 notes.
Run:  python make_figures.py      (creates/overwrites the images/ folder)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.datasets import make_blobs, make_moons, load_wine
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram

OUT = "images"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
COL = ["#2a78d6", "#e8742c", "#2ca58d", "#b04fc4", "#d9b21c", "#7a7a7a"]

def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}", bbox_inches="tight", facecolor="white")
    plt.close()

# ---------- running dataset (same as the notes) ----------
centers = [[25, 20], [25, 80], [55, 50], [85, 20], [85, 80]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=7, random_state=42)

# 1. supervised vs unsupervised
rng = np.random.RandomState(1)
a = rng.randn(30, 2) * 0.7 + [0, 0]; b = rng.randn(30, 2) * 0.7 + [3, 3]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
ax[0].scatter(*a.T, c=COL[0], label="Cat"); ax[0].scatter(*b.T, c=COL[1], label="Dog")
ax[0].set_title("Supervised: every point has a label"); ax[0].legend()
ax[1].scatter(*np.vstack([a, b]).T, c="#777777")
ax[1].set_title("Unsupervised: no labels, find the structure")
for x in ax: x.set_xlabel("feature 1")
ax[0].set_ylabel("feature 2")
save("01_supervised_vs_unsupervised.png")

# 2. clustering idea (before / after)
km = KMeans(5, n_init=10, random_state=42).fit(X)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
ax[0].scatter(*X.T, c="#777777", s=18); ax[0].set_title("Raw customers (no labels)")
for k in range(5): ax[1].scatter(*X[km.labels_ == k].T, c=COL[k], s=18)
ax[1].set_title("After clustering: 5 groups found")
for x in ax: x.set_xlabel("Annual income (k$)")
ax[0].set_ylabel("Spending score (1-100)")
save("02_clustering_before_after.png")

# 3. K-Means steps (manual implementation)
rng = np.random.RandomState(3)
cent = X[rng.choice(len(X), 5, replace=False)].copy()
def assign(c): return np.argmin(((X[:, None] - c[None]) ** 2).sum(2), 1)
def update(l): return np.array([X[l == k].mean(0) if (l == k).any() else cent[k] for k in range(5)])
panels = []
lab = np.zeros(len(X), int)
panels.append(("Step 1: pick 5 random centroids", cent.copy(), None))
lab = assign(cent); panels.append(("Step 2: assign each point to nearest centroid", cent.copy(), lab.copy()))
cent = update(lab); panels.append(("Step 3: move centroids to cluster means", cent.copy(), lab.copy()))
for _ in range(10):
    lab = assign(cent); cent = update(lab)
panels.append(("Repeat 2-3 until nothing changes", cent.copy(), lab.copy()))
fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
for a_, (t, c, l) in zip(ax.ravel(), panels):
    if l is None: a_.scatter(*X.T, c="#aaaaaa", s=14)
    else:
        for k in range(5): a_.scatter(*X[l == k].T, c=COL[k], s=14)
    a_.scatter(*c.T, c="black", marker="X", s=140, edgecolor="white", linewidth=1.2)
    a_.set_title(t, fontsize=9.5)
save("03_kmeans_steps.png")

# 4. distance
fig, ax = plt.subplots(figsize=(5, 4))
p, q = np.array([1, 1]), np.array([5, 4])
ax.plot(*zip(p, q), c=COL[1], lw=2.5, label="Euclidean = 5.0 (straight line)")
ax.plot([1, 5, 5], [1, 1, 4], c=COL[0], lw=2.5, ls="--", label="Manhattan = 7 (along the grid)")
ax.scatter(*np.array([p, q]).T, c="black", zorder=3, s=50)
ax.text(0.8, 0.5, "A (1,1)"); ax.text(5.1, 4.1, "B (5,4)")
ax.text(5.1, 2.4, "3"); ax.text(3, 0.6, "4")
ax.set_xlim(0, 6.5); ax.set_ylim(0, 5); ax.legend(loc="upper left", fontsize=8)
ax.set_title("Two ways to measure distance")
save("04_distance.png")

# 5/6. elbow + silhouette
ks = range(2, 11)
inertias = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
sils = [silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit_predict(X)) for k in ks]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(list(ks), inertias, "o-", c=COL[0]); ax.axvline(5, ls="--", c="gray")
ax.annotate("elbow", (5, inertias[3]), xytext=(6.2, inertias[3] + 25000),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Number of clusters K"); ax.set_ylabel("Inertia"); ax.set_title("Elbow method")
save("05_elbow.png")
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(list(ks), sils, "o-", c=COL[2]); ax.axvline(list(ks)[int(np.argmax(sils))], ls="--", c="gray")
ax.set_xlabel("Number of clusters K"); ax.set_ylabel("Average silhouette score")
ax.set_title("Silhouette score (higher is better)")
save("06_silhouette.png")
print("inertias", [round(i) for i in inertias]); print("sils", [round(s, 3) for s in sils])

# 7. toy dendrogram + real dendrogram
toy = np.array([[1], [2], [6], [7], [15]])
Z = linkage(toy, "single")
fig, ax = plt.subplots(figsize=(5, 3.6))
dendrogram(Z, labels=list("ABCDE"), ax=ax, color_threshold=0, above_threshold_color="#2a78d6")
ax.set_ylabel("Distance at which clusters merge"); ax.set_title("Toy dendrogram (5 points)")
save("07_dendrogram_toy.png")
Zr = linkage(X, "ward")
fig, ax = plt.subplots(figsize=(8, 4))
dendrogram(Zr, truncate_mode="lastp", p=12, ax=ax, color_threshold=0, above_threshold_color="#555555")
cut = 0.5 * (Zr[-5, 2] + Zr[-4, 2])
ax.axhline(cut, c=COL[1], ls="--"); ax.text(0.2, cut + 8, "cut here -> 5 clusters", color=COL[1])
ax.set_xlabel("Cluster (number of points in brackets)"); ax.set_ylabel("Merge distance (Ward)")
ax.set_title("Dendrogram of the customer data (last 12 merges shown)")
save("08_dendrogram_customers.png")

# 9. DBSCAN point types
rng = np.random.RandomState(7)
core = rng.randn(14, 2) * 0.45
pts = np.vstack([core, [[1.5, 0.4]], [[-1.5, -0.5]], [[3.2, 3.0]]])
eps = 0.9
fig, ax = plt.subplots(figsize=(5.6, 5))
ax.scatter(*core.T, c=COL[0], s=45, label="Core point (enough neighbours)")
ax.scatter(*pts[14:16].T, c=COL[4], s=60, label="Border point (near a core, few neighbours)")
ax.scatter(*pts[16:].T, c="black", marker="x", s=70, label="Noise (isolated)")
ax.add_patch(Circle(core[0], eps, fill=False, ec=COL[1], lw=2)); ax.annotate("eps", core[0] + [0.5, 0.75], color=COL[1])
ax.set_aspect("equal"); ax.legend(fontsize=7.5, loc="lower right"); ax.set_title("DBSCAN: three kinds of points")
save("09_dbscan_point_types.png")

# 10. moons: K-Means vs DBSCAN
M, _ = make_moons(300, noise=0.07, random_state=42)
Ms = StandardScaler().fit_transform(M)
klab = KMeans(2, n_init=10, random_state=42).fit_predict(Ms)
dlab = DBSCAN(eps=0.3, min_samples=5).fit_predict(Ms)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
ax[0].scatter(*Ms.T, c=[COL[l] for l in klab], s=16); ax[0].set_title("K-Means (K=2) cuts the moons wrongly")
ax[1].scatter(*Ms.T, c=[COL[l] if l >= 0 else "black" for l in dlab], s=16)
ax[1].set_title("DBSCAN follows the shape")
save("10_kmeans_vs_dbscan.png")
print("dbscan labels", np.unique(dlab, return_counts=True))

# 11. k-distance plot
d, _ = NearestNeighbors(n_neighbors=5).fit(Ms).kneighbors(Ms)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(np.sort(d[:, -1]), c=COL[0]); ax.set_xlabel("Points sorted by distance"); ax.set_ylabel("Distance to 5th neighbour")
ax.set_title("k-distance plot: pick eps near the bend"); ax.axhline(0.3, ls="--", c="gray")
save("11_dbscan_k_distance.png")

# 12. PCA intuition
rng = np.random.RandomState(0)
P = rng.randn(120, 2) @ np.array([[2.2, 1.4], [0, 0.6]])
P = P - P.mean(0)
pca2 = PCA(2).fit(P)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
ax[0].scatter(*P.T, c="#999999", s=15)
for i, (v, ev) in enumerate(zip(pca2.components_, pca2.explained_variance_)):
    ax[0].annotate("", v * 1.8 * np.sqrt(ev), (0, 0), arrowprops=dict(arrowstyle="->", lw=2.5, color=COL[i]))
    ax[0].text(*(v * 2.05 * np.sqrt(ev)), f"PC{i+1}", color=COL[i], weight="bold")
ax[0].set_xlim(-6.5, 7.5); ax[0].set_ylim(-4, 5); ax[0].set_aspect("equal"); ax[0].set_title("PC1 points along the spread,\nPC2 is perpendicular")
v1 = pca2.components_[0]; proj = (P @ v1)[:, None] * v1
ax[1].scatter(*P.T, c="#cccccc", s=12)
ax[1].scatter(*proj.T, c=COL[0], s=14, label="Projected on PC1 (keeps most spread)")
for a_, b_ in list(zip(P, proj))[:25]: ax[1].plot(*zip(a_, b_), c="#bbbbbb", lw=0.6)
ax[1].set_aspect("equal"); ax[1].legend(fontsize=7.5, loc="lower right")
ax[1].set_title("Shadow of the data on PC1")
save("12_pca_intuition.png")

# 13. good vs bad direction
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
v2 = pca2.components_[1]
for a_, v, t, c in [(ax[0], v1, "Project on PC1: points stay spread out", COL[2]),
                    (ax[1], v2, "Project on PC2: points pile up, information lost", COL[3])]:
    z = (P @ v)
    a_.scatter(z, np.zeros_like(z), c=c, s=14, alpha=0.7); a_.set_yticks([]); a_.set_title(t, fontsize=9.5)
    a_.set_xlim(-9, 9)
save("13_pca_good_vs_bad_direction.png")

# 14/15/16. wine PCA
W = load_wine(); Xw = StandardScaler().fit_transform(W.data)
pw = PCA().fit(Xw); Z2 = pw.transform(Xw)[:, :2]
fig, ax = plt.subplots(figsize=(5.6, 4.4))
for i, n in enumerate(W.target_names): ax.scatter(*Z2[W.target == i].T, c=COL[i], s=20, label=f"class {i}")
ax.set_xlabel(f"PC1 ({pw.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pw.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(); ax.set_title("Wine data: 13 features squeezed into 2")
save("14_pca_wine_scatter.png")
ev = pw.explained_variance_ratio_; cs = np.cumsum(ev)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].bar(range(1, 14), ev, color=COL[0]); ax[0].set_xlabel("Component"); ax[0].set_ylabel("Explained variance ratio"); ax[0].set_title("Scree plot")
ax[1].plot(range(1, 14), cs, "o-", c=COL[2]); ax[1].axhline(0.95, ls="--", c="gray"); ax[1].set_xlabel("Number of components")
ax[1].set_ylabel("Cumulative variance"); ax[1].set_title("Cumulative explained variance")
save("15_pca_explained_variance.png")
print("wine ev", np.round(ev, 3), "cum", np.round(cs, 3), "n95", int(np.argmax(cs >= 0.95)) + 1)

# 17. dimensionality reduction idea: 3D -> 2D shadow
rng = np.random.RandomState(5)
base = rng.randn(150, 2) * [3, 1.2]
T = np.c_[base, 0.15 * base[:, 0] + 0.1 * rng.randn(150)]
fig = plt.figure(figsize=(9, 4))
a3 = fig.add_subplot(121, projection="3d"); a3.scatter(*T.T, c=COL[0], s=12); a3.set_title("3 features")
a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z")
a2 = fig.add_subplot(122); a2.scatter(T[:, 0], T[:, 1], c=COL[0], s=12); a2.set_title("2 features (little lost)")
a2.set_xlabel("PC1"); a2.set_ylabel("PC2")
save("00_dimensionality_reduction_idea.png")
