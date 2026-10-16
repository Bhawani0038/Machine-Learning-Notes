# Module 8: Unsupervised Learning (Student Notes)

Seven short lessons. Each idea is explained **once**, in the file where it belongs. Later files only remind you and show what is new.

## Roadmap

| File | Topics covered | What is new here |
|------|----------------|------------------|
| [01_Unsupervised_Learning_and_Clustering.md](01_Unsupervised_Learning_and_Clustering.md) | 1. What is Unsupervised Learning, 2. Supervised vs Unsupervised, 3. Clustering | The big picture, plus the customer dataset used in files 02 to 04 |
| [02_KMeans.md](02_KMeans.md) | 4. K-Means, 6. Centroids, 7. Distance Calculation, 5. K-Means Algorithm | The first clustering algorithm, plus inertia |
| [03_Choosing_K.md](03_Choosing_K.md) | 8. Choosing K, 9. Elbow Method, 10. Silhouette Score | How to decide the number of clusters |
| [04_Hierarchical_Clustering.md](04_Hierarchical_Clustering.md) | 11. Hierarchical Clustering, 12. Dendrogram | Clustering without fixing K in advance |
| [05_DBSCAN.md](05_DBSCAN.md) | 13. DBSCAN | Density-based clusters, noise points, and a comparison of all three algorithms |
| [06_Dimensionality_Reduction_and_PCA.md](06_Dimensionality_Reduction_and_PCA.md) | 14. Dimensionality Reduction, 15. PCA, 16. PCA Intuition | Fewer columns, same information |
| [07_PCA_in_Python.md](07_PCA_in_Python.md) | 17. PCA in Python | Hands-on PCA on a real dataset, then a module recap |

**Why the order in two places differs slightly from the topic list:**
- In file 02, Centroids and Distance come *before* the K-Means Algorithm, because the algorithm uses both.
- In file 06, the plain definition of PCA comes first, then the visual intuition, then the maths.

## How to use the files

1. Keep the `images/` folder **next to** the `.md` files. The pictures use relative paths such as `images/03_kmeans_steps.png`, so VS Code's Markdown preview (`Ctrl+Shift+V`) will show them.
2. Run the code in **one Python notebook or script, top to bottom**. File 01 creates the variable `X` (customer data). Files 02 to 04 reuse it, so you will not see the dataset code again. Files 05 and 07 introduce their own datasets.
3. `make_figures.py` recreates every image, in case you want to change colours or sizes.

Install what you need once:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

## Quick memory line for the whole module

> Unsupervised learning = no answers given. **Clustering** groups similar rows. **Dimensionality reduction** shrinks the number of columns.
