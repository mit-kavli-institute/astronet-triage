import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
import umap
import hdbscan

from astronet.util import config_util
from astronet.astro_cnn_model import input_ds
from astronet.preprocess import preprocess


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; stable against outliers."""
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m))
    return (x - m) / (s + 1e-8)


@dataclass
class ClusterParams:
    # Embedding
    pca_components: int = 64
    whiten: bool = True
    cosine_normalize: bool = True
    outlier_quantile: float = 0.995  # drop top-q by L2 norm before fitting

    # HDBSCAN core knobs
    min_cluster_size: int = 10
    min_samples: int = 10
    cluster_selection_epsilon: float = 0.01
    cluster_selection_method: str = "eom"  # "eom" or "leaf"

    # UMAP (viz only)
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.05

    # Post-assign grey points
    postassign_prob_floor: float = 0.35


class Clustering:
    """
    End-to-end light-curve clustering helper for ExoDash.

    Pipeline:
      TFRecords -> id_to_view -> robust_z -> PCA (+whiten) -> (optional L2 normalize)
      -> HDBSCAN -> UMAP (viz only)
      -> optional post-assign noise points to nearest existing clusters (prob≥threshold)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        eval_files: List[str],
        config_path: str,
        view_key: str = "global_view",
        params: ClusterParams = ClusterParams(),
    ):
        self.df = df
        self.data_dir = data_dir
        self.eval_files = eval_files
        self.config_path = config_path
        self.view_key = view_key
        self.params = params

        # Filled during fit()
        self.tce_table: Optional[pd.DataFrame] = None
        self.id_to_view: Dict[int, np.ndarray] = {}
        self.ids: List[int] = []
        self.X: Optional[np.ndarray] = None            # raw (robust-z) views, aligned with self.ids
        self.keep_mask: Optional[np.ndarray] = None    # mask after outlier removal
        self.ids_c: List[int] = []                     # ids after mask
        self.Z: Optional[np.ndarray] = None            # PCA features
        self.Z_cos: Optional[np.ndarray] = None        # L2-normalized features (if enabled)
        self.clusterer: Optional[hdbscan.HDBSCAN] = None
        self.labels_: Optional[np.ndarray] = None      # base labels (-1 = noise)
        self.labels_post_: Optional[np.ndarray] = None # after post-assign (optional)
        self.probs_: Optional[np.ndarray] = None       # membership probabilities (n x K)
        self.viz_: Optional[np.ndarray] = None         # UMAP 2D
        self.nn_: Optional[NearestNeighbors] = None    # neighbors on Z or Z_cos

    # ---------- Data loading ----------

    def _load_tce_table(self) -> pd.DataFrame:
        if self.tce_table is None:
            self.tce_table = pd.read_csv(self.tce_csv, header=0, low_memory=False)
        return self.tce_table

    def _get_lightcurve(self, astro_id: int, aperture: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Raw light curve (time, flux) via astronet preprocess; not used for clustering directly."""
        aperture_key_map = {
            "s": "SAP_FLUX_SML",
            "m": "SAP_FLUX_MID",
            "l": "SAP_FLUX_LAG",
            None: "SAP_FLUX",
        }
        matching = self.df[self.df["Astro ID"] == astro_id]
        try:
            _, tce = next(matching.iterrows())
        except StopIteration as e:
            raise ValueError(f"Astro ID not found: {astro_id}") from e
        if "MinT" not in tce:
            tce["MinT"] = -np.inf
        if "MaxT" not in tce:
            tce["MaxT"] = np.inf
        lc = preprocess.read_and_process_light_curve(
            self.data_dir,
            aperture_key_map[aperture],
            tce.File,
            tce.MinT,
            tce.MaxT,
        )
        return lc

    def _build_eval_set(self):
        cfg = config_util.load_config(self.config_path)
        pairs = []

        for fp in self.eval_files:
            if ":" in fp:
                name, pattern = fp.split(":", 1)
            elif len(self.eval_files) == 1:
                name, pattern = "eval", fp
            else:
                raise ValueError("Multiple datasets must be named as name:file_pattern")
            ds = input_ds.build_eval_dataset(
                file_pattern=pattern,
                input_config=cfg.inputs,
                batch_size=cfg.hparams.batch_size,
                include_identifiers=True,
                include_labels=False,
            )
            pairs.append((name, ds))
        return pairs

    @staticmethod
    def _build_id_to_view(
        ds,
        view_key: str,
        filter_ids: Optional[Iterable[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Build astro_id -> view mapping from a tf.data.Dataset yielding (features, identifiers).

        Args:
            ds: dataset of (features, identifiers) batches
            view_key: key in `features` to read the view (shape: [B, ...])
            filter_ids: optional iterable of IDs to keep; if None/empty, keep all

        Returns:
            Dict[int, np.ndarray]: mapping from astro_id to its view row
        """
        mapping: Dict[int, np.ndarray] = {}
        filter_set = set(filter_ids) if filter_ids is not None else None
        print(f'Filter set: {filter_set}')

        for features, identifiers in ds:
            ids = identifiers.numpy()
            print(f'Ids: {len(ids)}')
            views = features[view_key].numpy()  # shape: (B, ...)

            for astro_id, v in zip(ids, views):
                # normalize type
                if isinstance(astro_id, (bytes, bytearray)):
                    astro_id = int(astro_id.decode())
                else:
                    astro_id = int(astro_id)

                # apply filter if provided
                if filter_set is None or astro_id in filter_set:
                    mapping[astro_id] = v

        return mapping
    def load_views(self, ids_to_filter=None) -> None:
        """Populate self.id_to_view from TFRecords."""
        id_to_view_all: Dict[int, np.ndarray] = {}
        for _, ds in self._build_eval_set():
            id_to_view_all.update(self._build_id_to_view(ds, self.view_key, ids_to_filter))
        self.id_to_view = id_to_view_all

        # Align X/ids arrays
        self.ids = sorted(self.id_to_view.keys())
        X = [robust_z(self.id_to_view[i]) for i in self.ids]
        self.X = np.vstack(X).astype(np.float32)

    # ---------- Fitting (embedding → clustering → viz) ----------

    def fit_pca(self) -> None:
        """Run PCA embedding, HDBSCAN clustering, UMAP viz, neighbor index, and soft membership."""
        assert self.X is not None, "Call load_views() first."

        # Optional outlier removal by L2
        norms = np.linalg.norm(self.X, axis=1)
        keep = norms < np.quantile(norms, self.params.outlier_quantile)
        self.keep_mask = keep
        Xc = self.X[keep]
        self.ids_c = self.ids # keep

        # # PCA
        self.pca = PCA(n_components=self.params.pca_components, random_state=0).fit(self.X)
        self.Z = self.pca.transform(self.X)
        self.nn_ = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(self.Z)

    def fit_clusters(self) -> None:
        # Cosine emphasis (shape)
        Z_cos = normalize(self.Z, norm="l2", axis=1) if self.params.cosine_normalize else self.Z

        # HDBSCAN
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.params.min_cluster_size,
            min_samples=self.params.min_samples,
            cluster_selection_epsilon=self.params.cluster_selection_epsilon,
            cluster_selection_method=self.params.cluster_selection_method,
            prediction_data=True,
            metric="euclidean",  # euclidean on L2-normalized ≈ cosine
        ).fit(Z_cos)
        labels = self.clusterer.labels_
        self.labels_ = labels

        # Soft memberships for post-assign
        self.probs_ = hdbscan.all_points_membership_vectors(self.clusterer)

        # UMAP (viz only)
        u = umap.UMAP(
            n_neighbors=self.params.umap_n_neighbors,
            min_dist=self.params.umap_min_dist,
            metric="euclidean",
            random_state=0,
        )
        self.viz_ = u.fit_transform(Z_cos)

        # Neighbor index (for quick “similar” queries)
        self.Z_cos = Z_cos

        # Optional post-assign of noise
        #self.labels_post_ = self.post_assign_noise(self.params.postassign_prob_floor)

    def post_assign_noise(self, p_min: float = 0.35) -> np.ndarray:
        """Assign -1 points to most likely existing cluster if prob ≥ p_min (no new clusters)."""
        assert self.labels_ is not None and self.probs_ is not None
        new_labels = self.labels_.copy()
        maxp = self.probs_.max(axis=1)
        best = self.probs_.argmax(axis=1)
        mask = (self.labels_ == -1) & (maxp >= p_min)
        new_labels[mask] = best[mask]
        return new_labels

    # ---------- UI / ExoDash helpers ----------

    def to_dataframe(
        self,
        properties_csv: Optional[str] = None,
        use_post_labels: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame you can display/filter in ExoDash:
        columns: astro_id, umap_x, umap_y, label, prob, label_post, cluster_size, ...
        Optionally joins an external CSV on 'astro_id'.
        """
        assert self.viz_ is not None and self.labels_ is not None and self.probs_ is not None

        labels = self.labels_
        labels_post = self.labels_post_ if (use_post_labels and self.labels_post_ is not None) else labels
        prob = self.probs_.max(axis=1)  # max membership prob

        df = pd.DataFrame({
            "astro_id": self.ids_c,
            "umap_x": self.viz_[:, 0],
            "umap_y": self.viz_[:, 1],
            "label": labels,
            "label_post": labels_post,
            "prob": prob,
        })

        # cluster sizes for convenience
        sizes = df.groupby("label_post").size().to_dict()
        df["cluster_size"] = [sizes.get(lb, np.nan) for lb in df["label_post"]]

        if properties_csv is not None and os.path.exists(properties_csv):
            props = pd.read_csv(properties_csv)
            # Expect 'astro_id' to exist
            if "astro_id" not in props.columns:
                # try common variants
                for c in ["Astro ID", "AstroID", "astroId"]:
                    if c in props.columns:
                        props = props.rename(columns={c: "astro_id"})
                        break
            df = df.merge(props, on="astro_id", how="left")

        return df

    def neighbors_by_id(self, astro_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Top-k nearest neighbors in embedding space."""
        assert self.nn_ is not None and self.Z is not None
        try:
            i = self.ids.index(astro_id)
        except ValueError:
            raise ValueError(f"astro_id {astro_id} not in clustered set (maybe filtered as outlier).")
        d, idx = self.nn_.kneighbors(self.Z[i].reshape(1, -1), n_neighbors=k)
        return [(self.ids[j], float(dist)) for j, dist in zip(idx[0], d[0])]

    def show_cluster(self, label: int, n: int = 9, rng: Optional[np.random.Generator] = None):
        """Quick-look a cluster's members by plotting their phase views."""
        assert self.labels_ is not None
        rng = rng or np.random.default_rng(0)
        idxs = [i for i, lb in enumerate(self.labels_) if lb == label]
        if not idxs:
            print("Empty cluster.")
            return
        take = rng.choice(idxs, size=min(n, len(idxs)), replace=False)
        for i in take:
            astro_id = self.ids_c[i]
            self._plot_view(self.id_to_view[astro_id], astro_id)

    @staticmethod
    def _plot_view(view: np.ndarray, astro_id: int) -> plt.Figure:
        phase = np.linspace(0, 1, len(view), endpoint=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(phase, view, marker='.', linestyle='-')
        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"Astro ID {astro_id}")
        ax.invert_yaxis()
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot(
        self,
        use_post_labels: bool = True,
        highlight_ids: Optional[List[int]] = None,
        alpha_bg: float = 0.12,          # how faint the background is
        s_bg: int = 10,                  # background marker size
        s_hi: int = 28,                  # highlighted marker size
        annotate: bool = False,          # label highlighted points with astro_id
    ) -> plt.Figure:
        """UMAP scatter: all points faint, selected astro_ids highlighted."""
        assert self.viz_ is not None and self.labels_ is not None

        labels = self.labels_post_ if (use_post_labels and self.labels_post_ is not None) else self.labels_

        # Stable color mapping across the WHOLE dataset
        all_non_noise = [u for u in np.unique(labels) if u != -1]
        palette = plt.cm.get_cmap("tab20", max(1, len(all_non_noise)))

        def color_for(lb, alpha=1.0):
            return (0.7, 0.7, 0.7, 0.25 * alpha)
            if lb == -1:  # noise
                return (0.7, 0.7, 0.7, 0.25 * alpha)
            idx = all_non_noise.index(lb) if lb in all_non_noise else 0
            r, g, b, _ = palette(idx % palette.N)
            return (r, g, b, alpha)

        fig, ax = plt.subplots(figsize=(7, 6))

        # 1) Background: all points (dimmed)
        bg_colors = [color_for(lb, alpha=alpha_bg) for lb in labels]
        ax.scatter(self.viz_[:, 0], self.viz_[:, 1], s=s_bg, c=bg_colors, linewidths=0, zorder=1)

        # 2) Highlights: filtered ids (full color, thicker edge)
        if highlight_ids:
            mask = np.isin(self.ids_c, list(highlight_ids))
            if np.any(mask):
                hi_colors = [color_for(lb, alpha=1.0) for lb in labels[mask]]
                ax.scatter(self.viz_[mask, 0], self.viz_[mask, 1],
                        s=s_hi, c=hi_colors, edgecolors="k", linewidths=0.6, zorder=3)

                if annotate:
                    for x, y, aid in zip(self.viz_[mask, 0], self.viz_[mask, 1], np.array(self.ids_c)[mask]):
                        ax.annotate(str(aid), (x, y), xytext=(3, 3), textcoords="offset points",
                                    fontsize=8, color="k", zorder=4)

                # small legend chip for "highlighted"
                legend_handles = [Line2D([0], [0], marker='o', linestyle='',
                                        markersize=np.sqrt(s_hi), markerfacecolor=(0.3,0.3,0.3,0.9),
                                        markeredgecolor='k', label='Highlighted')]
                ax.legend(handles=legend_handles, loc='best', frameon=True)
            else:
                ax.text(0.02, 0.98, "No astro_id matched the filter", transform=ax.transAxes,
                        va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, lw=0))

        ax.set_title("Light-curve islands (PCA→HDBSCAN, UMAP viz)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        return fig
    
    def show_nearest_neighbors(
        self,
        astro_id: int,
        df,
        n: int = 6,
        include_self: bool = False,
        layout: Literal["grid", "list"] = "grid",
        cols: int = 3,
        filter_ids=None,
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Plot the N nearest neighbors of `astro_id` with distances.

        Args:
            astro_id: query ID present in `self.ids`
            n: number of neighbors to display (excluding self unless include_self=True)
            include_self: include the query itself (distance ~0) in the results
            layout: "grid" -> single Figure with subplots, "list" -> list of Figures
            cols: number of columns for the grid layout

        Returns:
            - If layout="grid": a matplotlib Figure containing all neighbors
            - If layout="list": a list of matplotlib Figures (one per neighbor)
        """
        assert self.nn_ is not None and self.Z is not None, "Call fit() first."
        assert self.id_to_view, "Call load_views() first."
        try:
            i = self.ids.index(astro_id)
        except ValueError:
            raise ValueError(f"astro_id {astro_id} not in clustered set.")

        # Ask for extra neighbor to account for self at distance 0
        n_total = n + (0 if include_self else 1)
        dists, idxs = self.nn_.kneighbors(self.Z[i].reshape(1, -1), n_neighbors=1500)
        dists = dists[0].tolist()
        idxs = idxs[0].tolist()

        neighbors: List[Tuple[int, float]] = []
        for j, d in zip(idxs, dists):
            nid = self.ids[j]
            if filter_ids and nid not in filter_ids:
                continue
            if include_self or nid != astro_id:
                neighbors.append((nid, float(d)))
            if len(neighbors) == n:
                break

        if not neighbors:
            raise ValueError("No neighbors found (check n/include_self settings).")

        # Helper: make a single panel for one neighbor with distance in title
        def _one_panel(ax, view: np.ndarray, nid: int, dist: float, df):
            
            disp_p = df.loc[df["astro_id"] == nid, "disp_p"].iloc[0]
            disp_e = df.loc[df["astro_id"] == nid, "disp_e"].iloc[0]
            disp_j = df.loc[df["astro_id"] == nid, "disp_j"].iloc[0]
            phase = np.linspace(0, 1, len(view), endpoint=False)
            ax.plot(phase, view, marker='.', linestyle='-')
            ax.set_title(
                f"ID {nid}  (d={dist:.3f})\n"
                f"disp_p={disp_p:.2f}, disp_e={disp_e:.2f}, disp_j={disp_j:.2f}",
                fontsize=10
            )
            ax.set_xlabel("Phase")
            ax.set_ylabel("Norm. flux")
            ax.invert_yaxis()
            ax.grid(True)

        if layout == "list":
            figs: List[plt.Figure] = []
            for nid, dist in neighbors:
                view = self.id_to_view.get(nid)
                if view is None:
                    # skip if not loaded
                    continue
                fig, ax = plt.subplots(figsize=(6, 3.5))
                _one_panel(ax, view, nid, dist, df)
                fig.tight_layout()
                figs.append(fig)
            return figs

        # layout == "grid"
        rows = math.ceil(len(neighbors) / max(1, cols))
        # size each cell similarly to your _plot_view
        fig_w = 6
        fig_h = 3.5
        fig, axes = plt.subplots(rows, cols, figsize=(cols * fig_w * 0.55, rows * fig_h * 0.55))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        for k, (nid, dist) in enumerate(neighbors):
            r, c = divmod(k, cols)
            ax = axes[r, c]
            view = self.id_to_view.get(nid)
            if view is None:
                ax.axis("off")
                continue
            _one_panel(ax, view, nid, dist, df)

        # turn off any unused axes
        for k in range(len(neighbors), rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")

        fig.suptitle(f"Nearest neighbors of {astro_id}", y=0.98)
        fig.tight_layout()
        return fig