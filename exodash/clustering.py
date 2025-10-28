import math
import os
from dataclasses import dataclass, asdict, is_dataclass
import seaborn as sns
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

import plotly.graph_objects as go
from matplotlib.colors import Normalize
from astronet.util import config_util
from astronet.astro_cnn_model import input_ds
from astronet.preprocess import preprocess
import streamlit as st

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


def params_to_key(params) -> tuple:
    """Make your self.params hashable and stable."""
    if is_dataclass(params):
        d = asdict(params)
    elif isinstance(params, dict):
        d = params
    else:
        # Fallback: read attributes that matter
        d = {k: getattr(params, k) for k in dir(params) if not k.startswith("_") and
            isinstance(getattr(params, k), (int, float, str, bool, type(None)))}
    return tuple(sorted(d.items()))

@st.cache_data(show_spinner=True, max_entries=6)
def cached_l2_normalize(Z: np.ndarray, cosine_normalize: bool) -> np.ndarray:
    if not cosine_normalize:
        return Z
    from sklearn.preprocessing import normalize
    return normalize(Z, norm="l2", axis=1)

@st.cache_resource(show_spinner=True, max_entries=3)
def cached_hdbscan_fit(Z_cos: np.ndarray, hdb_key: tuple):
    import hdbscan
    # reconstruct kwargs from key
    hdb_kwargs = dict(hdb_key)
    cl = hdbscan.HDBSCAN(**hdb_kwargs).fit(Z_cos)
    return cl  # resource (fitted model)

@st.cache_data(show_spinner=True, max_entries=6)
def cached_membership_vectors(_clusterer) -> np.ndarray:
    import hdbscan
    return hdbscan.all_points_membership_vectors(_clusterer)

@st.cache_data(show_spinner=True, max_entries=6)
def cached_umap(Z_cos: np.ndarray, umap_key: tuple, seed: int = 0) -> np.ndarray:
    import umap
    u = umap.UMAP(random_state=seed, **dict(umap_key))
    return u.fit_transform(Z_cos)

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
        eval_files: List[str],
        config_path: str,
        view_key: str = "global_view",
        params: ClusterParams = ClusterParams(),
    ):
        self.df = df
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
        print(pairs)
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

        for features, identifiers in ds:
            ids = identifiers.numpy()
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
    
    def load_views(self, ids_to_filter=None, data_source: str = 'embeddings') -> None:
        print('Load views')
        id_to_view_all: Dict[int, np.ndarray] = {}
        for _, ds in self._build_eval_set():
            id_to_view_all.update(self._build_id_to_view(ds, self.view_key, ids_to_filter))
        self.id_to_view = id_to_view_all
        print(f'Built views: {len(self.id_to_view)}')
        if data_source == 'tfrecords':
            """Populate self.id_to_view from TFRecords."""
            self.ids = sorted(self.id_to_view.keys())
            X = [robust_z(self.id_to_view[i]) for i in self.ids]
            self.X = np.vstack(X).astype(np.float32)
        else: 
            fc_cols = [c for c in self.df.columns if c.startswith("fc_")]
            assert len(fc_cols) > 0, "No embedding columns found (fc_*). Did you export embeddings?"
            self.ids = self.df["astro_id"].astype(int).tolist()
            X = self.df[fc_cols].to_numpy(dtype=np.float32)
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
        self.pca = PCA(n_components=self.params.pca_components, random_state=42).fit(self.X)
        self.Z = self.pca.transform(self.X)
        self.nn_ = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(self.Z)

    def fit_clusters(self) -> None:
        # Cosine emphasis (shape)
        Z_cos = cached_l2_normalize(self.Z, self.params.cosine_normalize)

        # HDBSCAN
        hdb_params = {
            "min_cluster_size": self.params.min_cluster_size,
            "min_samples": self.params.min_samples,
            "cluster_selection_epsilon": self.params.cluster_selection_epsilon,
            "cluster_selection_method": self.params.cluster_selection_method,
            "prediction_data": True,
            "metric": "euclidean",
        }
        self.clusterer = cached_hdbscan_fit(Z_cos, tuple(sorted(hdb_params.items())))
        self.labels_ = self.clusterer.labels_

        # Soft memberships for post-assign
        self.probs_ = cached_membership_vectors(self.clusterer)

        # UMAP (viz only)
        umap_key = tuple(sorted({
            "n_neighbors": self.params.umap_n_neighbors,
            "min_dist": self.params.umap_min_dist,
            "metric": "euclidean",
        }.items()))
        self.viz_ = cached_umap(Z_cos, umap_key, seed=0)

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
        df,
        use_post_labels: bool = True,      # kept for compatibility with your pipeline even though colors come from df[color_by]
        highlight_ids: Optional[List[int]] = None,
        alpha_bg: float = 0.12,            # how faint the background is
        s_bg: int = 10,                    # background marker size
        s_hi: int = 28,                    # highlighted marker size
        annotate: bool = False,            # label highlighted points with astro_id
        color_by: Optional[str] = 'first_letter',  # column in df used to color points; None -> no coloring
    ) -> plt.Figure:
        """UMAP scatter: if color_by is set, color by df[color_by]; otherwise use a uniform background color.
        Selected astro_ids are highlighted."""
        assert self.viz_ is not None, "self.viz_ must be computed"
        assert hasattr(self, "ids"), "self.ids must exist and correspond to df['astro_id']"
        assert 'astro_id' in df.columns, "df must contain 'astro_id'"
        if color_by is not None:
            assert color_by in df.columns, f"df is missing the '{color_by}' column"

        # Utilities
        def _rgba_with_alpha(rgba, alpha):
            r, g, b, _ = rgba
            return (r, g, b, alpha)

        legend_handles = []

        # Build colors
        if color_by is None:
            # Uniform, de-emphasized background
            bg_colors = [(0.5, 0.5, 0.5, alpha_bg)] * len(self.viz_)
            def hi_color_at_index(_i):
                # single accent color for highlights
                return (0.1, 0.1, 0.1, 1.0)
            title = "Light-curve islands (UMAP)"
        else:
            # Map df rows to the embedding order via astro_id
            df = df.drop_duplicates(subset="astro_id", keep="first").set_index("astro_id")
            df_by_id = df.set_index('astro_id')
            series = df_by_id[color_by].reindex(self.ids)

            # Determine coloring mode
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_numeric:
                vals = series.astype(float)
                vmin = np.nanmin(vals) if np.isfinite(vals).any() else 0.0
                vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0

                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.get_cmap('viridis')

                bg_colors = []
                for v in vals:
                    if np.isfinite(v):
                        bg_colors.append(_rgba_with_alpha(cmap(norm(v)), alpha_bg))
                    else:
                        bg_colors.append((0.7, 0.7, 0.7, 0.25 * alpha_bg))  # gray for missing

                def hi_color_at_index(i):
                    v = vals[i]
                    if np.isfinite(v):
                        return _rgba_with_alpha(cmap(norm(v)), 1.0)
                    else:
                        return (0.5, 0.5, 0.5, 0.9)

                legend_handles.append(Line2D([0], [0], marker='o', linestyle='',
                                            markersize=np.sqrt(s_bg), markerfacecolor=cmap(norm(vmin)),
                                            markeredgecolor='none', label=f'{color_by} (low)'))
                legend_handles.append(Line2D([0], [0], marker='o', linestyle='',
                                            markersize=np.sqrt(s_bg), markerfacecolor=cmap(norm(vmax)),
                                            markeredgecolor='none', label=f'{color_by} (high)'))
            else:
                # Categorical palette
                cat_values = series.astype('string').fillna('NA')
                codes, uniques = pd.factorize(cat_values, sort=True)
                palette = sns.color_palette("colorblind", n_colors=len(uniques))
                bg_colors = []
                for code in codes:
                    if code >= 0:
                        r, g, b = palette[code % len(palette)]
                        bg_colors.append((r, g, b, alpha_bg))
                    else:
                        bg_colors.append((0.7, 0.7, 0.7, 0.25 * alpha_bg))

                def hi_color_at_index(i):
                    code = codes[i]
                    if code >= 0:
                        r, g, b = palette[code % len(palette)]
                        return (r, g, b, 1.0)
                    else:
                        return (0.5, 0.5, 0.5, 0.9)

                # Legend chips (cap to a reasonable number)
                max_legend = min(len(uniques), 12)
                for idx in range(max_legend):
                    r, g, b = palette[idx % len(palette)]
                    legend_handles.append(
                        Line2D([0], [0], marker='o', linestyle='',
                            markersize=np.sqrt(s_bg), markerfacecolor=(r, g, b, 0.9),
                            markeredgecolor='none', label=f'{color_by} = {str(uniques[idx])}')
                    )
                if len(uniques) > max_legend:
                    legend_handles.append(
                        Line2D([0], [0], linestyle='none', label=f'(+{len(uniques)-max_legend} more)')
                    )

            title = f"Light-curve islands (UMAP), colored by '{color_by}'"

        # Figure and axes
        fig, ax = plt.subplots(figsize=(7, 6))

        # 1) Background
        ax.scatter(self.viz_[:, 0], self.viz_[:, 1], s=s_bg, c=bg_colors, linewidths=0, zorder=1)

        # 2) Highlights
        if highlight_ids:
            mask = np.isin(np.asarray(self.ids), list(highlight_ids))
            if np.any(mask):
                hi_colors = [hi_color_at_index(i) for i in np.where(mask)[0]]
                ax.scatter(self.viz_[mask, 0], self.viz_[mask, 1],
                        s=s_hi, c=hi_colors, edgecolors="k", linewidths=0.6, zorder=3)

                if annotate:
                    for x, y, aid in zip(self.viz_[mask, 0], self.viz_[mask, 1], np.asarray(self.ids)[mask]):
                        ax.annotate(str(aid), (x, y), xytext=(3, 3), textcoords="offset points",
                                    fontsize=8, color="k", zorder=4)

                legend_handles.append(
                    Line2D([0], [0], marker='o', linestyle='',
                        markersize=np.sqrt(s_hi), markerfacecolor=(0.3, 0.3, 0.3, 0.9),
                        markeredgecolor='k', label='Highlighted')
                )
            else:
                ax.text(0.02, 0.98, "No astro_id matched the filter", transform=ax.transAxes,
                        va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, lw=0))

        if legend_handles:
            ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=9)

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        return fig
    

    def plot_interactive(
        self,
        df: pd.DataFrame,
        use_post_labels: bool = True,         # kept for parity with your API
        highlight_ids=None,
        alpha_bg: float = 0.12,
        s_bg: int = 6,
        color_by: str = "first_letter",
        select_mode: str = "box",             # "box" or "lasso"
        height: int = 640
    ):
        """
        Interactive UMAP with box/lasso selection.
        Returns (fig, colors, selected_ids_function) so Streamlit can render and fetch selections.
        """
        assert self.viz_ is not None, "self.viz_ must be computed"
        assert hasattr(self, "ids"), "self.ids must exist and correspond to df['astro_id']"
        assert "astro_id" in df.columns, "df must contain 'astro_id'"
        if color_by is not None:
            assert color_by in df.columns, f"df is missing the '{color_by}' column"

        # map df rows to embedding order via astro_id
        df_by_id = df.drop_duplicates(subset="astro_id", keep="first").set_index("astro_id")
        #df_by_id = df.set_index("astro_id")
        series = df_by_id[color_by].reindex(self.ids) if color_by is not None else None

        # colors
        if color_by is None:
            # uniform gray
            marker_color = "rgba(128,128,128,{})".format(alpha_bg)
            colors = [marker_color] * len(self.ids)
            colorbar = None
            show_colorbar = False
            hover_extra = None
        else:
            is_numeric = pd.api.types.is_numeric_dtype(series)
            if is_numeric:
                vals = series.astype(float)
                vmin = np.nanmin(vals) if np.isfinite(vals).any() else 0.0
                vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)

                # Plotly will color by a separate numeric array (better than precomputed RGBA for colorbar)
                colors = vals.values
                colorbar = dict(title=color_by)
                show_colorbar = True
                hover_extra = vals
            else:
                # categorical
                cat_values = series.astype("string").fillna("NA")
                codes, uniques = pd.factorize(cat_values, sort=True)
                palette = sns.color_palette("colorblind", n_colors=max(1, len(uniques)))
                # convert to rgba strings with alpha
                def code_to_rgba(c):
                    if c >= 0:
                        r, g, b = palette[c % len(palette)]
                        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha_bg})"
                    return f"rgba(180,180,180,{0.25*alpha_bg})"
                colors = [code_to_rgba(c) for c in codes]
                colorbar = None
                show_colorbar = False
                hover_extra = cat_values.values

        # highlight overlay (drawn later)
        hi_mask = None
        if highlight_ids:
            hi_mask = np.isin(np.asarray(self.ids), list(highlight_ids))

        print(self.ids)
        print(highlight_ids)

        # Build base scatter (Scattergl = WebGL; fast for 20k+ points)
        x = self.viz_[:, 0]
        y = self.viz_[:, 1]
        customdata = np.array(self.ids)  # we’ll read this back from selections

        base_kwargs = dict(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=s_bg, opacity=1.0),
            customdata=customdata,
            hovertemplate="<b>astro_id</b>: %{customdata}<br>UMAP-1=%{x:.3f}<br>UMAP-2=%{y:.3f}"
        )
        if color_by is None or not pd.api.types.is_numeric_dtype(series) if color_by is not None else True:
            # using per-point rgba strings, no colorbar
            base_kwargs["marker"]["color"] = colors
        else:
            # numeric color with colorbar
            base_kwargs["marker"]["color"] = colors
            base_kwargs["marker"]["colorbar"] = colorbar
            base_kwargs["marker"]["colorscale"] = "Viridis"
            base_kwargs["marker"]["showscale"] = show_colorbar
            if hover_extra is not None:
                base_kwargs["hovertemplate"] += f"<br><b>{color_by}</b>: %{{marker.color:.4g}}"

        fig = go.Figure(data=[go.Scattergl(**base_kwargs)])
        title = "Light-curve islands (UMAP)" + (f", colored by '{color_by}'" if color_by else "")
        fig.update_layout(
            title=title,
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            dragmode="select" if select_mode == "box" else "lasso",
            height=height,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        # overlay highlights (solid color + thin outline)
        if hi_mask is not None and np.any(hi_mask):
            fig.add_trace(go.Scattergl(
                x=x[hi_mask],
                y=y[hi_mask],
                mode="markers",
                marker=dict(size=max(s_bg+4, 10), color="rgba(40,40,40,1.0)", line=dict(width=0.8, color="black")),
                customdata=customdata[hi_mask],
                hovertemplate="<b>astro_id</b>: %{customdata}<extra></extra>",
                name="Highlighted",
            ))

        # function to post-process selection payloads into astro_ids
        def _selected_ids_from_plotly_events(selected_points):
            """selected_points = output of plotly_events(...) (list[dict])"""
            if not selected_points:
                return []
            # each item has 'customdata' when we set it above
            ids = []
            for pt in selected_points:
                cd = pt.get("customdata", None)
                if cd is not None:
                    ids.append(int(cd))
                else:
                    # Fallback: map by pointIndex for the base trace
                    # (pointIndex == row in your x/y/customdata arrays)
                    idx = pt.get("pointIndex", None)
                    curve = pt.get("curveNumber", 0)
                    if idx is not None:
                        if curve == 0:
                            # background trace: index aligns to self.ids
                            ids.append(int(self.ids[idx]))
                        elif curve == 1:
                            # highlight overlay: index aligns to the masked array
                            # rebuild mask the same way you did above:
                            hi_mask = np.isin(np.asarray(self.ids), list(highlight_ids)) if highlight_ids else None
                            if hi_mask is not None:
                                ids.append(int(np.asarray(self.ids)[hi_mask][idx]))
            # dedupe (lasso can return dups)
            return list(dict.fromkeys(ids))

        return fig, _selected_ids_from_plotly_events
    
    def get_nearest_neighbors(
        self,
        astro_id: int,
        df,
        n: int = 6,
        include_self: bool = True,
        layout: Literal["grid", "list"] = "grid",
        cols: int = 3,
        filter_ids=None,
        data_source: str = 'embeddings'
    ) -> pd.DataFrame:
        assert self.nn_ is not None and self.Z is not None, "Call fit() first."
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
        records = []
        for dist, idx in zip(dists, idxs):
            records.append({"astro_id": int(idx), "distance": float(dist)})

        df = pd.DataFrame(records)
        return df.head(n)
    
    def show_nearest_neighbors(
        self,
        astro_id: int,
        df,
        n: int = 6,
        include_self: bool = False,
        layout: Literal["grid", "list"] = "grid",
        cols: int = 3,
        filter_ids=None,
        data_source: str = 'embeddings'
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
        try:
            i = self.ids.index(astro_id)
        except ValueError:
            raise ValueError(f"astro_id {astro_id} not in clustered set.")
        

        # Ask for extra neighbor to account for self at distance 0
        n_total = n + (0 if include_self else 1)
        dists, idxs = self.nn_.kneighbors(self.Z[i].reshape(1, -1), n_neighbors=len(self.id_to_view))
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
        assert self.id_to_view, "Call load_views() first."

        # Helper: make a single panel for one neighbor with distance in title
        def _one_panel(ax, view: np.ndarray, nid: int, dist: float, df):
            
            disp_p = df.loc[df["astro_id"] == nid, "disp_p"].iloc[0]
            disp_e = df.loc[df["astro_id"] == nid, "disp_e"].iloc[0]
            disp_j = df.loc[df["astro_id"] == nid, "disp_j"].iloc[0]
            tmag = df.loc[df["astro_id"] == nid, "tmag"].iloc[0]
            period = df.loc[df["astro_id"] == nid, "period"].iloc[0]
            duration = df.loc[df["astro_id"] == nid, "duration"].iloc[0]
            #sector = df.loc[df["astro_id"] == nid, "sector"].iloc[0]
            phase = np.linspace(0, 1, len(view), endpoint=False)
            ax.plot(phase, view, marker='.', linestyle='-')
            ax.set_title(
                f"ID {nid} (d={dist:.3f})\n"
                f"disp_p={disp_p:.2f}, disp_e={disp_e:.2f}, disp_j={disp_j:.2f}\n"
                f"tmag={tmag:.2f}, period={period:.2f}, duration={duration:.2f}",
                fontsize=10
            )
            ax.set_xlabel("Phase")
            ax.set_ylabel("Norm. flux")
            #ax.invert_yaxis()
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