"""
scib-metrics analysis: original vs v4 (Harmony batch correction quality)
Batch key: orig.ident × BaseEditor × Phase (30 groups) — Harmony, scVI 동일 보정 변수
Bio key: cluster (Leiden/seurat clusters)
Embeddings tested: PCA (pre-correction) vs Harmony vs scVI
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

import os
import scib_metrics
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# ─── helpers ─────────────────────────────────────────────────────────────────

BASE = "/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/scib_input"
OUT  = "/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/scib_results"
os.makedirs(OUT, exist_ok=True)

def load_adata(label):
    d = f"{BASE}/{label}"
    mat   = csr_matrix(mmread(f"{d}/counts.mtx").T)     # cells x genes
    genes = pd.read_csv(f"{d}/genes.csv")["gene"].values
    cells = pd.read_csv(f"{d}/cells.csv")["cell"].values
    meta  = pd.read_csv(f"{d}/metadata.csv", index_col=0)
    pca   = pd.read_csv(f"{d}/pca.csv",     index_col=0).values
    harm  = pd.read_csv(f"{d}/harmony.csv", index_col=0).values
    umap  = pd.read_csv(f"{d}/umap.csv",    index_col=0).values

    adata = ad.AnnData(X=mat, obs=meta)
    adata.obs_names = cells
    adata.var_names = genes

    adata.obsm["X_pca"]     = pca.astype(np.float32)
    adata.obsm["X_harmony"] = harm.astype(np.float32)
    adata.obsm["X_umap"]    = umap.astype(np.float32)

    # Harmony, scVI 모두 orig.ident + BaseEditor + Phase 3가지를 보정 — 동일 기준으로 평가
    adata.obs["batch"] = (
        adata.obs["orig.ident"].astype(str) + "__" +
        adata.obs["BaseEditor"].astype(str) + "__" +
        adata.obs["Phase"].astype(str)
    )
    adata.obs["cluster"] = adata.obs["cluster"].astype(str)

    # scVI latent (optional)
    scvi_path = f"{d}/scvi.csv"
    if os.path.exists(scvi_path):
        scvi_df = pd.read_csv(scvi_path, index_col=0).reindex(cells)
        adata.obsm["X_scvi"] = scvi_df.values.astype(np.float32)
        print(f"[{label}] scVI latent loaded: {adata.obsm['X_scvi'].shape}")
    else:
        print(f"[{label}] scVI latent NOT found — run 02_scVI_Batch_tuned.py first")

    # scANVI latent (optional)
    scanvi_path = f"{d}/scanvi.csv"
    if os.path.exists(scanvi_path):
        scanvi_df = pd.read_csv(scanvi_path, index_col=0).reindex(cells)
        adata.obsm["X_scanvi"] = scanvi_df.values.astype(np.float32)
        print(f"[{label}] scANVI latent loaded: {adata.obsm['X_scanvi'].shape}")
    else:
        print(f"[{label}] scANVI latent NOT found — run 03_scANVI_Batch.py first")

    print(f"[{label}] cells={adata.n_obs}, genes={adata.n_vars}, "
          f"batches={adata.obs['batch'].nunique()}, "
          f"clusters={adata.obs['cluster'].nunique()}")
    return adata

# ─── load ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
orig = load_adata("orig")
v4   = load_adata("v4")

# ─── scib Benchmarker ────────────────────────────────────────────────────────
# We benchmark two embeddings per object:
#   X_pca     = pre-Harmony (unintegrated baseline)
#   X_harmony = post-Harmony (integrated)
#
# Metrics (scib-metrics 0.5.x):
#   Batch correction: iLISI, graph_iLISI, kBET, PCR_comparison, silhouette_batch
#   Bio conservation: NMI, ARI, isolated_label_silhouette, isolated_label_F1, cLISI

def run_benchmark(adata, label):
    print(f"\n{'='*60}")
    print(f"Running benchmark: {label}")

    embed_keys = ["X_pca", "X_harmony"]
    if "X_scvi" in adata.obsm:
        embed_keys.append("X_scvi")
    if "X_scanvi" in adata.obsm:
        embed_keys.append("X_scanvi")
    print(f"Embeddings to benchmark: {embed_keys}")

    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cluster",
        embedding_obsm_keys=embed_keys,
        pre_integrated_embedding_obsm_key="X_pca",
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=False,
            silhouette_label=True,
            isolated_labels=True,
            clisi_knn=True,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=True,
            ilisi_knn=True,
            kbet_per_label=True,
            graph_connectivity=True,
            pcr_comparison=True,
        ),
        n_jobs=4,
    )

    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    df.to_csv(f"{OUT}/{label}_raw.csv")

    df_scaled = bm.get_results(min_max_scale=True)
    df_scaled.to_csv(f"{OUT}/{label}_scaled.csv")

    print(f"\n[{label}] Raw scores:")
    print(df.to_string())
    return bm, df, df_scaled

bm_orig, df_orig, dfs_orig = run_benchmark(orig, "orig")
bm_v4,   df_v4,   dfs_v4   = run_benchmark(v4,   "v4")

# ─── side-by-side summary ────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY: Harmony vs scVI (v4 dataset)")
print("="*60)

def get_row(df, key):
    if key in df.index:
        return df.loc[key]
    mask = df.index.str.contains(key.lstrip("X_"), case=False)
    return df[mask].iloc[0] if mask.any() else None

row_pca     = get_row(df_v4, "X_pca")
row_harmony = get_row(df_v4, "X_harmony")
row_scvi    = get_row(df_v4, "X_scvi")

cols = {}
if row_pca     is not None: cols["PCA (baseline)"] = row_pca
if row_harmony is not None: cols["Harmony"]        = row_harmony
if row_scvi    is not None: cols["scVI"]           = row_scvi

summary = pd.DataFrame(cols)
if "Harmony" in summary.columns and "scVI" in summary.columns:
    summary["scVI - Harmony"] = summary["scVI"] - summary["Harmony"]
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
summary.to_csv(f"{OUT}/harmony_vs_scvi.csv")

# ─── legacy: orig vs v4 Harmony comparison ───────────────────────────────────
row_orig = get_row(df_orig, "X_harmony")
row_v4h  = get_row(df_v4,   "X_harmony")
if row_orig is not None and row_v4h is not None:
    cmp = pd.DataFrame({"orig (res=0.6)": row_orig, "v4 (res=0.9)": row_v4h})
    cmp["delta (v4-orig)"] = cmp["v4 (res=0.9)"] - cmp["orig (res=0.6)"]
    cmp.to_csv(f"{OUT}/harmony_comparison.csv")

print(f"\nAll results saved to: {OUT}/")
print("Files: v4_raw.csv, v4_scaled.csv, harmony_vs_scvi.csv")
