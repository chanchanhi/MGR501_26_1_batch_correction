"""
scVI batch correction — tuned version
02_Harmony_Batch.R 와 동일한 보정 변수 적용:
  batch_key = orig.ident  (Harmony group.by.vars[1])
  categorical_covariate_keys = [BaseEditor, Phase]  (Harmony group.by.vars[2,3])

추가 개선 사항 (원본 scVI 대비):
  1. use_layer_norm="both" + use_batch_norm="none"
     → Luecken et al. 2022 best practice: batch norm이 bio signal 날리는 것 방지
  2. encode_covariates=True + deeply_inject_covariates=True
     → encoder/decoder 전 layer에 모든 보정 변수 주입 → kBET↑
  3. n_latent=30 (20→30): 더 많은 차원 → bio structure 보존↑
  4. n_layers=3 (2→3): 더 깊은 네트워크
  5. n_hidden=256 (128→256): 더 넓은 네트워크
  6. early_stopping_patience=45 (20→45)
  7. lr=2e-3 → 수렴 가속

Run:  conda run -n scvi python 02_scVI_Batch_tuned.py
Out:  scib_input/v4/scvi.csv       — 30-dim latent
      scib_input/v4/scvi_umap.csv  — UMAP coords
      scVI/v4_model_tuned/         — saved model
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
from scipy.io import mmread
from scipy.sparse import csr_matrix

# ── config ───────────────────────────────────────────────────────────────────
LABEL     = "v4"
BASE      = "/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/scib_input"
MODEL_DIR = "/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/scVI"
N_LATENT  = 30       # 20→30: latent space 확장 (bio structure 보존↑)
N_HVG     = 3000
N_EPOCHS  = 400
SEED      = 42

os.makedirs(MODEL_DIR, exist_ok=True)
scvi.settings.seed = SEED

d = f"{BASE}/{LABEL}"

# ── load raw counts ───────────────────────────────────────────────────────────
print("[1/5] Loading raw RNA counts...")
raw_mtx  = f"{d}/raw_counts.mtx"
raw_gene = f"{d}/raw_genes.csv"

if not os.path.exists(raw_mtx):
    raise FileNotFoundError(
        f"{raw_mtx} not found.\n"
        "Run export_for_scib.R first to export raw RNA counts."
    )

mat   = csr_matrix(mmread(raw_mtx).T)
genes = pd.read_csv(raw_gene)["gene"].values
cells = pd.read_csv(f"{d}/cells.csv")["cell"].values
meta  = pd.read_csv(f"{d}/metadata.csv", index_col=0)

adata = ad.AnnData(X=mat, obs=meta)
adata.obs_names = cells
adata.var_names = genes
adata.obs["batch"]       = adata.obs["orig.ident"].astype(str)
adata.obs["BaseEditor"]  = adata.obs["BaseEditor"].astype(str)
adata.obs["Phase"]       = adata.obs["Phase"].astype(str)
adata.obs["cluster"]     = adata.obs["cluster"].astype(str)

print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
print(f"  Batches (orig.ident): {dict(adata.obs['batch'].value_counts())}")
print(f"  BaseEditor: {sorted(adata.obs['BaseEditor'].unique())}")
print(f"  Phase: {sorted(adata.obs['Phase'].unique())}")

# ── HVG selection ─────────────────────────────────────────────────────────────
print(f"\n[2/5] Selecting top {N_HVG} HVGs (seurat_v3, per batch)...")
sc.pp.highly_variable_genes(
    adata, flavor="seurat_v3", n_top_genes=N_HVG,
    batch_key="batch", subset=True,
)
print(f"  HVGs after selection: {adata.n_vars}")

# ── scVI model ────────────────────────────────────────────────────────────────
print(f"\n[3/5] Setting up SCVI (tuned params)...")
# Harmony와 동일한 보정 변수: orig.ident (batch_key) + BaseEditor + Phase (categorical_covariate_keys)
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="batch",                               # orig.ident — 주 배치 변수
    categorical_covariate_keys=["BaseEditor", "Phase"],  # Harmony group.by.vars[2,3]
)

model = scvi.model.SCVI(
    adata,
    n_latent=N_LATENT,              # 20→30
    n_layers=3,                     # 2→3
    n_hidden=256,                   # 128→256
    gene_likelihood="nb",
    dispersion="gene-batch",
    use_layer_norm="both",          # Luecken et al. 2022 권장
    use_batch_norm="none",          # batch norm 제거 → bio signal 보호
    encode_covariates=True,         # encoder에 모든 보정 변수 주입
    deeply_inject_covariates=True,  # decoder 전 layer에 보정 변수 주입
    dropout_rate=0.0,
)
print(model)

# ── training ──────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training (max_epochs={N_EPOCHS}, early stopping patience=45)...")
model.train(
    max_epochs=N_EPOCHS,
    early_stopping=True,
    early_stopping_patience=45,  # 20→45: 213 epoch 조기 종료 방지
    plan_kwargs={"lr": 2e-3},    # 1e-3→2e-3: 초기 학습 가속
)

# ── save latent + UMAP ────────────────────────────────────────────────────────
print("\n[5/5] Extracting latent representation and computing UMAP...")
latent = model.get_latent_representation()

orig_cells = pd.read_csv(f"{d}/cells.csv")["cell"].values
latent_df = pd.DataFrame(
    latent,
    index=adata.obs_names,
    columns=[f"scVI_{i+1}" for i in range(N_LATENT)],
).reindex(orig_cells)

latent_df.to_csv(f"{d}/scvi.csv")
print(f"  Saved latent  → {d}/scvi.csv")

# UMAP on scVI latent
adata.obsm["X_scvi"] = latent.astype(np.float32)
sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=30, metric="cosine")
sc.tl.umap(adata, min_dist=0.3, random_state=SEED)

umap_df = pd.DataFrame(
    adata.obsm["X_umap"],
    index=adata.obs_names,
    columns=["UMAP_1", "UMAP_2"],
).reindex(orig_cells)

umap_df.to_csv(f"{d}/scvi_umap.csv")
print(f"  Saved UMAP    → {d}/scvi_umap.csv")

model_path = os.path.join(MODEL_DIR, "v4_model_tuned")
model.save(model_path, overwrite=True)
print(f"  Saved model   → {model_path}")

print("\n===== scVI Tuned Done! =====")
print(f"Next: run 'conda run -n eugenie python scib_analysis.py' to compare vs Harmony")
