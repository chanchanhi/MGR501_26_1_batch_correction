suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

OUT <- "/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/scib_input"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

export_obj <- function(obj, label, cluster_col) {
  dir.create(file.path(OUT, label), showWarnings=FALSE)

  # Normalized counts (SCT data slot) - sparse matrix
  mat <- GetAssayData(obj, assay="SCT", layer="data")
  writeMM(mat, file.path(OUT, label, "counts.mtx"))
  write.csv(data.frame(gene=rownames(mat)), file.path(OUT, label, "genes.csv"), row.names=FALSE)
  write.csv(data.frame(cell=colnames(mat)), file.path(OUT, label, "cells.csv"), row.names=FALSE)

  # Raw RNA counts for scVI (integer counts, all genes)
  if ("RNA" %in% Assays(obj)) {
    raw_mat <- tryCatch(
      GetAssayData(obj, assay="RNA", layer="counts"),   # Seurat v5
      error = function(e) GetAssayData(obj, assay="RNA", slot="counts")  # Seurat v4
    )
    writeMM(raw_mat, file.path(OUT, label, "raw_counts.mtx"))
    write.csv(data.frame(gene=rownames(raw_mat)),
              file.path(OUT, label, "raw_genes.csv"), row.names=FALSE)
    cat("  Exported raw RNA counts:", nrow(raw_mat), "genes\n")
  }

  # PCA embedding
  pca <- Embeddings(obj, "pca")
  write.csv(pca, file.path(OUT, label, "pca.csv"))

  # Harmony embedding
  harm <- Embeddings(obj, "harmony")
  write.csv(harm, file.path(OUT, label, "harmony.csv"))

  # UMAP embedding
  umap <- Embeddings(obj, "umap")
  write.csv(umap, file.path(OUT, label, "umap.csv"))

  # Metadata
  meta <- obj@meta.data[, c("orig.ident", "BaseEditor", "Phase",
                              "Genotype", "ProteinChange", cluster_col)]
  colnames(meta)[colnames(meta)==cluster_col] <- "cluster"
  write.csv(meta, file.path(OUT, label, "metadata.csv"))

  cat("Exported:", label, "- cells:", ncol(obj), "\n")
}

cat("Loading orig...\n")
orig <- readRDS("/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/HCT116_Seurat_Share.rds")
export_obj(orig, "orig", "seurat_clusters")

cat("Loading v4...\n")
v4 <- readRDS("/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/results_multi_res_v4/HCT116.harmony.Cellcycle.aggr.rds")
export_obj(v4, "v4", "SCT_snn_res.0.9")

cat("Done.\n")
