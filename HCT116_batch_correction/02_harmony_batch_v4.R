library(harmony)
library(Seurat)
library(viridis)
library(ggplot2)
library(dplyr)
library(future)

dims.reduce = 20
cluster.res = 0.9
ram_gb = 48
options(future.globals.maxSize = ram_gb * 1024^3)

OUT_DIR <- '/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/results_multi_res_v4'
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat("Output directory:", OUT_DIR, "\n")

plan('multisession', workers=12)

# ── Full from-scratch pipeline ───────────────────────────────────────────────
# Harmony 1.0.3 is a full rewrite from 0.x; default params cause NGG_Bulk_1
# to form an isolated island. Fix: nclust=10, theta=c(8,2,2), sigma=0.1.
cat("[1/5] Loading Seurat object...\n")
aggr <- readRDS('/home/tech/variantseq/eugenie/variant_seq_EDA/kimhs/HCT116_Seurat_Share.rds')
DefaultAssay(aggr) <- "RNA"

cat("[2/5] SCTransform (vst.flavor=v1, ncells=5000, variable.features.n=3000)...\n")
aggr <- SCTransform(aggr,
                    vst.flavor          = "v1",
                    ncells              = 5000,
                    variable.features.n = 3000,
                    seed.use            = 1448145,
                    verbose             = FALSE)

cat("[3/5] RunPCA (npcs=50)...\n")
aggr <- RunPCA(aggr, npcs = 50, seed.use = 42, verbose = FALSE)

cat("[4/5] RunHarmony (nclust=10, theta=c(8,2,2), sigma=0.1)...\n")
aggr <- RunHarmony(aggr,
                   group.by.vars  = c("orig.ident", "BaseEditor", "Phase"),
                   reduction.use  = "pca",
                   lambda         = c(1, 1, 1),
                   nclust         = 10,
                   theta          = c(8, 2, 2),
                   sigma          = 0.1,
                   reduction.save = "harmony",
                   verbose        = FALSE)

h_cols <- colnames(aggr@reductions$harmony@cell.embeddings)
cat("Harmony dim2 NGG_Bulk_1 mean:",
    round(mean(aggr@reductions$harmony@cell.embeddings[
      aggr$orig.ident == "NGG_Bulk_1", h_cols[2]]), 2), "\n")

cat("[5/5] UMAP + FindNeighbors + FindClusters...\n")
aggr <- RunUMAP(aggr,
                reduction   = "harmony",
                dims        = 1:dims.reduce,
                umap.method = "uwot",
                metric      = "cosine",
                n.neighbors = 30,
                min.dist    = 0.3,
                seed.use    = 42)

aggr <- FindNeighbors(aggr, reduction = "harmony", dims = 1:dims.reduce,
                      k.param = 20) %>%
        FindClusters(resolution = cluster.res, random.seed = 0)

n_clusters <- length(unique(Idents(aggr)))
cat("Clusters found:", n_clusters, "\n")

rds_full <- file.path(OUT_DIR, "HCT116.harmony.Cellcycle.aggr.rds")
saveRDS(aggr, file = rds_full)
cat("Saved:", rds_full, "\n")


# ── ProteinChange subsetting ─────────────────────────────────────────────────
aggr <- readRDS(rds_full)
meta_lt_ncells <- function(seurat_obj, meta_col, min_cells) {
  counts <- table(seurat_obj@meta.data[[meta_col]])
  names(counts[counts < min_cells])
}
grps_remove <- meta_lt_ncells(aggr, 'ProteinChange', 5)
grps_remove <- append(grps_remove, 'NA')
aggr_sub <- subset(aggr, subset = ProteinChange %in% grps_remove, invert = TRUE)

aggr_sub$ProteinChange <- gsub('X', 'Ter', aggr_sub$ProteinChange)
aggr_sub@meta.data$ProteinChange <- as.factor(aggr_sub@meta.data$ProteinChange)
color <- scales::hue_pal()(length(unique(aggr_sub$ProteinChange)))
color[which(levels(aggr_sub@meta.data$ProteinChange) == 'WT')] <- '#5D5F60'

f1 <- file.path(OUT_DIR, 'Harmony_Merged_UMAP_ProteinChange.pdf')
pdf(f1, height = 9, width = 10)
print(DimPlot(aggr_sub, group.by = 'ProteinChange', cols = color, pt.size = 1) +
  guides(color = guide_legend(override.aes = list(size = 3), ncol = 2)) +
  ggtitle(''))
dev.off()
cat("Saved:", f1, "\n")

f2 <- file.path(OUT_DIR, 'Harmony_Merged_UMAP_ProteinChange.col3.pdf')
pdf(f2, height = 9, width = 12)
print(DimPlot(aggr_sub, group.by = 'ProteinChange', cols = color, pt.size = 1) +
  guides(color = guide_legend(override.aes = list(size = 3), ncol = 3)) +
  ggtitle(''))
dev.off()
cat("Saved:", f2, "\n")

f3 <- file.path(OUT_DIR, 'Harmony_Merged_UMAP.pdf')
pdf(f3)
print(DimPlot(aggr_sub, label = TRUE, pt.size = 1) +
  guides(color = guide_legend(override.aes = list(size = 5))) +
  theme(legend.key.size = unit(0.3, 'in')))
print(DimPlot(aggr_sub, group.by = 'BaseEditor', pt.size = 1))
dev.off()
cat("Saved:", f3, "\n")

f4 <- file.path(OUT_DIR, 'Harmony_Cell_Cycle_Phase.pdf')
pdf(f4)
print(DimPlot(aggr_sub, group.by = 'Phase', pt.size = 1))
dev.off()
cat("Saved:", f4, "\n")

rds_sub <- file.path(OUT_DIR, 'HCT116.harmony.CellCycle.aggr.subset.rds')
saveRDS(aggr_sub, file = rds_sub)
cat("Saved:", rds_sub, "\n")

cat("\n===== Done! =====\n")
cat("Results in:", OUT_DIR, "\n")
