library(Seurat)
library(Matrix)
library(dplyr) 

source('/mnt/ix1/Resources/LabSoftware/software/scRNA-v3/pipeline/_scRNA_methods.R')
#Link individual seurat objects from pipeline, already filtered for mitochondria and genes. Remove cells marked #doublet during seu_merged


robjs = list.files('.', pattern="*.seurat_dfx.annot.subset.sgrna.rds", full.names=F)
nr_seu = length(robjs)
sample_fn = gsub('.seurat_dfx.annot.subset.sgrna.rds','',robjs)

seu <- vector("list",nr_seu)
for (i in 1:nr_seu) {
  sprintf("Loading Seurat object: %s", robjs[i])
  seu[[i]] <- readRDS(robjs[i])
  #seu[[i]] <- RenameCells(seu[[i]], new.names=modifyBarcodeSuffix(colnames(seu[[i]]),i))
  seu[[i]] <- RenameCells(seu[[i]], add.cell.id=sample_fn[i])
  sprintf("%i cells loaded", length(Idents(seu[[i]])))
}

seu_merged <- merge(x=seu[[1]], y=seu[-1])

# doubletnumbers <- table(seu_merged@meta.data$orig.ident, seu_merged@meta.data$DF)
# write.csv(doubletnumbers, file='doubletnumbers.csv')

sprintf("Aggregate Seurat object has %i cells", length(Idents(seu_merged)))
seu_merged <- subset(seu_merged, subset=DF!='Doublet')
sprintf("Aggregate Seurat object has %i cells after removing doublets", length(Idents(seu_merged)))
seu_merged <- subset(seu_merged, subset=BaseEditor %in% c('ABE,CBE','ABE_NG,CBE_NG'), invert=T)
saveRDS(seu_merged, file="U2OS.merged.rds")

#checking
head(seu_merged@meta.data)
tail(seu_merged@meta.data)

