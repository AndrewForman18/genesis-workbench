# Databricks notebook source
# MAGIC %md
# MAGIC # Scanpy Analysis in a Databricks Notebook
# MAGIC  - load in the data from a **Unity Catalog Volume**
# MAGIC  - analyze the data using **scanpy**
# MAGIC  - track results using **mlflow**

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install 'numpy<2' scanpy==1.11.4 anndata scikit-network scrublet harmonypy leidenalg scimilarity celltypist
# MAGIC %restart_python

# COMMAND ----------

import scanpy as sc
import anndata as ad
import os
import pandas as pd
import copy
import tempfile
import time

# COMMAND ----------

# DBTITLE 1,Pipeline widgets
dbutils.widgets.text("data_path", "", "Data Path")
dbutils.widgets.text("user_email", "", "User Email")
dbutils.widgets.text("mlflow_experiment", "", "MLflow Experiment")
dbutils.widgets.text("mlflow_run_name", "", "MLflow Run Name")
dbutils.widgets.text("catalog", "", "Catalog Name")
dbutils.widgets.text("schema", "", "Schema Name")
dbutils.widgets.text("gene_name_column", "", "Gene Name Column (optional)")
dbutils.widgets.text("species", "hsapiens", "Species (hsapiens/mmusculus/rnorvegicus)")
dbutils.widgets.text("min_genes", "200", "Min Genes per Cell")
dbutils.widgets.text("min_cells", "3", "Min Cells per Gene")
dbutils.widgets.text("pct_counts_mt", "5", "Max % Mitochondrial Counts")
dbutils.widgets.text("n_genes_by_counts", "2500", "Max Genes by Counts")
dbutils.widgets.text("target_sum", "10000", "Target Sum for Normalization")
dbutils.widgets.text("n_top_genes", "500", "Number of Highly Variable Genes")
dbutils.widgets.text("n_pcs", "50", "Number of Principal Components")
dbutils.widgets.text("cluster_resolution", "0.2", "Cluster Resolution")

# CellExpress/Cellatria integration — doublet detection, batch correction, clustering
dbutils.widgets.dropdown("doublet_method", "scrublet", ["none", "scrublet"], "Doublet Detection Method")
dbutils.widgets.text("scrublet_threshold", "0.25", "Scrublet Score Threshold")
dbutils.widgets.dropdown("batch_correction", "none", ["none", "harmony"], "Batch Correction Method")
dbutils.widgets.text("batch_key", "", "Batch Key Column (for batch correction)")
dbutils.widgets.dropdown("clustering_method", "leiden", ["leiden", "louvain"], "Clustering Method")

# Cell type annotation (SCimilarity/CellTypist from Cellatria)
dbutils.widgets.dropdown("annotation_method", "none", ["none", "scimilarity", "celltypist"], "Cell Type Annotation Method")
dbutils.widgets.text("annotation_model", "", "Annotation Model (path or name)")

# COMMAND ----------

# DBTITLE 1,Parameters
parameters = {
  'data_path':dbutils.widgets.get("data_path"),
  'user_email':dbutils.widgets.get("user_email"),
  'mlflow_experiment':dbutils.widgets.get("mlflow_experiment"),
  'mlflow_run_name':dbutils.widgets.get("mlflow_run_name"),
  'catalog':dbutils.widgets.get("catalog"),
  'schema':dbutils.widgets.get("schema"),
  'gene_name_column':dbutils.widgets.get("gene_name_column"),
  'species':dbutils.widgets.get("species"),
  'min_genes': int(dbutils.widgets.get("min_genes")),
  'min_cells': int(dbutils.widgets.get("min_cells")),
  'pct_counts_mt': float(dbutils.widgets.get("pct_counts_mt")),
  'n_genes_by_counts': int(dbutils.widgets.get("n_genes_by_counts")),
  'target_sum': int(float(dbutils.widgets.get("target_sum"))),
  'n_top_genes': int(dbutils.widgets.get("n_top_genes")),
  'n_pcs': int(dbutils.widgets.get("n_pcs")),
  'cluster_resolution': float(dbutils.widgets.get("cluster_resolution")),
  # CellExpress/Cellatria integration parameters
  'doublet_method': dbutils.widgets.get("doublet_method"),
  'scrublet_threshold': float(dbutils.widgets.get("scrublet_threshold")),
  'batch_correction': dbutils.widgets.get("batch_correction"),
  'batch_key': dbutils.widgets.get("batch_key"),
  'clustering_method': dbutils.widgets.get("clustering_method"),
  # Cell type annotation parameters
  'annotation_method': dbutils.widgets.get("annotation_method"),
  'annotation_model': dbutils.widgets.get("annotation_model"),
}

metrics = {}

# COMMAND ----------

parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ### make a directory to save some results
# MAGIC  - we'll later do some logging of results with **mlflow** 

# COMMAND ----------

# we'll save some things to disk to move to our experiment run when we log our findings
tmpdir = tempfile.TemporaryDirectory()
sc.settings.figdir = tmpdir.name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load in an h5ad file from Unity Catalog Volume
# MAGIC  - The Volume is blob storage, but you can interact with it like a unix system

# COMMAND ----------

t0 = time.time()
adata = sc.read_h5ad(
    parameters['data_path'],
)
adata.obs_names_make_unique()

# here we use adata.raw if it exists as is assumeed to be unprocessed raw data
if adata.raw is not None:
    adata = adata.raw.to_adata()

# Load gene names: either from provided column or from Ensembl reference
if parameters['gene_name_column'] and parameters['gene_name_column'].strip() != "":
    # User provided gene name column - use it
    GENE_NAME_COLUMN = parameters['gene_name_column']
    adata.var = adata.var.reset_index()
    adata.var[GENE_NAME_COLUMN] = adata.var[GENE_NAME_COLUMN].astype(str) 
    adata.var['Gene name'] = adata.var[GENE_NAME_COLUMN]
    # Normalize to uppercase for consistent QC
    adata.var['Gene name'] = adata.var['Gene name'].str.upper()
    # Set as index but keep column for easy access
    adata.var = adata.var.set_index('Gene name', drop=False)
    print(f"Using provided gene name column: {GENE_NAME_COLUMN}")
else:
    # No gene name column - load reference and join
    if not parameters['species'] or parameters['species'].strip() == "":
        raise ValueError("Species parameter is required when gene_name_column is not provided")
    
    CATALOG = parameters['catalog']
    SCHEMA = parameters['schema']
    ref_path = f'/Volumes/{CATALOG}/{SCHEMA}/scanpy_reference/ensembl_genes_{parameters["species"]}.csv'
    
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference table not found at {ref_path}. "
            f"Please ensure the reference tables have been downloaded during deployment."
        )
    
    ref_df = pd.read_csv(ref_path)
    print(f"Loaded reference table for {parameters['species']}: {len(ref_df)} genes")
    
    # Join reference to var (Ensembl IDs in var.index)
    adata.var = adata.var.reset_index()
    adata.var.rename(columns={'index': 'ensembl_gene_id'}, inplace=True)
    
    # Merge with reference
    adata.var = adata.var.merge(
        ref_df, 
        left_on='ensembl_gene_id', 
        right_on='ensembl_gene_id',
        how='left'
    )
    
    # Use external_gene_name if available, otherwise fall back to Ensembl ID
    adata.var['Gene name'] = adata.var['external_gene_name'].fillna(adata.var['ensembl_gene_id'])
    
    # Normalize to uppercase for consistent QC (before setting index)
    adata.var['Gene name'] = adata.var['Gene name'].str.upper()
    
    # Calculate mapping success rate
    mapped_count = (adata.var['Gene name'] != adata.var['ensembl_gene_id'].str.upper()).sum()
    total_genes = len(adata.var)
    mapping_rate = 100.0 * mapped_count / total_genes if total_genes > 0 else 0
    metrics['gene_mapping_rate'] = mapping_rate
    print(f"Mapped {mapped_count}/{total_genes} genes ({mapping_rate:.1f}%) using {parameters['species']} reference")
    
    # Set as index but keep column for easy access
    adata.var = adata.var.set_index('Gene name', drop=False)

# Make var names unique
adata.var_names_make_unique()

# COMMAND ----------

# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var['Gene name'].str.startswith("MT-",na=False)
# ribosomal genes
adata.var["ribo"] = adata.var['Gene name'].str.startswith(("RPS", "RPL"),na=False)
# hemoglobin genes
adata.var["hb"] = adata.var['Gene name'].str.contains("^HB[^(P)]",na=False)

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

current_cells = adata.shape[0]
metrics['total_cells_starting'] = float(current_cells)

# COMMAND ----------

# depending on input data and requirements may which to filter
sc.pp.filter_cells(adata, min_genes=parameters['min_genes'])
sc.pp.filter_genes(adata, min_cells=parameters['min_cells'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's keep track of how many cells we retain on filtering

# COMMAND ----------

metrics['filter_simple_retention'] = 100.0*adata.shape[0]/current_cells
current_cells = adata.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### generate useful QC plots and save to disk

# COMMAND ----------

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save="counts_plot_prefilter.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Perform additional filtration of cells

# COMMAND ----------

# filtering cells further
adata = adata[adata.obs.n_genes_by_counts < parameters['n_genes_by_counts'], :] # could use scrublet etc for doublet removal as desired
adata = adata[adata.obs.pct_counts_mt < parameters['pct_counts_mt'], :].copy() # or other threshold to remove dead/dying cells

# COMMAND ----------

# DBTITLE 1,Doublet Detection (Scrublet)
# MAGIC %md
# MAGIC #### Doublet Detection (Scrublet)

# COMMAND ----------

# DBTITLE 1,Scrublet doublet detection
# Scrublet doublet detection — integrated from CellAtria/CellExpress pipeline
if parameters['doublet_method'] == 'scrublet':
    import scrublet as scr
    
    scrub = scr.Scrublet(adata.X)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2, min_cells=3, min_gene_variability_pctl=85)
    
    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets
    
    # Apply threshold from parameters
    threshold = parameters['scrublet_threshold']
    adata.obs['is_doublet'] = adata.obs['doublet_score'] > threshold
    
    n_doublets = adata.obs['is_doublet'].sum()
    n_total = adata.shape[0]
    print(f"Scrublet detected {n_doublets} doublets ({100*n_doublets/n_total:.1f}%) at threshold {threshold}")
    
    metrics['n_doublets_detected'] = int(n_doublets)
    metrics['doublet_rate'] = float(n_doublets / n_total)
    
    # Remove doublets
    adata = adata[~adata.obs['is_doublet']].copy()
    print(f"Retained {adata.shape[0]} singlets after doublet removal")
    
    metrics['filter_doublet_retention'] = 100.0 * adata.shape[0] / current_cells
    current_cells = adata.shape[0]
else:
    print("Doublet detection: skipped (method=none)")
    metrics['n_doublets_detected'] = 0
    metrics['doublet_rate'] = 0.0

# COMMAND ----------

metrics['filter_mtgenes_retention'] =  100.0*adata.shape[0]/current_cells
current_cells = adata.shape[0]

# COMMAND ----------

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save="counts_plot_post_filter.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize the data, identify variable genes, dimension reduction

# COMMAND ----------

sc.pp.normalize_total(adata,target_sum=parameters['target_sum'])
sc.pp.log1p(adata)

# COMMAND ----------

sc.pp.highly_variable_genes(adata, n_top_genes=parameters['n_top_genes'])

# COMMAND ----------

sc.pl.highly_variable_genes(adata, save="hvg.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### PCA

# COMMAND ----------

sc.tl.pca(adata)

# COMMAND ----------

sc.pl.pca(
    adata,
    color=["pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (1, 2)],
    ncols=2,
    size=2,
    save='pca.png'
)

# COMMAND ----------

# optionally add PCA coords into cell (obs) table
for i in range(4): #adata._obsm['X_pca'].shape[1]):
    adata.obs['PCA_'+str(i)] = adata._obsm['X_pca'][:,i]

# COMMAND ----------

# DBTITLE 1,Batch Correction (Harmony)
# MAGIC %md
# MAGIC #### Batch Correction (Harmony)

# COMMAND ----------

# DBTITLE 1,Harmony batch correction
# Harmony batch correction — integrated from CellAtria/CellExpress pipeline
if parameters['batch_correction'] == 'harmony':
    batch_key = parameters['batch_key'].strip()
    
    if not batch_key:
        print("⚠️ Batch correction requested but no batch_key provided. Skipping.")
    elif batch_key not in adata.obs.columns:
        print(f"⚠️ Batch key '{batch_key}' not found in adata.obs columns: {list(adata.obs.columns)[:10]}... Skipping.")
    else:
        import harmonypy
        
        n_batches = adata.obs[batch_key].nunique()
        print(f"Running Harmony batch correction on '{batch_key}' ({n_batches} batches)...")
        
        # Store uncorrected PCA for comparison
        adata.obsm['X_pca_uncorrected'] = adata.obsm['X_pca'].copy()
        
        # Run Harmony
        harmony_out = harmonypy.run_harmony(
            adata.obsm['X_pca'],
            adata.obs,
            batch_key,
            max_iter_harmony=20
        )
        
        # Replace PCA embeddings with Harmony-corrected ones
        adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
        
        metrics['batch_correction'] = 'harmony'
        metrics['n_batches'] = int(n_batches)
        print(f"Harmony correction complete. PCA embeddings updated.")
else:
    print("Batch correction: skipped (method=none)")
    metrics['batch_correction'] = 'none'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Do UMAP

# COMMAND ----------

sc.pp.neighbors(adata)

# COMMAND ----------

# DBTITLE 1,Clustering (Leiden or Louvain)
# MAGIC %md
# MAGIC #### Clustering (Leiden or Louvain)

# COMMAND ----------

# DBTITLE 1,Dual-method clustering
# Clustering — supports both Leiden (from CellExpress/Cellatria) and Louvain (original GWB)
if parameters['clustering_method'] == 'leiden':
    # Leiden clustering (recommended — used by CellExpress pipeline)
    sc.tl.leiden(
        adata,
        resolution=parameters['cluster_resolution'],
        n_iterations=10,
        flavor='leidenalg',
    )
    adata.obs['cluster'] = adata.obs['leiden']
    method_name = "Leiden"
else:
    # Louvain clustering (original GWB method via scikit-network)
    from sknetwork.clustering import Louvain
    louvain = Louvain(
        resolution=parameters['cluster_resolution'],
        modularity="dugue",
        shuffle_nodes=True,
        sort_clusters=True,
        random_state=0,
        verbose=False
    )
    adjacency = adata.obsp['connectivities']
    labels = louvain.fit_predict(adjacency)
    adata.obs['cluster'] = pd.Categorical(labels.astype(str))
    method_name = "Louvain"

n_clusters = len(adata.obs['cluster'].unique())
print(f"{method_name} clustering complete. Found {n_clusters} clusters at resolution {parameters['cluster_resolution']}.")
metrics['clustering_method'] = method_name.lower()
metrics['n_clusters'] = int(n_clusters)

# COMMAND ----------

sc.tl.umap(adata)

for i in range(2):
    adata.obs['UMAP_'+str(i)] = adata._obsm['X_umap'][:,i]

# COMMAND ----------

sc.pl.umap(
    adata,
    color="cluster",
    size=2,
    save='umap_cluster.png'
)

# COMMAND ----------

# DBTITLE 1,UMAP doublet score visualization
# Visualize doublet scores on UMAP (if Scrublet was run)
if parameters['doublet_method'] == 'scrublet' and 'doublet_score' in adata.obs.columns:
    # Note: doublets already removed, but scores of retained cells show distribution
    sc.pl.umap(adata, color="doublet_score", size=2, save='umap_doublet_scores.png')

# COMMAND ----------

# DBTITLE 1,UMAP batch visualization
# Visualize batch correction results on UMAP
if parameters['batch_correction'] == 'harmony' and parameters['batch_key'].strip() in adata.obs.columns:
    sc.pl.umap(adata, color=parameters['batch_key'].strip(), size=2, save='umap_batch.png')

# COMMAND ----------

# DBTITLE 1,Cell Type Annotation header
# MAGIC %md
# MAGIC #### Cell Type Annotation (SCimilarity / CellTypist)

# COMMAND ----------

# DBTITLE 1,SCimilarity / CellTypist annotation
# Cell type annotation — integrated from CellAtria/CellExpress pipeline
import gc

if parameters['annotation_method'] == 'scimilarity':
    from scimilarity.utils import lognorm_counts, align_dataset
    from scimilarity.cell_annotation import CellAnnotation

    model_path = parameters['annotation_model'].strip()
    if not model_path:
        print("Warning: SCimilarity selected but no model path provided. Skipping.")
        metrics['annotation_method'] = 'scimilarity_skipped'
        metrics['n_celltypes_annotated'] = 0
    else:
        print(f"Running SCimilarity annotation with model: {model_path}")
        gc.collect()  # Free memory before loading model

        # Initialize annotation engine
        ca = CellAnnotation(model_path=model_path)

        # Align dataset genes to model's expected gene order
        adata_aligned = align_dataset(adata, ca.gene_order)
        n_aligned = adata_aligned.shape[1]
        n_total = adata.shape[1]
        print(f"Aligned {n_aligned}/{n_total} genes to SCimilarity model gene set")

        # Get cell embeddings from pre-trained foundation model
        embeddings = ca.get_embeddings(adata_aligned.X)
        adata.obsm['X_scimilarity'] = embeddings

        # Free aligned dataset before loading kNN index (~8.6 GB)
        del adata_aligned
        gc.collect()

        # Annotate cell types via kNN search against ~23M cell reference atlas
        print("Loading kNN index and predicting cell types...")
        knn_result = ca.get_predictions_knn(embeddings)
        
        # Handle variable return signature across SCimilarity versions
        if isinstance(knn_result, tuple):
            print(f"  kNN returned {len(knn_result)} values")
            predictions = knn_result[0]  # Cell type predictions are always first
            # Try to extract distances (usually last element)
            nn_dists = knn_result[-1] if len(knn_result) > 1 else None
        else:
            predictions = knn_result
            nn_dists = None

        adata.obs['celltype_annotation'] = predictions
        if nn_dists is not None:
            import numpy as np
            if hasattr(nn_dists, 'mean'):
                adata.obs['celltype_nn_dist'] = nn_dists.mean(axis=1) if nn_dists.ndim > 1 else nn_dists

        # Free kNN objects to reclaim memory
        del knn_result, embeddings
        if hasattr(ca, 'reset_knn'):
            ca.reset_knn()
        del ca
        gc.collect()

        n_types = adata.obs['celltype_annotation'].nunique()
        type_counts = adata.obs['celltype_annotation'].value_counts()
        print(f"\nSCimilarity annotated {n_types} cell types:")
        print(type_counts.head(15).to_string())

        metrics['annotation_method'] = 'scimilarity'
        metrics['n_celltypes_annotated'] = int(n_types)
        metrics['annotation_genes_aligned'] = int(n_aligned)

elif parameters['annotation_method'] == 'celltypist':
    import celltypist
    from celltypist import models as ct_models

    model_name = parameters['annotation_model'].strip()
    if not model_name:
        model_name = 'Immune_All_Low.pkl'
        print(f"No model specified, using default: {model_name}")

    print(f"Running CellTypist annotation with model: {model_name}")
    ct_models.download_models(model=model_name)
    model = ct_models.Model.load(model=model_name)

    predictions = celltypist.annotate(
        adata,
        model=model,
        majority_voting=True,
        over_clustering='cluster'
    )
    adata_annotated = predictions.to_adata()

    adata.obs['celltype_annotation'] = adata_annotated.obs['predicted_labels']
    adata.obs['celltype_majority_voting'] = adata_annotated.obs['majority_voting']
    adata.obs['celltype_conf_score'] = adata_annotated.obs['conf_score']

    del adata_annotated, predictions, model
    gc.collect()

    n_types = adata.obs['celltype_annotation'].nunique()
    type_counts = adata.obs['celltype_annotation'].value_counts()
    print(f"\nCellTypist annotated {n_types} cell types:")
    print(type_counts.head(15).to_string())

    metrics['annotation_method'] = 'celltypist'
    metrics['n_celltypes_annotated'] = int(n_types)

else:
    print("Cell type annotation: skipped (method=none)")
    metrics['annotation_method'] = 'none'
    metrics['n_celltypes_annotated'] = 0

# COMMAND ----------

# DBTITLE 1,UMAP annotation visualization
# Visualize cell type annotations on UMAP
if parameters['annotation_method'] != 'none' and 'celltype_annotation' in adata.obs.columns:
    sc.pl.umap(
        adata,
        color='celltype_annotation',
        size=2,
        legend_loc='on data',
        legend_fontsize=6,
        save='umap_celltype_annotation.png'
    )

    # CellTypist majority voting view
    if 'celltype_majority_voting' in adata.obs.columns:
        sc.pl.umap(
            adata,
            color='celltype_majority_voting',
            size=2,
            legend_loc='on data',
            legend_fontsize=6,
            save='umap_celltype_majority_voting.png'
        )

    # Annotation quality: kNN distance (SCimilarity) or confidence score (CellTypist)
    if 'celltype_nn_dist' in adata.obs.columns:
        sc.pl.umap(adata, color='celltype_nn_dist', size=2, save='umap_annotation_distance.png')
    elif 'celltype_conf_score' in adata.obs.columns:
        sc.pl.umap(adata, color='celltype_conf_score', size=2, save='umap_annotation_confidence.png')

    # Save annotation counts CSV for MLflow artifact
    type_counts = adata.obs['celltype_annotation'].value_counts()
    type_counts.to_csv(tmpdir.name + '/celltype_annotation_counts.csv')
    print(f"Saved annotation counts ({len(type_counts)} types) to MLflow artifacts")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find marker genes for each cluster

# COMMAND ----------

# Run differential expression analysis
sc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon")

# COMMAND ----------

# Visualize top markers
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, save='_marker_genes.png')

# COMMAND ----------

# Extract top N marker genes per cluster
n_markers_per_cluster = 10  # Adjust as needed

# Get all unique marker genes across all clusters
marker_genes = set()
n_clusters = len(adata.obs['cluster'].unique())

for i in range(n_clusters):
    cluster_markers = adata.uns['rank_genes_groups']['names'][str(i)][:n_markers_per_cluster]
    marker_genes.update(cluster_markers)

marker_genes = list(marker_genes)
print(f"Total unique marker genes: {len(marker_genes)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create marker gene dataframe for logging

# COMMAND ----------

# Create a DataFrame with top markers per cluster (for visualization)
marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(n_markers_per_cluster)
marker_df.to_csv(tmpdir.name + "/top_markers_per_cluster.csv", index=False)

# Also save full marker gene list
pd.DataFrame({'marker_genes': marker_genes}).to_csv(
    tmpdir.name + "/top_marker_genes.csv", 
    index=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subsample the data to a maximum number of cells

# COMMAND ----------

import scipy
import numpy as np

adata_markers = adata[:, marker_genes].copy()

max_cells = 10000
if adata_markers.shape[0] > max_cells:
    # Get cluster counts and proportions
    cluster_counts = adata_markers.obs['cluster'].value_counts().sort_index()
    total_cells = cluster_counts.sum()
    metrics['total_cells_before_subsample'] = float(total_cells)
    
    # Calculate target cells per cluster (proportional)
    target_per_cluster = {}
    for cluster_id, count in cluster_counts.items():
        proportion = count / total_cells
        target_per_cluster[cluster_id] = int(np.round(proportion * max_cells))
    
    # Adjust to ensure exactly max_cells (handle rounding errors)
    current_total = sum(target_per_cluster.values())
    if current_total != max_cells:
        diff = max_cells - current_total
        # Adjust the largest cluster
        largest_cluster = cluster_counts.idxmax()
        target_per_cluster[largest_cluster] += diff
    
    # Perform stratified sampling
    np.random.seed(42)
    sampled_indices = []
    
    for cluster_id, target_n in target_per_cluster.items():
        # Get all cells in this cluster
        cluster_mask = adata_markers.obs['cluster'] == cluster_id
        cluster_cells = adata_markers.obs.index[cluster_mask]
        
        # Sample the target number (or all if cluster is smaller than target)
        n_to_sample = min(target_n, len(cluster_cells))
        sampled_cells = np.random.choice(cluster_cells, size=n_to_sample, replace=False)
        sampled_indices.extend(sampled_cells)
    
    # Create the subsampled object
    adata_markers = adata_markers[sampled_indices, :].copy()
    print(f"Subsampled to {len(adata_markers)} cells")
    print("\nCells per cluster (before → after):")
    for cluster_id in sorted(cluster_counts.index):
        before = cluster_counts[cluster_id]
        after = (adata_markers.obs['cluster'] == cluster_id).sum()
        pct_before = 100 * before / total_cells
        pct_after = 100 * after / len(adata_markers)
        print(f"  Cluster {cluster_id}: {before:>6} ({pct_before:>5.1f}%) → {after:>5} ({pct_after:>5.1f}%)")
        # add to metrics
        metrics[f"cluster_{cluster_id}_pct_before_subample"] = pct_before
        metrics[f"cluster_{cluster_id}_pct_after_subample"] = pct_after
        metrics[f"cluster_{cluster_id}_cells_before_subample"] = before
        metrics[f"cluster_{cluster_id}_cells_after_subample"] = after
else:
    # No subsampling needed
    metrics['total_cells_before_subsample'] = float(adata_markers.shape[0])

# Convert to a flat DataFrame
df_flat = adata_markers.obs.copy()

# Add marker gene expression as columns
# Convert sparse matrix to dense if needed
if scipy.sparse.issparse(adata_markers.X):
    expression_matrix = adata_markers.X.toarray()
else:
    expression_matrix = adata_markers.X

# Add each marker gene as a column
for i, gene in enumerate(marker_genes):
    df_flat[f"expr_{gene}"] = expression_matrix[:, i]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mlflow to track runs with parameters
# MAGIC
# MAGIC  - save processed runs as adata to mlflow, also with parameters, metrics. Can place these in adata.uns also.
# MAGIC  - But keeping track of experiments with varying parameters can be useful for later review
# MAGIC  - **mlflow** often used in both classic ML and agentic AI offers some features that can be useful `here` 

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# Set the MLflow experiment (use path-based name for Databricks)
exp_name = parameters['mlflow_experiment']
if not exp_name.startswith('/'):
    # Convert simple name to path-based experiment under user's folder
    user_email = parameters.get('user_email', 'default')
    exp_name = f"/Users/{user_email}/{exp_name}"

experiment = mlflow.set_experiment(exp_name)
print(f"MLflow experiment: {experiment.name} (ID: {experiment.experiment_id})")

# Tag the experiment for Genesis Workbench search
mlflow.set_experiment_tags({
    "used_by_genesis_workbench": "yes"
})

# COMMAND ----------

# save adata and our figures to disk
# Drop Gene name column before saving (only needed during analysis, causes conflict with index)
adata.var = adata.var.drop(columns=['Gene name'], errors='ignore')

# Ensure all var columns are h5ad-compatible (convert NaN and non-string objects)
for col in adata.var.columns:
    if adata.var[col].dtype == object or str(adata.var[col].dtype) == 'category':
        adata.var[col] = adata.var[col].fillna('').astype(str)

adata.write_h5ad(tmpdir.name+"/adata_output.h5ad")
# save the flat dataframe to disk
df_flat.to_parquet(tmpdir.name + "/markers_flat.parquet")


t1 = time.time()
total_time = t1-t0
metrics['total_time'] = total_time

run_name = parameters['mlflow_run_name'] if parameters['mlflow_run_name'] else None

# Separate numeric metrics from string metrics for MLflow
numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
string_metrics = {k: str(v) for k, v in metrics.items() if not isinstance(v, (int, float))}

# Log to MLflow with proper tags for Genesis Workbench
with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
    # Log metrics and params
    mlflow.log_metrics(numeric_metrics)
    mlflow.log_params(parameters)
    # Log string metrics as tags
    for k, v in string_metrics.items():
        mlflow.set_tag(f"metric_{k}", v)
    mlflow.log_artifacts(tmpdir.name)
    
    # Set required tags for Genesis Workbench search
    mlflow.set_tag("origin", "genesis_workbench")
    mlflow.set_tag("created_by", parameters['user_email'])
    mlflow.set_tag("processing_mode", "scanpy")

print(f"MLflow run logged: {run.info.run_id}")
print(f"Experiment: {experiment.name}")
print(f"Metrics: {numeric_metrics}")
print(f"Total time: {total_time:.1f}s")

# COMMAND ----------

# DBTITLE 1,Log lineage to metadata tier
# =============================================================================
# Lineage Logging — Metadata Tier Integration
# Records the scanpy analysis as a lineage edge: input h5ad → annotated h5ad (platinum)
# =============================================================================

try:
    from genesis_workbench.lineage import LineageLogger

    lineage = LineageLogger(
        module="single_cell",
        run_id=run.info.run_id,
        user_email=parameters["user_email"],
        run_source="mlflow",
        catalog=parameters["catalog"] or "dhbl_discovery_us_dev",
        schema=parameters["schema"] or "genesis_schema",
    )

    # ── Register input: source h5ad (gold — analysis-ready count matrix) ──
    input_h5ad = lineage.register_asset(
        path=parameters["data_path"],
        asset_type="volume_file",
        tier="gold",
        format="h5ad",
        display_name=f"Input h5ad ({parameters.get('mlflow_run_name', 'scanpy')})",
        description=f"{int(metrics.get('total_cells_starting', 0))} cells before filtering",
        tags={
            "species": parameters["species"],
            "n_cells_raw": str(int(metrics.get("total_cells_starting", 0))),
        },
    )

    # ── Register output: annotated h5ad (platinum — scientist end result) ──
    output_h5ad = lineage.register_asset(
        path=f"mlflow-artifacts://runs/{run.info.run_id}/adata_output.h5ad",
        asset_type="mlflow_artifact",
        tier="platinum",
        format="h5ad",
        display_name=f"Annotated h5ad ({parameters.get('mlflow_run_name', 'scanpy')})",
        description=(
            f"{adata.n_obs} cells, {adata.n_vars} genes, "
            f"{int(metrics.get('n_clusters', 0))} clusters, "
            f"{parameters.get('annotation_method', 'none')} annotation"
        ),
        row_count=adata.n_obs,
        tags={
            "species": parameters["species"],
            "n_cells": str(adata.n_obs),
            "n_genes": str(adata.n_vars),
            "n_clusters": str(int(metrics.get("n_clusters", 0))),
            "doublet_method": parameters.get("doublet_method", "none"),
            "batch_correction": parameters.get("batch_correction", "none"),
            "clustering_method": parameters.get("clustering_method", "leiden"),
            "annotation_method": parameters.get("annotation_method", "none"),
            "annotation_model": parameters.get("annotation_model", ""),
        },
    )

    # ── Register output: markers parquet (platinum) ──
    markers_asset = lineage.register_asset(
        path=f"mlflow-artifacts://runs/{run.info.run_id}/markers_flat.parquet",
        asset_type="mlflow_artifact",
        tier="platinum",
        format="parquet",
        display_name=f"Marker Gene Matrix ({parameters.get('mlflow_run_name', 'scanpy')})",
        description=f"Flat dataframe with cluster assignments + marker gene expression",
        tags={"n_marker_genes": str(len(marker_genes)) if 'marker_genes' in dir() else "0"},
    )

    # ── Record lineage edges ──
    lineage.link(
        source=input_h5ad,
        target=output_h5ad,
        relationship="consumed_by",
        step="scanpy_pipeline",
    )
    lineage.link(
        source=input_h5ad,
        target=markers_asset,
        relationship="consumed_by",
        step="marker_gene_extraction",
    )

    # ── Register the run ──
    lineage.register_run(
        run_name=parameters.get("mlflow_run_name", "scanpy_analysis"),
        experiment_name=parameters.get("mlflow_experiment", ""),
        status="completed",
        parameters={
            k: str(v) for k, v in parameters.items()
            if k not in ("data_path",)  # exclude long paths from params map
        },
        metrics={
            k: float(v) for k, v in numeric_metrics.items()
        },
        start_time=None,  # Could add t0 if converted to datetime
        end_time=None,
        tags={
            "doublet_method": parameters.get("doublet_method", "none"),
            "batch_correction": parameters.get("batch_correction", "none"),
            "annotation_method": parameters.get("annotation_method", "none"),
            "species": parameters["species"],
        },
    )

    # ── Flush to Delta ──
    result = lineage.flush()
    print(f"\n\U0001f4cb Lineage logged: {result['assets_written']} assets, "
          f"{result['edges_written']} edges, {result['run_written']} run")

except ImportError:
    print("\u26a0\ufe0f  genesis_workbench.lineage not available — skipping lineage logging")
    print("   Install wheel v0.1.3+ to enable metadata tier integration")
except Exception as e:
    # Non-fatal: lineage logging should never break the analysis
    print(f"\u26a0\ufe0f  Lineage logging failed (non-fatal): {e}")

# COMMAND ----------


