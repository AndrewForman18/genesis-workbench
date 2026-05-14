
import streamlit as st
import pandas as pd
import time
import os
from genesis_workbench.models import (ModelCategory, 
                                      get_available_models, 
                                      get_deployed_models)

from utils.streamlit_helper import (display_import_model_uc_dialog,
                                    display_deploy_model_dialog,
                                    get_user_info)
from utils.authorization import require_module_access
from utils.single_cell_analysis import (start_scanpy_job, 
                                        start_rapids_singlecell_job,
                                        download_singlecell_markers_df,
                                        download_cluster_markers_mapping,
                                        get_mlflow_run_url,
                                        search_singlecell_runs)
import plotly.express as px
from mlflow.tracking import MlflowClient
from utils.geo_data_ingestion import fetch_geo_metadata, download_geo_to_volume, detect_data_format, create_metadata_csv
from utils.sra_data_ingestion import fetch_sra_metadata, download_sra_to_volume, detect_sra_data
from utils.fetchngs_pipeline import (fetch_ena_metadata, validate_accessions, classify_accession,
                                       estimate_download_size, start_fetchngs_job,
                                       SUPPORTED_DOWNSTREAM_PIPELINES, DOWNLOAD_METHODS)
from utils.nextflow_pipeline import (ALIGNERS, GENOME_PRESETS, CHEMISTRY_PRESETS, COMPUTE_TIERS,
                                        build_new_cluster_spec, build_samplesheet,
                                        split_samplesheet_by_sample, build_parallel_tasks)

def reset_available_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["available_single_cell_models_df"]
        st.rerun()

def reset_deployed_models():
    with st.spinner("Refreshing data.."):
        time.sleep(1)
        del st.session_state["deployed_single_cell_models_df"]
        st.rerun()

def display_settings_tab(available_models_df,deployed_models_df):

    p1,p2 = st.columns([2,1])

    with p1:
        st.markdown("###### Import Models:")
        with st.form("import_model_form"):
            col1, col2, = st.columns([1,1], vertical_alignment="bottom")    
            with col1:
                import_model_source = st.selectbox("Source:",["Unity Catalog","Hugging Face","PyPi"],label_visibility="visible")

            with col2:
                import_button = st.form_submit_button('Import')
        
        if import_button:
            if import_model_source=="Unity Catalog":
                display_import_model_uc_dialog(ModelCategory.SINGLE_CELL, success_callback=reset_available_models)


        st.markdown("###### Available Models:")
        with st.form("deploy_model_form"):
            col1, col2, = st.columns([1,1])    
            with col1:
                selected_model_for_deploy = st.selectbox("Model:",available_models_df["model_labels"],label_visibility="collapsed",)

            with col2:
                deploy_button = st.form_submit_button('Deploy')
        if deploy_button:
            display_deploy_model_dialog(selected_model_for_deploy)


    if len(deployed_models_df) > 0:
        with st.form("modify_deployed_model_form"):
            col1,col2 = st.columns([2,1])
            with col1:
                st.markdown("###### Deployed Models")
            with col2:
                st.form_submit_button("Manage")
            
            st.dataframe(deployed_models_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            column_config={
                                "Model Id": None,
                                "Deploy Id" : None,
                                "Endpoint Name" : None
                            })
    else:
        st.write("There are no deployed models")


def display_scanpy_analysis_tab():
    st.markdown("###### Run Analysis")

    # Mode selection OUTSIDE form so defaults update on change
    st.markdown("**Analysis Mode:**")
    mode_display = st.selectbox(
        "Mode",
        options=["scanpy", "rapids-singlecell [GPU-accelerated]"],
        label_visibility="collapsed",
        key="scanpy_mode_selector"
    )
    mode = "rapids-singlecell" if "rapids-singlecell" in mode_display else mode_display
    default_experiment = "rapidssinglecell_genesis_workbench" if mode == "rapids-singlecell" else "scanpy_genesis_workbench"
    default_run_name = "rapidssinglecell_analysis" if mode == "rapids-singlecell" else "scanpy_analysis"

    with st.form("scanpy_analysis_form", enter_to_submit=False):
        st.divider()

        # Data Input
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Data Configuration:**")
            data_path = st.text_input(
                "Data Path (h5ad file)",
                placeholder="/Volumes/catalog/schema/volume/file.h5ad",
                help="Path to the h5ad file in Unity Catalog Volumes"
            )
            gene_name_column = st.text_input(
                "Gene Name Column (optional)",
                value="",
                placeholder="e.g., gene_name, feature_name",
                help="Name of column in var containing gene names. Leave empty to use Ensembl reference mapping. Note: Gene names will be normalized to uppercase for consistent QC analysis."
            )

            # Conditional species selector
            if not gene_name_column or gene_name_column.strip() == "":
                species = st.selectbox(
                    "Species",
                    options=["hsapiens", "mmusculus", "rnorvegicus"],
                    index=0,
                    help="Species for Ensembl gene name mapping. Required when gene name column is not provided. Note: All gene names are normalized to uppercase for consistent QC detection (mitochondrial, ribosomal genes) regardless of input capitalization.",
                    format_func=lambda x: {
                        "hsapiens": "Human (Homo sapiens)",
                        "mmusculus": "Mouse (Mus musculus)",
                        "rnorvegicus": "Rat (Rattus norvegicus)"
                    }[x]
                )
            else:
                species = "hsapiens"  # Default, won't be used since gene_name_column is provided

            st.markdown("**MLflow Tracking:**")
            mlflow_experiment = st.text_input(
                "MLflow Experiment Name",
                value=default_experiment,
                help="Simple experiment name (will be created in your MLflow folder)"
            )
            mlflow_run_name = st.text_input(
                "MLflow Run Name",
                value=default_run_name,
                help="Name for this specific analysis run"
            )
        
        with col2:
            st.markdown("**Filtering Parameters:**")
            min_genes = st.number_input(
                "Min Genes per Cell",
                min_value=0,
                value=200,
                step=10
            )
            min_cells = st.number_input(
                "Min Cells per Gene",
                min_value=0,
                value=3,
                step=1
            )
            pct_counts_mt = st.number_input(
                "Max % Mitochondrial Counts",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1
            )
            n_genes_by_counts = st.number_input(
                "Max Genes by Counts",
                min_value=0,
                value=2500,
                step=100
            )
        
        st.divider()
        
        # Analysis Parameters
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("**Normalization & Feature Selection:**")
            target_sum = st.number_input(
                "Target Sum for Normalization",
                min_value=0,
                value=10000,
                step=1000
            )
            n_top_genes = st.number_input(
                "Number of Highly Variable Genes",
                min_value=0,
                value=500,
                step=50
            )
        
        with col4:
            st.markdown("**Dimensionality Reduction & Clustering:**")
            n_pcs = st.number_input(
                "Number of Principal Components",
                min_value=0,
                value=50,
                step=5
            )
            cluster_resolution = st.number_input(
                "Cluster Resolution",
                min_value=0.0,
                max_value=2.0,
                value=0.15,
                step=0.05,
                format="%.2f"
            )
        
        # CellExpress/Cellatria integration — advanced pipeline options
        st.divider()
        st.markdown("**Advanced Pipeline Options** *(integrated from CellExpress/Cellatria)*")
        col5, col6 = st.columns([1, 1])
        
        with col5:
            st.markdown("**Doublet Detection:**")
            doublet_method = st.selectbox(
                "Method",
                options=["scrublet", "none"],
                index=0,
                help="Scrublet detects and removes cell doublets. Recommended for most datasets."
            )
            if doublet_method == "scrublet":
                scrublet_threshold = st.number_input(
                    "Scrublet Score Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    step=0.05,
                    format="%.2f",
                    help="Cells with doublet scores above this threshold are removed (default: 0.25)"
                )
            else:
                scrublet_threshold = 0.25
        
        with col6:
            st.markdown("**Batch Correction & Clustering:**")
            batch_correction = st.selectbox(
                "Batch Correction",
                options=["none", "harmony"],
                index=0,
                help="Harmony corrects batch effects in PCA space. Requires a batch key column in your data."
            )
            if batch_correction == "harmony":
                batch_key = st.text_input(
                    "Batch Key Column",
                    value="",
                    placeholder="e.g., sample, batch, donor",
                    help="Column in adata.obs identifying batches. Required for Harmony."
                )
            else:
                batch_key = ""
            
            clustering_method = st.selectbox(
                "Clustering Method",
                options=["leiden", "louvain"],
                index=0,
                help="Leiden (recommended, from CellExpress) or Louvain (original GWB)"
            )
        
        # Cell type annotation (SCimilarity/CellTypist from Cellatria)
        st.divider()
        st.markdown("**Cell Type Annotation** *(SCimilarity / CellTypist from Cellatria)*")
        col7, col8 = st.columns([1, 1])
        
        with col7:
            annotation_method = st.selectbox(
                "Annotation Method",
                options=["none", "scimilarity", "celltypist"],
                index=0,
                help="SCimilarity uses a foundation model + kNN against a 23M cell atlas. CellTypist uses logistic regression with majority voting."
            )
        
        with col8:
            if annotation_method == "scimilarity":
                annotation_model = st.text_input(
                    "SCimilarity Model Path",
                    value="",
                    placeholder="/Volumes/catalog/schema/vol/scimilarity_model/",
                    help="Path to the downloaded SCimilarity model directory in a UC Volume."
                )
            elif annotation_method == "celltypist":
                annotation_model = st.selectbox(
                    "CellTypist Model",
                    options=[
                        "Immune_All_Low.pkl",
                        "Immune_All_High.pkl",
                        "Human_Lung_Atlas.pkl",
                        "Developing_Human_Brain.pkl",
                        "Cells_Intestinal_Tract.pkl",
                        "Human_Colorectal_Cancer.pkl",
                        "Pan_Fetal_Human.pkl",
                    ],
                    index=0,
                    help="Pre-trained CellTypist model. Downloaded automatically at runtime."
                )
            else:
                annotation_model = ""
        
        st.divider()
        submit_button = st.form_submit_button("Start Analysis", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        is_valid = True
        
        # Validation
        if not data_path.strip():
            st.error("Please provide a data path")
            is_valid = False
        elif not data_path.endswith(".h5ad"):
            st.error("Data path must point to an .h5ad file")
            is_valid = False
        elif not data_path.startswith("/Volumes"):
            st.error("Data path must start with /Volumes (Unity Catalog Volume)")
            is_valid = False
        
        if not mlflow_experiment.strip() or not mlflow_run_name.strip():
            st.error("Please provide MLflow experiment and run names")
            is_valid = False
        
        # Validate gene name column or species is provided
        if (not gene_name_column or gene_name_column.strip() == "") and (not species or species == ""):
            st.error("❌ Please either provide a Gene Name Column OR select a Species for reference mapping.")
            is_valid = False
        
        if is_valid:
            user_info = get_user_info()
            
            if mode == "scanpy":
                try:
                    with st.spinner("Starting scanpy analysis job..."):
                        scanpy_job_id, job_run_id = start_scanpy_job(
                            data_path=data_path,
                            mlflow_experiment=mlflow_experiment,
                            mlflow_run_name=mlflow_run_name,
                            gene_name_column=gene_name_column,
                            species=species,
                            min_genes=min_genes,
                            min_cells=min_cells,
                            pct_counts_mt=pct_counts_mt,
                            n_genes_by_counts=n_genes_by_counts,
                            target_sum=target_sum,
                            n_top_genes=n_top_genes,
                            n_pcs=n_pcs,
                            cluster_resolution=cluster_resolution,
                            doublet_method=doublet_method,
                            scrublet_threshold=scrublet_threshold,
                            batch_correction=batch_correction,
                            batch_key=batch_key,
                            clustering_method=clustering_method,
                        annotation_method=annotation_method,
                        annotation_model=annotation_model,
                            user_info=user_info
                        )
                        
                        # Construct the run URL
                        host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                        if not host_name.startswith("https://"):
                            host_name = "https://" + host_name
                        run_url = f"{host_name}/jobs/{scanpy_job_id}/runs/{job_run_id}"
                        
                        st.success(f"✅ Job started successfully! Run ID: {job_run_id}")
                        st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                
                except Exception as e:
                    st.error(f"❌ An error occurred while starting the job: {str(e)}")
                    print(e)
            
            elif mode == "rapids-singlecell":
                try:
                    with st.spinner("Starting rapids-singlecell analysis job..."):
                        rapids_job_id, job_run_id = start_rapids_singlecell_job(
                            data_path=data_path,
                            mlflow_experiment=mlflow_experiment,
                            mlflow_run_name=mlflow_run_name,
                            gene_name_column=gene_name_column,
                            species=species,
                            min_genes=min_genes,
                            min_cells=min_cells,
                            pct_counts_mt=pct_counts_mt,
                            n_genes_by_counts=n_genes_by_counts,
                            target_sum=target_sum,
                            n_top_genes=n_top_genes,
                            n_pcs=n_pcs,
                            cluster_resolution=cluster_resolution,
                            doublet_method=doublet_method,
                            scrublet_threshold=scrublet_threshold,
                            batch_correction=batch_correction,
                            batch_key=batch_key,
                            clustering_method=clustering_method,
                        annotation_method=annotation_method,
                        annotation_model=annotation_model,
                            user_info=user_info
                        )
                        
                        # Construct the run URL
                        host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                        if not host_name.startswith("https://"):
                            host_name = "https://" + host_name
                        run_url = f"{host_name}/jobs/{rapids_job_id}/runs/{job_run_id}"
                        
                        st.success(f"✅ Job started successfully! Run ID: {job_run_id}")
                        st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                
                except Exception as e:
                    st.error(f"❌ An error occurred while starting the job: {str(e)}")
                    print(e)
            
            else:
                st.error(f"❌ Unknown mode: {mode}")


def display_singlecell_results_viewer():
    """Interactive viewer for single-cell analysis results (scanpy, rapids-singlecell, etc.)"""
    
    # Helper function to find cluster column (supports legacy and new naming)
    def get_cluster_column(df):
        """Return the cluster column name (prioritizes 'cluster', falls back to 'leiden' or 'louvain')"""
        if 'cluster' in df.columns:
            return 'cluster'
        elif 'leiden' in df.columns:
            return 'leiden'
        elif 'louvain' in df.columns:
            return 'louvain'
        return None
    
    st.markdown("##### Single-Cell Results Viewer")
    st.markdown("Select a completed single-cell analysis run to visualize")
    
    # Get user info
    user_info = get_user_info()
    
    # Initialize session state for filters
    if 'date_filter' not in st.session_state:
        st.session_state['date_filter'] = None  # All time by default
    if 'processing_mode_filter' not in st.session_state:
        st.session_state['processing_mode_filter'] = "All"
    
    # Compact top row: Experiment (50%), Mode (15%), Time filters (25%), Refresh (10%)
    main_col1, main_col2, main_col3, main_col4 = st.columns([5, 1.5, 2.5, 1], vertical_alignment="bottom")
    
    with main_col1:
        experiment_filter = st.text_input(
            "MLflow Experiment:",
            value="genesis_workbench",
            help="Enter experiment name to filter runs (partial match supported). Default shows both scanpy and rapids-singlecell experiments."
        )
    
    with main_col2:
        processing_mode_option = st.radio(
            "Mode:",
            ["All", "Scanpy", "Rapids-SingleCell"],
            index=["All", "Scanpy", "Rapids-SingleCell"].index(st.session_state['processing_mode_filter']),
            help="Filter by processing pipeline",
            label_visibility="visible"
        )
        if processing_mode_option != st.session_state['processing_mode_filter']:
            st.session_state['processing_mode_filter'] = processing_mode_option
            if 'singlecell_runs_df' in st.session_state:
                del st.session_state['singlecell_runs_df']
    
    with main_col3:
        st.markdown("**Time Period:**")
        # 2x2 grid for time filters
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if st.button("Today", use_container_width=True, help="Show runs from today"):
                st.session_state['date_filter'] = 0
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
        with row1_col2:
            if st.button("Last 7 Days", use_container_width=True, help="Show runs from last week"):
                st.session_state['date_filter'] = 7
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
        
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if st.button("Last 30 Days", use_container_width=True, help="Show runs from last month"):
                st.session_state['date_filter'] = 30
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
        with row2_col2:
            if st.button("All Time", use_container_width=True, help="Show all runs"):
                st.session_state['date_filter'] = None
                if 'singlecell_runs_df' in st.session_state:
                    del st.session_state['singlecell_runs_df']
    
    with main_col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        refresh_button = st.button("🔄 Refresh", use_container_width=True, help="Reload available runs from MLflow")
    
    # Load or refresh runs list
    if refresh_button or 'singlecell_runs_df' not in st.session_state:
        with st.spinner("Searching for your single-cell analysis runs..."):
            try:
                # Convert processing mode to lowercase for tag matching
                processing_mode = None
                if st.session_state['processing_mode_filter'] != "All":
                    processing_mode = st.session_state['processing_mode_filter'].lower().replace("-", "-")
                    if processing_mode == "rapids-singlecell":
                        processing_mode = "rapids-singlecell"
                    elif processing_mode == "scanpy":
                        processing_mode = "scanpy"
                
                runs_df = search_singlecell_runs(
                    user_email=user_info.user_email,
                    processing_mode=processing_mode,
                    days_back=st.session_state['date_filter']
                )
                st.session_state['singlecell_runs_df'] = runs_df
                if len(runs_df) == 0:
                    st.info("No single-cell analysis runs found. Run an analysis first!")
            except Exception as e:
                st.error(f"❌ Error searching runs: {str(e)}")
                return
    
    runs_df = st.session_state.get('singlecell_runs_df', pd.DataFrame())
    
    if len(runs_df) == 0:
        st.info("💡 No single-cell runs found. Go to 'Run New Analysis' to create one!")
        return
    
    # Apply experiment text filter
    if experiment_filter and experiment_filter.strip():
        runs_df = runs_df[runs_df['experiment'].str.contains(experiment_filter.strip(), case=False, na=False)]
        if len(runs_df) == 0:
            st.warning(f"No runs found matching experiment: '{experiment_filter}'")
            return
    
    st.divider()
    
    # Sort by most recent (default)
    runs_df = runs_df.sort_values('start_time', ascending=False)
    
    # Create display names with date and status
    runs_df['display_name'] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['experiment']}) - {row['start_time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['start_time'], 'strftime') else row['start_time']}",
        axis=1
    )
    
    run_options = dict(zip(runs_df['display_name'], runs_df['run_id']))
    
    # Consolidated run selection: Run name filter (40%), Runs dropdown (50%), Load button (10%)
    select_col1, select_col2, select_col3 = st.columns([4, 5, 1], vertical_alignment="bottom")
    
    with select_col1:
        run_name_filter = st.text_input(
            "Search by Run Name:",
            value="",
            placeholder="Type to filter runs...",
            help="Enter run name to filter runs (partial match supported)."
        )
    
    # Apply run name text filter
    if run_name_filter and run_name_filter.strip():
        filtered_runs = runs_df[runs_df['run_name'].str.contains(run_name_filter.strip(), case=False, na=False)]
        if len(filtered_runs) == 0:
            st.warning(f"No runs found matching run name: '{run_name_filter}'")
            return
        filtered_options = dict(zip(filtered_runs['display_name'], filtered_runs['run_id']))
    else:
        filtered_runs = runs_df
        filtered_options = run_options
    
    with select_col2:
        st.markdown(f"**{len(filtered_runs)} Runs:**")
        selected_display = st.selectbox(
            "Select:",
            list(filtered_options.keys()),
            help="Runs sorted by most recent. Experiment name in parentheses.",
            label_visibility="collapsed"
        )
    
    with select_col3:
        st.write("")  # Spacer to align button
        load_button = st.button("Load", type="primary", use_container_width=True)
    
    run_id = filtered_options[selected_display]
    
    # Load button action
    if load_button:
        with st.spinner("Loading data from MLflow..."):
            try:
                df = download_singlecell_markers_df(run_id)
                mlflow_url = get_mlflow_run_url(run_id)
                
                st.session_state['singlecell_df'] = df
                st.session_state['singlecell_run_id'] = run_id
                st.session_state['singlecell_mlflow_url'] = mlflow_url
                
                st.success(f"✅ Loaded {len(df):,} cells with {len(df.columns)} features")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("💡 Tip: Make sure this run includes the markers_flat.parquet artifact")
                return
    
    # Step 3: Display if data is loaded
    if 'singlecell_df' in st.session_state:
        df = st.session_state['singlecell_df']
        run_id = st.session_state.get('singlecell_run_id', '')
        mlflow_url = st.session_state.get('singlecell_mlflow_url', '')
        
        st.markdown("---")
        st.markdown("##### Interactive UMAP Visualization")
        
        # Info about subsampled data
        st.info(
            "📊 **Note:** This viewer displays a subsampled dataset (max 10,000 cells) with marker genes only "
            "for faster, interactive plotting. The complete output AnnData object with all genes is available "
            "in the MLflow run (see 'QC & Other Analysis Outputs' section below)."
        )
        
        # Identify column types
        expr_cols = [c for c in df.columns if c.startswith('expr_')]
        obs_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        obs_numerical = df.select_dtypes(include=['number']).columns.tolist()
        # Remove UMAP/PC columns from numerical
        obs_numerical = [c for c in obs_numerical if not c.startswith(('UMAP_', 'PC_'))]
        
        # Controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            color_type = st.selectbox(
                "Color by:",
                ["Cluster", "Marker Gene", "QC Metric"],
                help="Choose what to color the cells by"
            )
        
        with col2:
            if color_type == "Cluster":
                cluster_options = [c for c in obs_categorical if c in ['leiden', 'louvain', 'cluster']]
                if not cluster_options:
                    cluster_options = obs_categorical
                color_col = st.selectbox("Select cluster column:", cluster_options if cluster_options else ['leiden'])
            elif color_type == "Marker Gene":
                # Calculate which cluster each gene is most expressed in
                cluster_col = get_cluster_column(df)
                if not cluster_col:
                    st.warning("⚠️ No cluster column found in data. Cannot annotate genes by cluster.")
                    # Fallback: just use gene names without cluster annotation
                    selected_gene = st.selectbox("Select gene:", sorted([c.replace('expr_', '') for c in expr_cols]))
                    color_col = f"expr_{selected_gene}"
                else:
                    mean_expr_by_cluster = df.groupby(cluster_col)[expr_cols].mean()
                    
                    # For each gene, find the cluster with highest mean expression
                    gene_to_cluster = {}
                    for gene_col in expr_cols:
                        gene_name = gene_col.replace('expr_', '')
                        cluster_with_max = mean_expr_by_cluster[gene_col].idxmax()
                        gene_to_cluster[gene_name] = cluster_with_max
                    
                    # Create annotated gene options: "GENE (Cluster X)"
                    gene_options_annotated = [
                        f"{gene} (Cluster {gene_to_cluster[gene]})" 
                        for gene in sorted(gene_to_cluster.keys())
                    ]
                    
                    # Create a mapping for reverse lookup
                    annotated_to_gene = {
                        f"{gene} (Cluster {gene_to_cluster[gene]})": gene
                        for gene in gene_to_cluster.keys()
                    }
                    
                    selected_gene_annotated = st.selectbox("Select gene:", gene_options_annotated)
                    # Extract the actual gene name from the annotated selection
                    selected_gene = annotated_to_gene[selected_gene_annotated]
                    color_col = f"expr_{selected_gene}"
            else:  # QC Metric
                metric_options = [c for c in obs_numerical if c in ['n_genes', 'n_counts', 'pct_counts_mt', 'n_genes_by_counts']]
                if not metric_options:
                    metric_options = obs_numerical[:5] if len(obs_numerical) > 0 else ['n_genes']
                color_col = st.selectbox("Select metric:", metric_options)
        
        with col3:
            point_size = st.slider("Point size:", 1, 10, 3)
        
        # Additional options
        with st.expander("⚙️ Advanced Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.8)
            with col_b:
                color_scale = st.selectbox(
                    "Color scale:",
                    ["Viridis", "Plasma", "Blues", "Reds", "RdBu", "Portland", "Turbo"]
                )
        
        # Create the plot
        if 'UMAP_0' not in df.columns or 'UMAP_1' not in df.columns:
            st.warning("⚠️ UMAP coordinates not found in data. Cannot display plot.")
        else:
            # Determine if categorical or continuous
            is_categorical = color_col in obs_categorical or color_type == "Cluster"
            
            # Prepare hover data
            hover_data_dict = {
                'UMAP_0': ':.2f',
                'UMAP_1': ':.2f',
            }
            # Add a few marker genes to hover
            for gene_col in expr_cols[:3]:
                hover_data_dict[gene_col] = ':.2f'
            
            if is_categorical:
                fig = px.scatter(
                    df,
                    x='UMAP_0',
                    y='UMAP_1',
                    color=color_col,
                    hover_data=hover_data_dict,
                    title=f"UMAP colored by {color_col}",
                    width=900,
                    height=650
                )
            else:
                fig = px.scatter(
                    df,
                    x='UMAP_0',
                    y='UMAP_1',
                    color=color_col,
                    color_continuous_scale=color_scale.lower(),
                    hover_data=hover_data_dict,
                    title=f"UMAP colored by {color_col}",
                    width=900,
                    height=650
                )
            
            # Update layout
            fig.update_traces(
                marker=dict(size=point_size, opacity=opacity, line=dict(width=0))
            )
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray', title='UMAP 1'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', title='UMAP 2'),
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats - moved below plot
        st.markdown("---")
        st.markdown("##### 📊 Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Get total cells from MLflow metrics if available
            try:
                client = MlflowClient()
                run_info = client.get_run(run_id)
                total_cells_actual = run_info.data.metrics.get('total_cells_before_subsample', None)
                if total_cells_actual:
                    st.metric("Total Cells (Full)", f"{int(total_cells_actual):,}")
                    st.caption(f"Viewing subsample of: {len(df):,}")
                else:
                    st.metric("Total Cells", f"{len(df):,}*")
                    st.caption("*Subsampled")
            except:
                st.metric("Total Cells", f"{len(df):,}*")
                st.caption("*Subsampled")
        
        with col2:
            cluster_col = get_cluster_column(df)
            if cluster_col:
                st.metric("Clusters", df[cluster_col].nunique())
        with col3:
            st.metric("Marker Genes", len(expr_cols))
        with col4:
            if 'UMAP_0' in df.columns:
                st.metric("Embeddings", "UMAP ✓")
        
        # Dot plot section
        st.markdown("---")
        with st.expander("Marker Gene Expression by Cluster", expanded=False):
            cluster_col = get_cluster_column(df)
            if cluster_col and expr_cols:
                # Load the cluster-to-marker mapping from MLflow
                try:
                    marker_mapping = download_cluster_markers_mapping(run_id)
                except:
                    marker_mapping = None
                
                # Gene selection - calculate top genes per cluster
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    n_top_genes = st.number_input(
                        "Top genes per cluster:",
                        min_value=1,
                        max_value=20,
                        value=3,
                        help="Select top N marker genes per cluster (ranked by Wilcoxon test)"
                    )
                    
                    # Get ordered list of genes using Wilcoxon rankings
                    ordered_genes_by_cluster = []
                    
                    if marker_mapping is not None:
                        # Use the pre-computed marker rankings from scanpy
                        for cluster_id in sorted(marker_mapping.columns, key=lambda x: int(x) if x.isdigit() else x):
                            # Get top N genes for this cluster (already ranked by Wilcoxon)
                            top_genes = marker_mapping[cluster_id].head(n_top_genes).dropna().tolist()
                            ordered_genes_by_cluster.extend(top_genes)
                    else:
                        # Fallback: use z-score ranking if CSV not available
                        mean_expr_by_cluster = df.groupby(cluster_col)[expr_cols].mean()
                        mean_expr_zscored = (mean_expr_by_cluster - mean_expr_by_cluster.mean()) / mean_expr_by_cluster.std()
                        
                        for cluster in sorted(mean_expr_zscored.index):
                            cluster_zscores = mean_expr_zscored.loc[cluster]
                            top_genes = cluster_zscores.nlargest(n_top_genes).index.tolist()
                            top_genes = [g.replace('expr_', '') for g in top_genes]
                            ordered_genes_by_cluster.extend(top_genes)
                    
                    # Remove duplicates while preserving order
                    ordered_genes = []
                    seen = set()
                    for gene in ordered_genes_by_cluster:
                        if gene not in seen:
                            ordered_genes.append(gene)
                            seen.add(gene)
                    
                    # Allow user to customize selection
                    selected_genes = st.multiselect(
                        "Customize gene selection:",
                        [c.replace('expr_', '') for c in expr_cols],
                        default=ordered_genes,
                        help="Pre-populated with top genes per cluster (ordered by cluster)"
                    )
                
                with col2:
                    st.write("")
                    st.write("")
                    scale_data = st.checkbox("Scale expression", value=True, help="Z-score normalization (enhances relative differences)")
                
                with col3:
                    st.write("")
                    st.write("")
                    font_size = st.slider("Font size:", 10, 20, 14)
                
                if selected_genes:
                    # Maintain the cluster-ordered gene list
                    expr_cols_to_plot = [f"expr_{g}" for g in selected_genes]
                    genes_ordered = selected_genes  # Keep for x-axis ordering
                else:
                    expr_cols_to_plot = expr_cols
                    genes_ordered = [c.replace('expr_', '') for c in expr_cols]
                
                # Calculate mean expression per cluster
                heatmap_data = df.groupby(cluster_col)[expr_cols_to_plot].mean()
                heatmap_data.columns = [c.replace('expr_', '') for c in heatmap_data.columns]
                
                # Reorder columns to match cluster-based ordering
                heatmap_data = heatmap_data[genes_ordered]
                
                # Optional scaling
                if scale_data:
                    heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
                    color_label = "Z-score"
                    color_scale = "RdBu_r"
                else:
                    color_label = "Mean Expression"
                    color_scale = "Viridis"
                
                # Prepare data for dot plot (convert from wide to long format)
                dotplot_data = []
                for cluster in heatmap_data.index:
                    for gene in heatmap_data.columns:
                        dotplot_data.append({
                            'Cluster': str(cluster),
                            'Gene': gene,
                            'Expression': heatmap_data.loc[cluster, gene]
                        })
                
                dotplot_df = pd.DataFrame(dotplot_data)
                
                # Create dot plot
                # For size, use absolute values or original expression (size must be non-negative)
                if scale_data:
                    # When z-scored, use absolute values for size, but keep z-scores for color
                    dotplot_df['Size'] = dotplot_df['Expression'].abs()
                else:
                    dotplot_df['Size'] = dotplot_df['Expression']
                
                fig_dotplot = px.scatter(
                    dotplot_df,
                    x='Gene',
                    y='Cluster',
                    color='Expression',  # Color by z-score (can be negative)
                    size='Size',  # Size by absolute value (always positive)
                    color_continuous_scale=color_scale,
                    labels={'Expression': color_label},
                    title=f"Marker Expression by Cluster ({color_label})",
                    height=max(400, len(heatmap_data.index) * 50)
                )
                
                # Update layout for better readability
                fig_dotplot.update_traces(
                    marker=dict(
                        sizemode='diameter',
                        sizeref=dotplot_df['Size'].max() / 15,  # Adjust dot size scaling
                        line=dict(width=0.5, color='white')
                    )
                )
                fig_dotplot.update_xaxes(tickangle=45, tickfont=dict(size=font_size))
                fig_dotplot.update_yaxes(tickfont=dict(size=font_size))
                fig_dotplot.update_layout(
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    font=dict(size=font_size),
                    title=dict(font=dict(size=font_size + 2))
                )
                
                st.plotly_chart(fig_dotplot, use_container_width=True)
            else:
                st.warning("Clustering information or marker genes not available")
        
        # QC and additional analysis section
        st.markdown("---")
        with st.expander("QC & Other Analysis Outputs", expanded=False):
            st.info(
                "**View Standard Analysis Plots**: All analysis outputs (QC plots, "
                "PCA, highly variable genes, marker genes heatmap, UMAP, etc.) are available in the MLflow run."
                "We may add QC interactive vizualization here in the future."
            )
            st.link_button("🔗 Open MLflow Run (View All Plots & Artifacts)", mlflow_url, type="primary")
            
            st.markdown("---")
            st.markdown("**About the MLflow Run:**")
            st.markdown(
                "- Quality control plots (cell/gene filtering, mitochondrial content)\n"
                "- PCA and variance explained plots\n"
                "- Highly variable genes identification\n"
                "- Full-resolution UMAP (all cells, all genes)\n"
                "- Marker genes heatmap\n"
                "- Complete AnnData object with all genes"
            )
        
        # Data table preview
        st.markdown("---")
        with st.expander("📊 View Raw Data Table"):
            # Show selected columns
            cluster_col = get_cluster_column(df)
            default_cols = [cluster_col, 'UMAP_0', 'UMAP_1'] if cluster_col else ['UMAP_0', 'UMAP_1']
            default_cols += expr_cols[:3]
            
            display_cols = st.multiselect(
                "Select columns to display:",
                df.columns.tolist(),
                default=[c for c in default_cols if c in df.columns]
            )
            if display_cols:
                st.dataframe(df[display_cols].head(100), use_container_width=True)
                st.caption(f"Showing first 100 of {len(df):,} cells")
        
        # Download section
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"singlecell_results_{st.session_state['singlecell_run_id'][:8]}.csv",
                mime="text/csv"
            )
        with col3:
            st.link_button("🔗 MLflow Run", mlflow_url, type="secondary")


def display_geo_ingestion_tab():
    """GEO Data Ingestion tab — integrated from CellAtria/CellExpress pipeline.
    Fetch metadata from NCBI GEO, preview samples, and download to UC Volumes."""
    
    st.markdown("###### GEO Data Ingestion")
    st.caption("Fetch scRNA-seq datasets from NCBI GEO and stage them to Unity Catalog Volumes for analysis. *Integrated from CellAtria.*")
    
    # ── Step 1: GEO Accession Lookup ──
    st.markdown("**Step 1: Fetch GEO Metadata**")
    
    col_input, col_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_input:
        gse_id = st.text_input(
            "GEO Accession",
            value=st.session_state.get("geo_gse_id", ""),
            placeholder="e.g., GSE204716",
            help="Enter a GEO Series accession ID. Metadata will be fetched from NCBI.",
            key="geo_accession_input"
        )
    with col_btn:
        fetch_clicked = st.button("Fetch Metadata", type="primary", use_container_width=True)
    
    if fetch_clicked and gse_id.strip():
        with st.spinner(f"Fetching metadata for {gse_id}..."):
            metadata = fetch_geo_metadata(gse_id.strip())
        st.session_state["geo_metadata"] = metadata
        st.session_state["geo_gse_id"] = gse_id.strip()
    
    # Display metadata if available
    metadata = st.session_state.get("geo_metadata")
    if metadata and "error" not in metadata:
        st.divider()
        st.markdown(f"**{metadata.get('title', 'N/A')}**")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Organism", metadata.get("organism", "N/A"))
        with col_b:
            st.metric("Samples", metadata.get("n_samples", 0))
        with col_c:
            st.markdown(f"[View on NCBI]({metadata.get('link', '#')})")
        
        with st.expander("Summary", expanded=False):
            st.write(metadata.get("summary", "No summary available."))
            if metadata.get("design"):
                st.markdown(f"**Design:** {metadata['design']}")
        
        # ── Step 2: Sample Selection ──
        samples = metadata.get("samples", [])
        if samples:
            st.divider()
            st.markdown("**Step 2: Select Samples to Download**")
            
            samples_df = pd.DataFrame(samples)
            samples_df.insert(0, "Download", True)
            
            edited_df = st.data_editor(
                samples_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Download": st.column_config.CheckboxColumn("Download", default=True),
                    "gsm_id": st.column_config.TextColumn("GSM ID", width="small"),
                    "description": st.column_config.TextColumn("Description", width="large"),
                },
                key="geo_sample_editor"
            )
            
            selected = edited_df[edited_df["Download"] == True]
            st.caption(f"{len(selected)} of {len(samples_df)} samples selected")
            
            # ── Step 3: Download Configuration ──
            st.divider()
            st.markdown("**Step 3: Download to UC Volume**")
            
            with st.form("geo_download_form", enter_to_submit=False):
                col_vol, col_proj = st.columns([2, 1])
                with col_vol:
                    volume_path = st.text_input(
                        "UC Volume Path",
                        value=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'catalog')}/{os.environ.get('CORE_SCHEMA_NAME', 'schema')}/singlecell_data",
                        help="Base path in UC Volumes where data will be downloaded."
                    )
                with col_proj:
                    project_name = st.text_input(
                        "Project Name (optional)",
                        value="",
                        placeholder="e.g., P105_HBI_scRNAseq",
                        help="Creates a subdirectory for organization."
                    )
                
                st.caption("Files will be downloaded to: `{volume_path}/geo_downloads/{project_or_gse}/{gsm_id}/`")
                download_btn = st.form_submit_button("Download Selected Samples", type="primary", use_container_width=True)
            
            if download_btn and len(selected) > 0:
                gsm_ids = selected["gsm_id"].tolist()
                
                progress = st.progress(0, text=f"Downloading {len(gsm_ids)} samples...")
                
                with st.spinner(f"Downloading {len(gsm_ids)} samples from GEO..."):
                    result = download_geo_to_volume(
                        gse_id=metadata["gse_id"],
                        gsm_ids=gsm_ids,
                        volume_path=volume_path,
                        project_name=project_name
                    )
                
                progress.progress(100, text="Download complete!")
                st.session_state["geo_download_result"] = result
            
            # Display download results
            dl_result = st.session_state.get("geo_download_result")
            if dl_result and "error" not in dl_result:
                st.divider()
                
                n_ok = dl_result.get("n_downloaded", 0)
                n_fail = dl_result.get("n_failed", 0)
                
                if n_ok > 0:
                    st.success(f"Downloaded {n_ok} samples to `{dl_result.get('output_dir', '')}`")
                if n_fail > 0:
                    st.warning(f"{n_fail} samples failed to download")
                
                # Show per-sample results
                for s in dl_result.get("samples", []):
                    if s["status"] == "success":
                        st.markdown(f"- **{s['gsm_id']}**: {len(s.get('files', []))} files ({s.get('size_mb', 0)} MB)")
                    else:
                        st.markdown(f"- **{s['gsm_id']}**: Failed — {s.get('error', 'unknown')}")
                
                # ── Step 4: Format Detection ──
                st.divider()
                st.markdown("**Step 4: Detect Data Format**")
                
                if st.button("Scan for Data Formats"):
                    with st.spinner("Scanning downloaded files..."):
                        fmt_result = detect_data_format(dl_result["output_dir"])
                    
                    if fmt_result.get("n_formats_found", 0) > 0:
                        for fmt in fmt_result["formats"]:
                            col_f, col_r = st.columns([1, 2])
                            with col_f:
                                st.markdown(f"**{fmt['format']}**")
                                if "size_mb" in fmt:
                                    st.caption(f"{fmt['size_mb']} MB")
                            with col_r:
                                st.code(fmt.get("path", ""), language=None)
                                st.caption(fmt.get("recommendation", ""))
                                
                                # For h5ad files, offer direct link to analysis
                                if fmt["format"] == "h5ad (AnnData)":
                                    st.info(f"Ready for analysis! Copy this path to the **Run New Analysis** tab:\n`{fmt['path']}`")
                    else:
                        st.info("No recognized single-cell formats found. Files may need conversion (10X → h5ad).")
            
            elif dl_result and "error" in dl_result:
                st.error(dl_result["error"])
        else:
            st.info("No samples found for this accession. Try a different GSE ID.")
    
    elif metadata and "error" in metadata:
        st.error(metadata["error"])





def display_sra_ingestion_tab():
    """SRA Data Ingestion tab — download raw sequencing data from NCBI SRA."""
    
    st.markdown("###### SRA Data Ingestion")
    st.caption("Download raw sequencing data (FASTQ) from NCBI SRA for alignment and variant calling.")
    
    # ── Step 1: SRA Accession Lookup ──
    st.markdown("**Step 1: Fetch SRA Metadata**")
    
    col_input, col_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_input:
        sra_accession = st.text_input(
            "SRA Accession",
            value=st.session_state.get("sra_accession_id", ""),
            placeholder="e.g., SRP123456, SRR1234567, PRJNA123456",
            help="Enter an SRA Study (SRP), Run (SRR), Experiment (SRX), or BioProject (PRJNA) accession.",
            key="sra_accession_input"
        )
    with col_btn:
        sra_fetch_clicked = st.button("Fetch Metadata", type="primary", use_container_width=True, key="sra_fetch_btn")
    
    if sra_fetch_clicked and sra_accession.strip():
        with st.spinner(f"Fetching SRA metadata for {sra_accession}..."):
            sra_metadata = fetch_sra_metadata(sra_accession.strip())
        st.session_state["sra_metadata"] = sra_metadata
        st.session_state["sra_accession_id"] = sra_accession.strip()
    
    # Display metadata if available
    sra_metadata = st.session_state.get("sra_metadata")
    if sra_metadata and "error" not in sra_metadata:
        st.divider()
        st.markdown(f"**{sra_metadata.get('title', 'N/A')}**")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Organism", sra_metadata.get("organism", "N/A"))
        with col_b:
            st.metric("Platform", sra_metadata.get("platform", "N/A"))
        with col_c:
            st.metric("Library", sra_metadata.get("library_strategy", "N/A"))
        with col_d:
            st.metric("Runs", sra_metadata.get("n_runs", 0))
        
        # ── Step 2: Run Selection ──
        runs = sra_metadata.get("runs", [])
        if runs:
            st.divider()
            st.markdown("**Step 2: Select Runs to Download**")
            
            runs_df = pd.DataFrame(runs)
            runs_df.insert(0, "Download", True)
            
            # Format columns for display
            if "size_mb" in runs_df.columns:
                runs_df["size_mb"] = runs_df["size_mb"].apply(lambda x: f"{x:,.1f}")
            
            edited_runs = st.data_editor(
                runs_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Download": st.column_config.CheckboxColumn("Download", default=True),
                    "srr_id": st.column_config.TextColumn("SRR ID", width="small"),
                    "spots": st.column_config.NumberColumn("Spots", format="%d"),
                    "bases": st.column_config.NumberColumn("Bases", format="%d"),
                    "size_mb": st.column_config.TextColumn("Size (MB)", width="small"),
                },
                key="sra_run_editor"
            )
            
            selected_runs = edited_runs[edited_runs["Download"] == True]
            st.caption(f"{len(selected_runs)} of {len(runs_df)} runs selected")
            
            # ── Step 3: Download Configuration ──
            st.divider()
            st.markdown("**Step 3: Download to UC Volume**")
            
            with st.form("sra_download_form", enter_to_submit=False):
                col_vol, col_proj, col_method = st.columns([2, 1, 1])
                with col_vol:
                    sra_volume_path = st.text_input(
                        "UC Volume Path",
                        value=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'catalog')}/{os.environ.get('CORE_SCHEMA_NAME', 'schema')}/singlecell_data",
                        help="Base path in UC Volumes where data will be downloaded.",
                        key="sra_volume_path"
                    )
                with col_proj:
                    sra_project_name = st.text_input(
                        "Project Name (optional)",
                        value="",
                        placeholder="e.g., WGS_study_2024",
                        help="Creates a subdirectory for organization.",
                        key="sra_project_name"
                    )
                with col_method:
                    download_method = st.selectbox(
                        "Download Method",
                        options=["https", "aws_s3"],
                        index=0,
                        help="AWS S3 is faster on AWS-hosted workspaces. HTTPS works everywhere.",
                        format_func=lambda x: {"https": "HTTPS (universal)", "aws_s3": "AWS S3 (fastest)"}[x],
                        key="sra_download_method"
                    )
                
                st.caption("Files will be downloaded to: `{volume_path}/sra_downloads/{project}/{srr_id}/`")
                st.caption("If `sra-tools` (fasterq-dump) is installed on the cluster, .sra files are auto-converted to FASTQ.")
                sra_download_btn = st.form_submit_button("Download Selected Runs", type="primary", use_container_width=True)
            
            if sra_download_btn and len(selected_runs) > 0:
                srr_ids = selected_runs["srr_id"].tolist()
                
                with st.spinner(f"Downloading {len(srr_ids)} SRA runs (this may take several minutes for large files)..."):
                    sra_result = download_sra_to_volume(
                        srr_ids=srr_ids,
                        volume_path=sra_volume_path,
                        project_name=sra_project_name,
                        method=download_method
                    )
                
                st.session_state["sra_download_result"] = sra_result
            
            # Display download results
            sra_dl_result = st.session_state.get("sra_download_result")
            if sra_dl_result and "error" not in sra_dl_result:
                st.divider()
                
                n_ok = sra_dl_result.get("n_downloaded", 0)
                n_fail = sra_dl_result.get("n_failed", 0)
                
                if n_ok > 0:
                    st.success(f"Downloaded {n_ok} runs to `{sra_dl_result.get('output_dir', '')}`")
                if n_fail > 0:
                    st.warning(f"{n_fail} runs failed to download")
                
                for r in sra_dl_result.get("runs", []):
                    if r["status"] == "success":
                        fastq_info = f" → FASTQ: {', '.join(r.get('fastq_files', []))}" if r.get("fastq_files") else " (raw .sra, install sra-tools to convert)"
                        st.markdown(f"- **{r['srr_id']}**: {r.get('size_mb', 0)} MB{fastq_info}")
                    else:
                        st.markdown(f"- **{r['srr_id']}**: Failed — {r.get('error', 'unknown')}")
                
                # ── Step 4: Format Detection ──
                st.divider()
                st.markdown("**Step 4: Detect Data Format**")
                
                if st.button("Scan for Data Formats", key="sra_scan_formats"):
                    with st.spinner("Scanning downloaded files..."):
                        sra_fmt_result = detect_sra_data(sra_dl_result["output_dir"])
                    
                    if sra_fmt_result.get("n_formats_found", 0) > 0:
                        for fmt in sra_fmt_result["formats"]:
                            col_f, col_r = st.columns([1, 2])
                            with col_f:
                                st.markdown(f"**{fmt['format']}**")
                                if "size_mb" in fmt:
                                    st.caption(f"{fmt['size_mb']} MB")
                            with col_r:
                                st.code(fmt.get("path", ""), language=None)
                                st.caption(fmt.get("recommendation", ""))
                    else:
                        st.info("No FASTQ/BAM files found. Ensure sra-tools is installed for automatic .sra → FASTQ conversion.")
            
            elif sra_dl_result and "error" in sra_dl_result:
                st.error(sra_dl_result["error"])
        else:
            st.info("No runs found for this accession.")
    
    elif sra_metadata and "error" in sra_metadata:
        st.error(sra_metadata["error"])



def display_nextflow_alignment_tab():
    """Nextflow Alignment tab — run nf-core/scrnaseq to convert FASTQ → h5ad."""
    
    st.markdown("###### Nextflow Alignment (FASTQ → h5ad)")
    st.caption("Run nf-core/scrnaseq (sandwich wrapper) to align FASTQ files and produce count matrices for scanpy. Automated QC gates and execution trace logging. Requires cluster init script: `install_nextflow.sh`.")
    
    # ── Step 1: FASTQ Source ──
    st.markdown("**Step 1: Select FASTQ Source**")
    
    col_dir, col_scan = st.columns([3, 1], vertical_alignment="bottom")
    with col_dir:
        nf_fastq_dir = st.text_input(
            "FASTQ Directory",
            value=st.session_state.get("nf_fastq_dir", ""),
            placeholder=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'catalog')}/{os.environ.get('CORE_SCHEMA_NAME', 'schema')}/singlecell_data/sra_downloads/...",
            help="Path to directory containing FASTQ files (from SRA download or manual upload).",
            key="nf_fastq_dir_input"
        )
    with col_scan:
        nf_scan = st.button("Scan Directory", type="primary", use_container_width=True, key="nf_scan_btn")
    
    if nf_scan and nf_fastq_dir.strip():
        import glob
        fastq_files = []
        for pattern in ["**/*.fastq.gz", "**/*.fastq", "**/*.fq.gz", "**/*.fq"]:
            fastq_files.extend(glob.glob(os.path.join(nf_fastq_dir.strip(), pattern), recursive=True))
        st.session_state["nf_fastq_dir"] = nf_fastq_dir.strip()
        st.session_state["nf_fastq_files"] = fastq_files
    
    fastq_files = st.session_state.get("nf_fastq_files", [])
    if fastq_files:
        st.success(f"Found {len(fastq_files)} FASTQ files")
        with st.expander("View files", expanded=False):
            for f in fastq_files[:20]:
                st.text(os.path.basename(f))
            if len(fastq_files) > 20:
                st.caption(f"... and {len(fastq_files) - 20} more")
        
        # ── Step 2: Pipeline Configuration ──
        st.divider()
        st.markdown("**Step 2: Configure Alignment Pipeline**")
        
        with st.form("nf_alignment_form", enter_to_submit=False):
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                nf_aligner = st.selectbox(
                    "Aligner",
                    options=list(ALIGNERS.keys()),
                    index=0,
                    format_func=lambda x: f"{ALIGNERS[x]['name']} — {ALIGNERS[x]['speed']}",
                    help="STARsolo recommended for most 10x datasets. Salmon/kallisto are faster alternatives.",
                    key="nf_aligner"
                )
                nf_genome = st.selectbox(
                    "Reference Genome",
                    options=list(GENOME_PRESETS.keys()),
                    index=0,
                    format_func=lambda x: GENOME_PRESETS[x],
                    key="nf_genome"
                )
                nf_chemistry = st.selectbox(
                    "10x Chemistry",
                    options=list(CHEMISTRY_PRESETS.keys()),
                    index=0,
                    format_func=lambda x: CHEMISTRY_PRESETS[x],
                    help="Auto-detect works for most datasets. Override if you know the chemistry version.",
                    key="nf_chemistry"
                )
            
            with col_b:
                nf_output_dir = st.text_input(
                    "Output Directory",
                    value=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'catalog')}/{os.environ.get('CORE_SCHEMA_NAME', 'schema')}/singlecell_data/nextflow_output",
                    help="Where aligned count matrices will be written.",
                    key="nf_output_dir"
                )
                nf_sample_name = st.text_input(
                    "Sample Name (optional)",
                    value="",
                    placeholder="e.g., PBMC_10k",
                    help="Override sample name. Leave empty to auto-detect from filenames.",
                    key="nf_sample_name"
                )
                nf_extra_args = st.text_input(
                    "Extra Nextflow Args (advanced)",
                    value="",
                    placeholder="e.g., --max_cpus 8 --max_memory 28.GB",
                    help="Additional arguments passed to nextflow run.",
                    key="nf_extra_args"
                )
                
                st.markdown("**Compute Size:**")
                nf_compute_tier = st.selectbox(
                    "Cluster Size",
                    options=list(COMPUTE_TIERS.keys()),
                    index=1,  # default to "medium"
                    format_func=lambda x: COMPUTE_TIERS[x]["label"],
                    help="Scales the on-demand cluster for this run. Larger tiers have more vCPUs and RAM for bigger studies. Cluster auto-terminates when the pipeline finishes.",
                    key="nf_compute_tier"
                )
                tier_info = COMPUTE_TIERS.get(nf_compute_tier, {})
                st.caption(f"💡 {tier_info.get('description', '')} — {tier_info.get('vcpus', '?')} vCPUs, {tier_info.get('ram_gb', '?')} GB RAM")
            
            st.divider()
            st.markdown("**Alignment QC Gates (Sandwich Wrapper):**")
            col_qc1, col_qc2, col_qc3 = st.columns([1, 1, 1])
            with col_qc1:
                nf_qc_enabled = st.toggle(
                    "Enable QC Gates",
                    value=True,
                    help="Run automated quality checks on alignment output (cell count, median genes/cell, mitochondrial %).",
                    key="nf_qc_toggle"
                )
            with col_qc2:
                nf_min_cells = st.number_input("Min Cells", value=200, step=50, help="Minimum cells to pass QC.", key="nf_min_cells")
                nf_min_median_genes = st.number_input("Min Median Genes/Cell", value=500, step=100, help="Minimum median genes per cell.", key="nf_min_median_genes")
            with col_qc3:
                nf_trigger_scanpy = st.toggle(
                    "Auto-trigger Scanpy",
                    value=False,
                    help="Automatically launch scanpy analysis when alignment and QC pass.",
                    key="nf_trigger_scanpy"
                )
                nf_project_name = st.text_input(
                    "Project Name",
                    value="",
                    placeholder="e.g., PBMC_study_2026",
                    help="Used in Delta audit trail and run naming.",
                    key="nf_project_name"
                )

            st.divider()
            col_mode, col_info = st.columns([1, 2])
            with col_mode:
                nf_parallel = st.toggle(
                    "Parallel Mode",
                    value=False,
                    help="Fan out each sample to its own cluster for parallel alignment. Recommended for multi-sample studies (3+ samples).",
                    key="nf_parallel_toggle"
                )
            with col_info:
                if nf_parallel:
                    st.info("Each sample will run on its own cluster in parallel, then outputs are merged. Much faster for multi-sample studies.")
                else:
                    st.caption("Single-job mode: all samples processed sequentially on one cluster.")
            
            st.caption(f"Pipeline: nf-core/scrnaseq v2.7.1 (Sandwich Wrapper) | Aligner: {ALIGNERS.get(nf_aligner, {}).get('description', '')} | QC Gates: {'ON' if nf_qc_enabled else 'OFF'}")
            nf_submit = st.form_submit_button("Start Alignment Pipeline", type="primary", use_container_width=True)
        
        if nf_submit:
            # Validate inputs
            if not nf_output_dir.strip():
                st.error("Please specify an output directory")
            else:
                # Submit as a Databricks job
                user_info = get_user_info()
                
                try:
                    from databricks.sdk import WorkspaceClient
                    w = WorkspaceClient()
                    
                    # Inject Nextflow resource limits matching the cluster size
                    tier_cfg = COMPUTE_TIERS.get(nf_compute_tier, COMPUTE_TIERS["medium"])
                    nf_resource_args = f"--max_cpus {tier_cfg['vcpus']} --max_memory {tier_cfg['ram_gb']}.GB"
                    combined_extra = f"{nf_extra_args} {nf_resource_args}".strip() if nf_extra_args else nf_resource_args
                    
                    from databricks.sdk.service.jobs import SubmitTask, NotebookTask, TaskDependency
                    
                    notebook_path = "/Users/andrew_forman@eisai.com/genesis-workbench/modules/single_cell/scanpy/scanpy_v0.0.1/notebooks/nextflow_sandwich_wrapper"
                    merge_notebook = "/Users/andrew_forman@eisai.com/genesis-workbench/modules/single_cell/scanpy/scanpy_v0.0.1/notebooks/merge_nextflow_outputs"
                    
                    if nf_parallel:
                        # ── Parallel fan-out: one task per sample ──
                        with st.spinner("Building samplesheet and splitting by sample..."):
                            # First build the samplesheet to know how many samples
                            samplesheet_path = os.path.join(nf_output_dir.strip(), "samplesheet.csv")
                            sheet_result = build_samplesheet(nf_fastq_dir.strip(), samplesheet_path, nf_sample_name or None)
                            
                            if "error" in sheet_result:
                                st.error(f"Samplesheet error: {sheet_result['error']}")
                                raise ValueError(sheet_result["error"])
                        
                        with st.spinner("Building parallel tasks..."):
                            parallel_result = build_parallel_tasks(
                                samplesheet_path=samplesheet_path,
                                output_dir=nf_output_dir.strip(),
                                aligner=nf_aligner,
                                genome=nf_genome,
                                chemistry=nf_chemistry,
                                pipeline_version="2.7.1",
                                extra_args=combined_extra,
                                compute_tier=nf_compute_tier,
                                merge_notebook_path=merge_notebook,
                            )
                        
                        if parallel_result.get("single_sample"):
                            st.warning("Only 1 sample found — falling back to single-job mode.")
                            nf_parallel = False  # Fall through to single-job below
                        elif "error" in parallel_result:
                            st.error(f"Fan-out error: {parallel_result['error']}")
                            raise ValueError(parallel_result["error"])
                        else:
                            # Convert task dicts to SDK SubmitTask objects
                            submit_tasks = []
                            for t in parallel_result["tasks"]:
                                deps = [TaskDependency(task_key=d["task_key"]) for d in t.get("depends_on", [])]
                                submit_tasks.append(
                                    SubmitTask(
                                        task_key=t["task_key"],
                                        notebook_task=NotebookTask(
                                            notebook_path=t["notebook_task"]["notebook_path"],
                                            base_parameters=t["notebook_task"]["base_parameters"],
                                            source="WORKSPACE",
                                        ),
                                        new_cluster=t["new_cluster"],
                                        depends_on=deps if deps else None,
                                    )
                                )
                            
                            n_samples = parallel_result["sample_count"]
                            with st.spinner(f"Submitting {n_samples} parallel tasks + merge..."):
                                run = w.jobs.submit(
                                    run_name=f"nf-core-scrnaseq_parallel_{n_samples}samples_{nf_compute_tier}",
                                    tasks=submit_tasks,
                                ).result()
                            
                            host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                            if not host_name.startswith("https://"):
                                host_name = "https://" + host_name
                            run_url = f"{host_name}/jobs/runs/{run.run_id}"
                            
                            st.success(f"✅ Parallel job submitted! {n_samples} samples × {tier_cfg['label']} clusters + merge task")
                            st.markdown(f"**Samples:** {', '.join(parallel_result['sample_names'])}")
                            st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                            st.info(f"Each sample runs on its own cluster in parallel. The merge task collects all h5ad files when done.\nOutput: `{nf_output_dir.strip()}`")
                    
                    if not nf_parallel:
                        # ── Single-job mode (original behavior) ──
                        cluster_spec = build_new_cluster_spec(tier=nf_compute_tier)
                        
                        with st.spinner(f"Submitting Nextflow alignment job ({tier_cfg['label']})..."):
                            run = w.jobs.submit(
                                run_name=f"nf-core-scrnaseq_{nf_sample_name or 'alignment'}_{nf_compute_tier}",
                                tasks=[
                                    SubmitTask(
                                        task_key="nextflow_alignment",
                                        notebook_task=NotebookTask(
                                            notebook_path=notebook_path,
                                            base_parameters={
                                                "fastq_dir": nf_fastq_dir.strip(),
                                                "output_dir": nf_output_dir.strip(),
                                                "aligner": nf_aligner,
                                                "genome": nf_genome,
                                                "chemistry": nf_chemistry,
                                                "sample_name": nf_sample_name or "",
                                                "pipeline_version": "2.7.1",
                                                "extra_args": combined_extra,
                                                "qc_gate_enabled": str(nf_qc_enabled).lower(),
                                                "min_cells": str(nf_min_cells),
                                                "min_median_genes": str(nf_min_median_genes),
                                                "trigger_scanpy": str(nf_trigger_scanpy).lower(),
                                                "project_name": nf_project_name or "unnamed",
                                            },
                                            source="WORKSPACE",
                                        ),
                                        new_cluster=cluster_spec,
                                    )
                                ],
                            ).result()
                            
                            host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                            if not host_name.startswith("https://"):
                                host_name = "https://" + host_name
                            run_url = f"{host_name}/jobs/runs/{run.run_id}"
                            
                            st.success(f"✅ Nextflow alignment job submitted! Run ID: {run.run_id}")
                            st.link_button("🔗 View Run in Databricks", run_url, type="primary")
                            st.info(f"When complete, the h5ad file will be in: `{nf_output_dir.strip()}`\nCopy the path to the **Run Analysis** tab to continue with scanpy.")
                
                except ValueError:
                    pass  # Already displayed error via st.error
                except Exception as e:
                    st.error(f"❌ Failed to submit job: {str(e)}")
                    st.info("Cluster will be provisioned automatically with Nextflow init scripts. Check your permissions if this persists.")



def display_bulk_rnaseq_alignment_tab():
    """Bulk RNA-seq Alignment tab — run nf-core/rnaseq (sandwich wrapper) to produce gene counts."""
    
    st.markdown("###### Bulk RNA-seq Alignment (FASTQ → Counts)")
    st.caption("Run nf-core/rnaseq v3.16.1 (sandwich wrapper) to align FASTQ files and produce DESeq2-ready gene counts. Automated QC gates and Delta audit trail.")
    
    # ── Step 1: FASTQ Source ──
    st.markdown("**Step 1: Select FASTQ Source**")
    
    col_dir, col_scan = st.columns([3, 1], vertical_alignment="bottom")
    with col_dir:
        bulk_fastq_dir = st.text_input(
            "FASTQ Directory",
            value=st.session_state.get("bulk_fastq_dir", ""),
            placeholder=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'catalog')}/{os.environ.get('CORE_SCHEMA_NAME', 'schema')}/bulk_rnaseq_data/...",
            help="Path to directory containing paired-end FASTQ files.",
            key="bulk_fastq_dir_input"
        )
    with col_scan:
        bulk_scan = st.button("Scan Directory", type="primary", use_container_width=True, key="bulk_scan_btn")
    
    if bulk_scan and bulk_fastq_dir.strip():
        import glob
        fastq_files = []
        for pattern in ["**/*.fastq.gz", "**/*.fastq", "**/*.fq.gz", "**/*.fq"]:
            fastq_files.extend(glob.glob(os.path.join(bulk_fastq_dir.strip(), pattern), recursive=True))
        st.session_state["bulk_fastq_dir"] = bulk_fastq_dir.strip()
        st.session_state["bulk_fastq_files"] = fastq_files
    
    fastq_files = st.session_state.get("bulk_fastq_files", [])
    if fastq_files:
        st.success(f"Found {len(fastq_files)} FASTQ files")
        with st.expander("View files", expanded=False):
            for f in fastq_files[:20]:
                st.text(os.path.basename(f))
            if len(fastq_files) > 20:
                st.caption(f"... and {len(fastq_files) - 20} more")
        
        # ── Step 2: Pipeline Configuration ──
        st.divider()
        st.markdown("**Step 2: Configure Bulk RNA-seq Pipeline**")
        
        with st.form("bulk_rnaseq_form", enter_to_submit=False):
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                bulk_aligner = st.selectbox(
                    "Aligner",
                    options=["star_salmon", "star_rsem", "hisat2"],
                    index=0,
                    format_func=lambda x: {
                        "star_salmon": "STAR + Salmon (recommended)",
                        "star_rsem": "STAR + RSEM",
                        "hisat2": "HISAT2 + featureCounts"
                    }[x],
                    help="star_salmon is the gold standard for most bulk RNA-seq. RSEM for isoform-level quantification. HISAT2 for splice-aware lightweight alignment.",
                    key="bulk_aligner"
                )
                bulk_genome = st.selectbox(
                    "Reference Genome",
                    options=["GRCh38", "GRCh37", "GRCm39", "GRCm38"],
                    index=0,
                    format_func=lambda x: {
                        "GRCh38": "Human GRCh38 (hg38)",
                        "GRCh37": "Human GRCh37 (hg19)",
                        "GRCm39": "Mouse GRCm39 (mm39)",
                        "GRCm38": "Mouse GRCm38 (mm10)"
                    }[x],
                    key="bulk_genome"
                )
                bulk_strandedness = st.selectbox(
                    "Strandedness",
                    options=["auto", "forward", "reverse", "unstranded"],
                    index=0,
                    help="'auto' uses Salmon infer_experiment. Most Illumina kits are 'reverse'. Unstranded for older libraries.",
                    key="bulk_strandedness"
                )
            
            with col_b:
                bulk_output_dir = st.text_input(
                    "Output Directory",
                    value=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'catalog')}/{os.environ.get('CORE_SCHEMA_NAME', 'schema')}/bulk_rnaseq_data/nextflow_output",
                    help="Where aligned counts and QC reports will be written.",
                    key="bulk_output_dir"
                )
                bulk_project_name = st.text_input(
                    "Project Name",
                    value="",
                    placeholder="e.g., TCGA_reanalysis_2026",
                    help="Used in Delta audit trail and run naming.",
                    key="bulk_project_name"
                )
                bulk_extra_args = st.text_input(
                    "Extra Nextflow Args (advanced)",
                    value="",
                    placeholder="e.g., --skip_trimming --skip_dupradar",
                    help="Additional arguments passed to nextflow run.",
                    key="bulk_extra_args"
                )
                
                st.markdown("**Compute Size:**")
                bulk_compute_tier = st.selectbox(
                    "Cluster Size",
                    options=list(COMPUTE_TIERS.keys()),
                    index=1,  # default to "medium"
                    format_func=lambda x: COMPUTE_TIERS[x]["label"],
                    help="Bulk RNA-seq is memory-intensive for STAR indexing. Use medium+ for >6 samples.",
                    key="bulk_compute_tier"
                )
                tier_info = COMPUTE_TIERS.get(bulk_compute_tier, {})
                st.caption(f"\U0001f4a1 {tier_info.get('description', '')} — {tier_info.get('vcpus', '?')} vCPUs, {tier_info.get('ram_gb', '?')} GB RAM")
            
            st.divider()
            st.markdown("**QC Gates (Sandwich Wrapper):**")
            col_qc1, col_qc2, col_qc3 = st.columns([1, 1, 1])
            with col_qc1:
                bulk_qc_enabled = st.toggle(
                    "Enable QC Gates",
                    value=True,
                    help="Automated quality checks on MultiQC stats: mapping rate, detected genes, rRNA contamination.",
                    key="bulk_qc_toggle"
                )
            with col_qc2:
                bulk_min_mapping = st.number_input("Min Mapping Rate (%)", value=70.0, step=5.0, min_value=0.0, max_value=100.0, help="Flag samples below this uniquely-mapped %.", key="bulk_min_mapping")
                bulk_min_genes = st.number_input("Min Detected Genes", value=10000, step=1000, help="Flag samples with fewer detected genes.", key="bulk_min_genes_qc")
            with col_qc3:
                bulk_max_rrna = st.number_input("Max rRNA %", value=15.0, step=5.0, min_value=0.0, max_value=100.0, help="Flag samples with high rRNA contamination.", key="bulk_max_rrna")
                bulk_trigger_de = st.toggle(
                    "Auto-trigger DE Analysis",
                    value=False,
                    help="Automatically launch differential expression notebook when alignment completes and QC passes.",
                    key="bulk_trigger_de"
                )
            
            st.divider()
            st.caption(f"Pipeline: nf-core/rnaseq v3.16.1 (Sandwich Wrapper) | Aligner: {bulk_aligner} | QC Gates: {'ON' if bulk_qc_enabled else 'OFF'}")
            bulk_submit = st.form_submit_button("Start Bulk RNA-seq Pipeline", type="primary", use_container_width=True)
        
        if bulk_submit:
            if not bulk_output_dir.strip():
                st.error("Please specify an output directory")
            else:
                user_info = get_user_info()
                try:
                    from databricks.sdk import WorkspaceClient
                    from databricks.sdk.service.jobs import SubmitTask, NotebookTask
                    
                    w = WorkspaceClient()
                    tier_cfg = COMPUTE_TIERS.get(bulk_compute_tier, COMPUTE_TIERS["medium"])
                    nf_resource_args = f"--max_cpus {tier_cfg['vcpus']} --max_memory {tier_cfg['ram_gb']}.GB"
                    combined_extra = f"{bulk_extra_args} {nf_resource_args}".strip() if bulk_extra_args else nf_resource_args
                    
                    cluster_spec = build_new_cluster_spec(tier=bulk_compute_tier)
                    notebook_path = "/Users/andrew_forman@eisai.com/genesis-workbench/modules/bulk_rnaseq/notebooks/nextflow_sandwich_wrapper_rnaseq"
                    
                    with st.spinner(f"Submitting Bulk RNA-seq job ({tier_cfg['label']})..."):
                        run = w.jobs.submit(
                            run_name=f"nf-core-rnaseq_{bulk_project_name or 'alignment'}_{bulk_compute_tier}",
                            tasks=[
                                SubmitTask(
                                    task_key="bulk_rnaseq_alignment",
                                    notebook_task=NotebookTask(
                                        notebook_path=notebook_path,
                                        base_parameters={
                                            "fastq_dir": bulk_fastq_dir.strip(),
                                            "output_dir": bulk_output_dir.strip(),
                                            "aligner": bulk_aligner,
                                            "genome": bulk_genome,
                                            "strandedness": bulk_strandedness,
                                            "pipeline_version": "3.16.1",
                                            "extra_args": combined_extra,
                                            "qc_gate_enabled": str(bulk_qc_enabled).lower(),
                                            "min_mapping_rate": str(bulk_min_mapping),
                                            "min_detected_genes": str(bulk_min_genes),
                                            "max_rrna_pct": str(bulk_max_rrna),
                                            "trigger_de_analysis": str(bulk_trigger_de).lower(),
                                            "project_name": bulk_project_name or "unnamed",
                                        },
                                        source="WORKSPACE",
                                    ),
                                    new_cluster=cluster_spec,
                                )
                            ],
                        ).result()
                        
                        host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                        if not host_name.startswith("https://"):
                            host_name = "https://" + host_name
                        run_url = f"{host_name}/jobs/runs/{run.run_id}"
                        
                        st.success(f"\u2705 Bulk RNA-seq job submitted! Run ID: {run.run_id}")
                        st.link_button("\U0001f517 View Run in Databricks", run_url, type="primary")
                        st.info(f"When complete, gene counts will be in: `{bulk_output_dir.strip()}`\nThe sandwich wrapper will log execution traces to Delta and run QC gates automatically.")
                
                except Exception as e:
                    st.error(f"\u274c Failed to submit job: {str(e)}")
                    st.info("Cluster will be provisioned automatically with Nextflow init scripts. Check your permissions if this persists.")


def display_fetchngs_tab():
    """Fetch Public Data tab — nf-core/fetchngs integration for downloading raw FASTQs from any public repository."""
    
    st.markdown("###### Fetch Public Data (nf-core/fetchngs)")
    st.caption(
        "Download raw FASTQ data from SRA, ENA, DDBJ, GEO, or Synapse using nf-core/fetchngs. "
        "Outputs a ready-made samplesheet for direct input to scRNA-seq or Bulk RNA-seq alignment pipelines."
    )
    

    # ── Quick-Start Guide (expandable) ──
    with st.expander("📖 How It Works — End-to-End Pipeline Guide", expanded=False):
        st.markdown("""
**What fetchngs does**: Given accession IDs from any public repository, it downloads raw FASTQ files
and generates a samplesheet compatible with nf-core alignment pipelines — fully automated.

---

**End-to-End Flow:**
```
Accession IDs (SRR, GSE, PRJNA...)
  │
  ▼ [This form]
  Validate → Preview metadata → Configure → Launch
  │
  ▼ [Sandwich Wrapper - automated]
  Pre-flight: gen config → Black box: nf-core/fetchngs → Post-flight: QC gates
  │
  ▼ [Auto-trigger - if enabled]
  scrnaseq/rnaseq alignment → scanpy/DE analysis → MLflow
```

**Total time**: ~3–5 hours from accession to annotated cells (fully unattended).

---

**Supported Accession Types:**

| Type | Pattern | Example | Scope |
| --- | --- | --- | --- |
| Run | SRR/ERR/DRR | SRR9123032 | Single sample |
| Study | SRP/ERP/DRP | SRP188369 | All runs in study |
| BioProject | PRJNA/PRJEB/PRJDB | PRJNA544731 | All runs in project |
| GEO Series | GSE | GSE204716 | Resolved to SRP |
| Synapse | syn | syn1234567 | Requires SYNAPSE_AUTH_TOKEN |

**Mixing types is supported** — paste any combination and fetchngs routes each to the right repository.

---

**Download Methods:**

| Method | Speed | When to Use |
| --- | --- | --- |
| **sratools** (default) | Fast, parallel | Most use cases — fasterq-dump with retries |
| **aspera** | Fastest (50+ MB/s) | Large studies (>100 samples, >500 GB) |
| **ftp** | Slow but simple | Debugging or restricted networks |

---

**What happens after you click Launch:**

1. **Cluster starts** (~5 min) — r5.4xlarge, 16 vCPU, 128 GB RAM, Nextflow pre-installed
2. **Pre-flight** — validates IDs, generates institutional config, registers audit trail
3. **fetchngs runs** — downloads FASTQs in parallel, verifies MD5 checksums
4. **Post-flight** — catalogs FASTQs to Delta, runs QC gates (size, count, existence)
5. **Auto-trigger** (if enabled) — submits scrnaseq or rnaseq wrapper with the samplesheet

---

**Monitoring**: After launch, click "View Run in Databricks" or query:
```sql
SELECT run_id, status, n_samples, elapsed_minutes
FROM dhbl_discovery_us_dev.genesis_schema.nextflow_run_audit
WHERE pipeline = 'fetchngs' ORDER BY started_at DESC LIMIT 5
```

**GEO Ingestion vs fetchngs**: Use GEO Ingestion (next tab) when you want *processed* data
(h5ad, MTX) from a paper. Use fetchngs when you want *raw reads* for custom re-alignment.
""")

    # ── Step 1: Accession Input and Validation ──
    st.markdown("**Step 1: Enter Accession IDs**")
    st.caption("Supports: SRR/ERR/DRR (runs), SRP/ERP/DRP (studies), PRJNA/PRJEB/PRJDB (projects), GSE (GEO), syn (Synapse)")
    
    accession_input = st.text_area(
        "Accession IDs (one per line, or comma-separated)",
        value=st.session_state.get("fetchngs_accessions", ""),
        height=120,
        placeholder="SRP123456\nSRR1234567\nGSE204716\nPRJNA123456",
        help="Enter one or more accession IDs. Mixing types is supported — fetchngs handles routing automatically.",
        key="fetchngs_accession_input"
    )
    
    col_validate, col_lookup = st.columns([1, 1])
    
    with col_validate:
        validate_clicked = st.button("Validate & Classify", type="secondary", use_container_width=True, key="fetchngs_validate_btn")
    with col_lookup:
        lookup_clicked = st.button("Fetch ENA Metadata", type="primary", use_container_width=True, key="fetchngs_lookup_btn")
    
    # Parse accessions
    if accession_input.strip():
        # Split on newlines, commas, or spaces
        raw_ids = re.split(r"[\n,\s]+", accession_input.strip())
        parsed_ids = [x.strip() for x in raw_ids if x.strip()]
    else:
        parsed_ids = []
    
    # Validate
    if validate_clicked and parsed_ids:
        validation = validate_accessions(parsed_ids)
        st.session_state["fetchngs_validation"] = validation
        st.session_state["fetchngs_accessions"] = accession_input.strip()
    
    if lookup_clicked and parsed_ids:
        # Use first study/project accession for metadata lookup
        validation = validate_accessions(parsed_ids)
        st.session_state["fetchngs_validation"] = validation
        st.session_state["fetchngs_accessions"] = accession_input.strip()
        
        # Find best accession to look up (prefer study/project over individual runs)
        lookup_acc = None
        for v in validation["valid"]:
            if v["type"] in ("study", "bioproject", "geo"):
                lookup_acc = v["accession"]
                break
        if not lookup_acc and validation["valid"]:
            lookup_acc = validation["valid"][0]["accession"]
        
        if lookup_acc:
            with st.spinner(f"Querying ENA API for {lookup_acc}..."):
                ena_meta = fetch_ena_metadata(lookup_acc)
            st.session_state["fetchngs_ena_metadata"] = ena_meta
    
    # Display validation results
    validation = st.session_state.get("fetchngs_validation")
    if validation:
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.metric("Valid IDs", validation["n_valid"])
        with col_v2:
            st.metric("Invalid IDs", validation["n_invalid"])
        with col_v3:
            types_str = ", ".join(f"{k}: {v}" for k, v in validation.get("type_counts", {}).items())
            st.metric("Types", types_str or "—")
        
        if validation["invalid"]:
            for inv in validation["invalid"]:
                st.warning(f"Invalid: `{inv['accession']}` — {inv['reason']}")
    
    # Display ENA metadata
    ena_meta = st.session_state.get("fetchngs_ena_metadata")
    if ena_meta and "error" not in ena_meta:
        st.divider()
        st.markdown(f"**{ena_meta.get('title', 'N/A')}**")
        if ena_meta.get("note"):
            st.caption(ena_meta["note"])
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Organism", ena_meta.get("organism", "N/A"))
        with col_m2:
            st.metric("Platform", ena_meta.get("platform", "N/A"))
        with col_m3:
            st.metric("Library", ena_meta.get("library_strategy", "N/A"))
        with col_m4:
            st.metric("Runs", ena_meta.get("n_runs", 0))
        with col_m5:
            st.metric("Total Size", f"{ena_meta.get('total_size_gb', 0)} GB")
        
        # Show run table
        runs = ena_meta.get("runs", [])
        if runs:
            runs_df = pd.DataFrame(runs)
            display_cols = ["run_accession", "sample_title", "library_layout", "read_count", "size_mb"]
            available_cols = [c for c in display_cols if c in runs_df.columns]
            st.dataframe(
                runs_df[available_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )
            if len(runs) > 50:
                st.caption(f"Showing 50 of {len(runs)} runs.")
        
        # Download size estimate
        size_est = estimate_download_size(ena_meta)
        st.caption(
            f"Estimated download: {size_est.get('total_size_gb', 0)} GB | "
            f"~{size_est.get('estimated_time_ftp_min', 0):.0f} min (FTP) / "
            f"~{size_est.get('estimated_time_aspera_min', 0):.0f} min (Aspera) | "
            f"PE: {size_est.get('paired_end', 0)}, SE: {size_est.get('single_end', 0)}"
        )
    elif ena_meta and "error" in ena_meta:
        st.error(ena_meta["error"])
    
    # ── Step 2: fetchngs Configuration and Launch ──
    st.divider()
    st.markdown("**Step 2: Configure and Launch fetchngs**")
    
    with st.form("fetchngs_launch_form", enter_to_submit=False):
        col_cfg1, col_cfg2 = st.columns([1, 1])
        
        with col_cfg1:
            st.markdown("**Output & Pipeline:**")
            fetchngs_output_dir = st.text_input(
                "Output Directory (UC Volume)",
                value=f"/Volumes/{os.environ.get('CORE_CATALOG_NAME', 'dhbl_discovery_us_dev')}/{os.environ.get('CORE_SCHEMA_NAME', 'genesis_schema')}/fetchngs_output",
                help="UC Volume path where FASTQs and samplesheet will be written.",
                key="fetchngs_output_dir"
            )
            fetchngs_project = st.text_input(
                "Project Name",
                value="",
                placeholder="e.g., neurodegeneration_rnaseq_2026",
                help="Creates a subdirectory and tags the audit trail.",
                key="fetchngs_project"
            )
            fetchngs_downstream = st.selectbox(
                "Target Downstream Pipeline",
                options=SUPPORTED_DOWNSTREAM_PIPELINES,
                index=0,
                help="Tailors samplesheet columns for the target pipeline. 'none' outputs full metadata.",
                format_func=lambda x: {
                    "scrnaseq": "scRNA-seq (nf-core/scrnaseq → STARsolo/Salmon)",
                    "rnaseq": "Bulk RNA-seq (nf-core/rnaseq → STAR/Salmon)",
                    "taxprofiler": "Taxonomic Profiling (nf-core/taxprofiler)",
                    "atacseq": "ATAC-seq (nf-core/atacseq)",
                    "none": "None (full metadata samplesheet)",
                }[x],
                key="fetchngs_downstream_pipeline"
            )
        
        with col_cfg2:
            st.markdown("**Download & Execution:**")
            fetchngs_download_method = st.selectbox(
                "Download Method",
                options=DOWNLOAD_METHODS,
                index=0,
                help="sratools: fasterq-dump (most compatible). aspera: fastest. ftp: universal fallback.",
                format_func=lambda x: {"sratools": "SRA Tools (fasterq-dump)", "aspera": "Aspera (fastest)", "ftp": "FTP (universal)"}[x],
                key="fetchngs_download_method"
            )
            fetchngs_version = st.text_input(
                "nf-core/fetchngs Version",
                value="1.12.0",
                help="Pin to a specific release for reproducibility.",
                key="fetchngs_version"
            )
            fetchngs_genome = st.selectbox(
                "Reference Genome (for downstream)",
                options=["GRCh38", "GRCm39", "GRCm38"],
                index=0,
                help="Used when auto-triggering downstream alignment pipeline.",
                key="fetchngs_genome"
            )
            fetchngs_trigger = st.checkbox(
                "Auto-trigger downstream pipeline on success",
                value=True,
                help="Automatically submit the target pipeline (scrnaseq/rnaseq) with fetchngs output samplesheet.",
                key="fetchngs_auto_trigger"
            )
        
        # Advanced: extra args for prefetch/Nextflow
        with st.expander("⚙️ Advanced Options", expanded=False):
            fetchngs_extra_args = st.text_input(
                "Extra Args",
                value="--max-size 50G",
                help=(
                    "Prefetch flags (--max-size, --min-size) are automatically routed to the "
                    "SRATOOLS_PREFETCH process config. Nextflow params (--skip_fastq_check) go to the CLI. "
                    "Default --max-size 50G prevents prefetch from silently skipping large SRA files (tool default is 20 GB)."
                ),
                key="fetchngs_extra_args"
            )
        
        st.divider()
        fetchngs_submit = st.form_submit_button("🚀 Launch Nextflow Pipeline", type="primary", use_container_width=True)
    
    # Handle form submission
    if fetchngs_submit:
        is_valid = True
        
        if not parsed_ids:
            st.error("Please enter at least one accession ID.")
            is_valid = False
        
        if not fetchngs_output_dir.startswith("/Volumes"):
            st.error("Output directory must be a UC Volume path (starting with /Volumes).")
            is_valid = False
        
        # Re-validate accessions
        if is_valid:
            val_result = validate_accessions(parsed_ids)
            if val_result["n_valid"] == 0:
                st.error("No valid accession IDs found. Check format and try again.")
                is_valid = False
        
        if is_valid:
            try:
                user_info = get_user_info()
                valid_ids = [v["accession"] for v in val_result["valid"]]
                
                with st.spinner(f"Submitting Nextflow fetchngs pipeline for {len(valid_ids)} accessions..."):
                    job_id, run_id = start_fetchngs_job(
                        accession_ids=valid_ids,
                        output_dir=fetchngs_output_dir,
                        nf_core_pipeline=fetchngs_downstream,
                        download_method=fetchngs_download_method,
                        pipeline_version=fetchngs_version,
                        genome=fetchngs_genome,
                        trigger_downstream=fetchngs_trigger,
                        project_name=fetchngs_project,
                        extra_args=fetchngs_extra_args,
                        user_info=user_info,
                    )
                
                host_name = os.getenv("DATABRICKS_HOSTNAME", "")
                if not host_name.startswith("https://"):
                    host_name = "https://" + host_name
                
                if job_id:
                    run_url = f"{host_name}/jobs/{job_id}/runs/{run_id}"
                else:
                    run_url = f"{host_name}/#job/run/{run_id}"
                
                st.success(f"fetchngs pipeline submitted! Run ID: {run_id}")
                st.link_button("View Run in Databricks", run_url, type="primary")
                
                st.info(
                    f"Pipeline: nf-core/fetchngs v{fetchngs_version} | "
                    f"Accessions: {len(valid_ids)} | "
                    f"Target: {fetchngs_downstream} | "
                    f"Auto-trigger: {'ON' if fetchngs_trigger else 'OFF'}"
                )
            
            except Exception as e:
                st.error(f"Failed to submit fetchngs job: {str(e)}")


def display_data_ingestion_tab():
    """Data Ingestion wrapper — provides GEO, SRA, scRNA-seq and Bulk RNA-seq Alignment sub-tabs."""
    fetchngs_tab, geo_tab, sra_tab, nf_tab, bulk_tab = st.tabs([
        "🧬 Nextflow Data Acquisition",
        "GEO (Processed Data)",
        "SRA (Raw Sequencing)",
        "scRNA-seq Alignment (FASTQ → h5ad)",
        "Bulk RNA-seq Alignment (FASTQ → Counts)"
    ])
    with fetchngs_tab:
        display_fetchngs_tab()
    with geo_tab:
        display_geo_ingestion_tab()
    with sra_tab:
        display_sra_ingestion_tab()
    with nf_tab:
        display_nextflow_alignment_tab()
    with bulk_tab:
        display_bulk_rnaseq_alignment_tab()

def display_embeddings_tab(deployed_models_df):
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("###### Generate Embeddings")
    with col2:        
        st.button("View Past Runs")    

    if len(deployed_models_df) > 0:
        with st.form("run_embedding_form"):
            st.write("Select Models:")


            st.dataframe(deployed_models_df, 
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="multi-row",
                            column_config={
                                "Model Id": None,
                                "Deploy Id" : None,
                                "Endpoint Name" : None
                            })
        
            st.write("NOTE: A result table will be created for EACH model selected.")

            col1, col2, col3 = st.columns([1,1,1], vertical_alignment="bottom")
            with col1:        
                st.text_input("Data Location:","")
                st.text_input("Result Schema Name:","")
                st.text_input("Result Table Prefix:","")
            
            with col2:
                st.write("")
                st.toggle("Perform Evaluation?")            
                st.text_input("Ground Truth Data Location:","")
                st.text_input("MLflow Experiment Name:","")
            
            st.form_submit_button("Generate Embeddings")

    else:
        st.write("There are no deployed models")

#load data for page
with st.spinner("Loading data"):
    if "available_single_cell_models_df" not in st.session_state:
            available_single_cell_models_df = get_available_models(ModelCategory.SINGLE_CELL)
            available_single_cell_models_df["model_labels"] = (available_single_cell_models_df["model_id"].astype(str) + " - " 
                                                + available_single_cell_models_df["model_display_name"].astype(str) + " [ " 
                                                + available_single_cell_models_df["model_uc_name"].astype(str) + " v"
                                                + available_single_cell_models_df["model_uc_version"].astype(str) + " ]"
                                                )
            st.session_state["available_single_cell_models_df"] = available_single_cell_models_df
    available_single_cell_models_df = st.session_state["available_single_cell_models_df"]

    if "deployed_single_cell_models_df" not in st.session_state:
        deployed_single_cell_models_df = get_deployed_models(ModelCategory.SINGLE_CELL)
        deployed_single_cell_models_df.columns = ["Model Id","Deploy Id", "Name", "Description", "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]

        st.session_state["deployed_single_cell_models_df"] = deployed_single_cell_models_df
    deployed_single_cell_models_df = st.session_state["deployed_single_cell_models_df"]



st.title(":material/microbiology:  Single Cell Studies")
require_module_access("single_cell", get_user_info())

# settings_tab, processing_tab, embeddings_tab = st.tabs([
#     "Settings", 
#     "Raw Single Cell Processing",
#     "Embeddings"
# ])

settings_tab, processing_tab = st.tabs([
    "Settings", 
    "Raw Single Cell Processing",
    # "Embeddings"
])


with settings_tab:
    display_settings_tab(available_single_cell_models_df,deployed_single_cell_models_df)

with processing_tab:
    # Sub-sections within processing tab
    st.markdown("### Raw Single Cell Analysis")
    
    # Custom CSS to make tab text bigger
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use tabs instead of nested expanders to avoid nesting issues
    ingest_tab, run_tab, view_tab = st.tabs(["Data Ingestion", "Run New Analysis", "View Analysis Results"])
    
    with ingest_tab:
        display_data_ingestion_tab()
    
    with run_tab:
        display_scanpy_analysis_tab()
    
    with view_tab:
        display_singlecell_results_viewer()

# with embeddings_tab:
#     display_embeddings_tab(deployed_single_cell_models_df)
