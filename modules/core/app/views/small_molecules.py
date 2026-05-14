import os
import json
import logging
import streamlit as st
import pandas as pd

from genesis_workbench.models import (ModelCategory,
                                      get_available_models,
                                      get_deployed_models)
from genesis_workbench.workbench import execute_workflow, execute_select_query
from utils.streamlit_helper import (get_user_info,
                                    display_import_model_uc_dialog,
                                    display_deploy_model_dialog)
from utils.authorization import require_module_access
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# ── Shared endpoint helper ────────────────────────────────────────────────

_workspace_client = None


def _get_ws_client():
    global _workspace_client
    if _workspace_client is None:
        _workspace_client = WorkspaceClient()
    return _workspace_client


def _query_endpoint(endpoint_name: str, inputs):
    """Query a model serving endpoint and return predictions."""
    try:
        logger.info(f"Querying endpoint: {endpoint_name}")
        response = _get_ws_client().serving_endpoints.query(
            name=endpoint_name,
            inputs=inputs,
        )
        logger.info(f"Response received from {endpoint_name}")
        return response.predictions
    except Exception as e:
        logger.error(f"Error querying {endpoint_name}: {e}", exc_info=True)
        raise


def get_batch_models(module_category: str) -> pd.DataFrame:
    """Gets batch models (job-based, non-endpoint models) for a given module category."""
    try:
        query = f"""SELECT 
                        s.key as name, 
                        s.value as description,
                        s.value as endpoint_name,
                        '' as cluster
                    FROM 
                        {os.environ['CORE_CATALOG_NAME']}.{os.environ['CORE_SCHEMA_NAME']}.settings s
                    WHERE 
                        s.module = '{module_category}' 
                        AND s.key LIKE '%job_id'
                    ORDER BY s.key"""
        return execute_select_query(query)
    except Exception:
        return pd.DataFrame(columns=['name', 'description', 'endpoint_name', 'cluster'])


# ── Page header ───────────────────────────────────────────────────────────

st.title(":material/vaccines: Small Molecules")

# ── Load model data ───────────────────────────────────────────────────────

with st.spinner("Loading data"):
    if "available_sm_models_df" not in st.session_state:
        available_sm_models_df = get_available_models(ModelCategory.SMALL_MOLECULES)
        available_sm_models_df["model_labels"] = (
            available_sm_models_df["model_id"].astype(str) + " - "
            + available_sm_models_df["model_display_name"].astype(str) + " [ "
            + available_sm_models_df["model_uc_name"].astype(str) + " v"
            + available_sm_models_df["model_uc_version"].astype(str) + " ]"
        )
        st.session_state["available_sm_models_df"] = available_sm_models_df
    available_sm_models_df = st.session_state["available_sm_models_df"]

    if "deployed_sm_models_df" not in st.session_state:
        rt_df = get_deployed_models(ModelCategory.SMALL_MOLECULES)
        rt_df.columns = ["Model Id", "Deploy Id", "Name", "Description",
                         "Model Name", "Source Version", "UC Name/Version", "Endpoint Name"]
        rt_df["Type"] = "Real-time"
        rt_df["Cluster"] = ""
        rows = [rt_df]
        try:
            batch_df = get_batch_models("small_molecules")
            if not batch_df.empty:
                batch_df.columns = ["Name", "Description", "Endpoint Name", "Cluster"]
                batch_df["Type"] = "Batch"
                batch_df["Model Id"] = None
                batch_df["Deploy Id"] = None
                batch_df["Model Name"] = None
                batch_df["Source Version"] = None
                batch_df["UC Name/Version"] = None
                rows.append(batch_df)
        except Exception:
            pass
        deployed_sm_models_df = pd.concat(rows, ignore_index=True)
        for col in ["Model Id", "Deploy Id"]:
            if col in deployed_sm_models_df.columns:
                deployed_sm_models_df[col] = deployed_sm_models_df[col].astype(str).replace("None", "")
        st.session_state["deployed_sm_models_df"] = deployed_sm_models_df
    deployed_sm_models_df = st.session_state["deployed_sm_models_df"]

user_info = get_user_info()
require_module_access("small_molecules", user_info)

# ── Tabs ──────────────────────────────────────────────────────────────────

(deployed_models_tab,
 docking_tab,
 binder_design_tab,
 ligand_binder_tab,
 motif_scaffolding_tab,
 admet_tab) = st.tabs([
    "Deployed Models",
    "Molecular Docking",
    "Protein Binder Design",
    "Ligand Binder Design",
    "Motif Scaffolding",
    "ADMET Prediction",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Deployed Models
# ══════════════════════════════════════════════════════════════════════════

with deployed_models_tab:
    st.subheader("Deployed Models")

    if deployed_sm_models_df is not None and not deployed_sm_models_df.empty:
        display_cols = [c for c in ["Name", "Description", "Type", "Endpoint Name",
                                     "Model Name", "Source Version", "UC Name/Version"]
                        if c in deployed_sm_models_df.columns]
        st.dataframe(deployed_sm_models_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No deployed small molecule models found.")

    st.divider()
    st.subheader("Available Models")

    if available_sm_models_df is not None and not available_sm_models_df.empty:
        st.dataframe(available_sm_models_df, use_container_width=True, hide_index=True)
    else:
        st.info("No available small molecule models found.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button(":material/add: Import Model from UC", use_container_width=True, key="sm_import"):
            display_import_model_uc_dialog(
                ModelCategory.SMALL_MOLECULES,
                success_callback=lambda: st.session_state.pop("available_sm_models_df", None),
            )

    with col2:
        if available_sm_models_df is not None and not available_sm_models_df.empty:
            selected = st.selectbox(
                "Select model to deploy",
                options=available_sm_models_df["model_labels"].tolist(),
                label_visibility="collapsed",
                key="sm_deploy_select",
            )
            if st.button(":material/rocket_launch: Deploy Selected Model", use_container_width=True, key="sm_deploy"):
                display_deploy_model_dialog(
                    selected,
                    success_callback=lambda: st.session_state.pop("deployed_sm_models_df", None),
                )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Molecular Docking (DiffDock)
# ══════════════════════════════════════════════════════════════════════════

with docking_tab:
    st.subheader("Molecular Docking — DiffDock")
    st.caption(
        "Predicts how a small molecule binds to a protein using diffusion-based "
        "blind docking. No pre-defined binding pocket required."
    )

    # Protein input
    st.markdown("**Protein Input**")
    protein_input_method = st.radio(
        "Provide protein structure via:",
        ["Upload PDB file", "Paste PDB text"],
        horizontal=True,
        key="dock_protein_method",
    )

    pdb_text = ""
    if protein_input_method == "Upload PDB file":
        pdb_file = st.file_uploader("Upload PDB file", type=["pdb"], key="dock_pdb_upload")
        if pdb_file is not None:
            pdb_text = pdb_file.getvalue().decode("utf-8")
            st.text_area("PDB Preview", pdb_text[:500] + "…", height=100, disabled=True)
    else:
        pdb_text = st.text_area(
            "Paste PDB text",
            height=150,
            placeholder="Paste PDB file content here…",
            key="dock_pdb_text",
        )

    # Ligand input
    st.markdown("**Ligand Input**")
    with st.form("docking_form", enter_to_submit=False):
        smiles_input = st.text_area(
            "SMILES string(s)",
            height=80,
            placeholder="Enter one SMILES per line, e.g.:\nCC(=O)Oc1ccccc1C(=O)O\nc1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
            key="dock_smiles",
        )
        n_poses = st.slider("Number of poses per ligand", 1, 20, 5, key="dock_n_poses")
        dock_submitted = st.form_submit_button("Dock")

    if dock_submitted:
        if not pdb_text.strip():
            st.warning("Please provide a protein structure (PDB).")
        elif not smiles_input.strip():
            st.warning("Please enter at least one SMILES string.")
        else:
            smiles_list = [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]

            with st.spinner("Step 1/2 — Computing ESM protein embeddings…"):
                try:
                    esm_response = _query_endpoint(
                        "gwb_diffdock_esm_embeddings_endpoint",
                        [{"pdb": pdb_text}],
                    )
                    embeddings = esm_response[0] if isinstance(esm_response, list) else esm_response
                    st.session_state["dock_embeddings"] = embeddings
                except Exception as e:
                    st.error(f"ESM embeddings error: {e}")
                    embeddings = None

            if embeddings is not None:
                with st.spinner("Step 2/2 — Running DiffDock blind docking…"):
                    try:
                        dock_inputs = []
                        for smi in smiles_list:
                            dock_inputs.append({
                                "pdb": pdb_text,
                                "smiles": smi,
                                "embeddings": embeddings,
                                "n_poses": n_poses,
                            })
                        dock_results = _query_endpoint(
                            "gwb_diffdock_endpoint",
                            dock_inputs,
                        )
                        st.session_state["dock_results"] = dock_results
                        st.session_state["dock_smiles_list"] = smiles_list
                    except Exception as e:
                        st.error(f"DiffDock error: {e}")

    if "dock_results" in st.session_state:
        results = st.session_state["dock_results"]
        smiles_list = st.session_state.get("dock_smiles_list", [])
        st.success(f"Docking complete — {len(smiles_list)} ligand(s) processed")

        for i, (smi, result) in enumerate(zip(smiles_list, results if isinstance(results, list) else [results])):
            with st.expander(f"Ligand {i + 1}: `{smi[:60]}`"):
                if isinstance(result, dict):
                    poses = result.get("poses", result.get("predictions", [result]))
                    if isinstance(poses, list):
                        rows = []
                        for j, pose in enumerate(poses):
                            if isinstance(pose, dict):
                                rows.append({
                                    "Pose": j + 1,
                                    "Confidence": pose.get("confidence", pose.get("score", "N/A")),
                                    "RMSD": pose.get("rmsd", "N/A"),
                                })
                            else:
                                rows.append({"Pose": j + 1, "Result": str(pose)[:200]})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # Download best pose if available
                    if isinstance(poses, list) and len(poses) > 0:
                        best = poses[0]
                        if isinstance(best, dict) and "sdf" in best:
                            st.download_button(
                                "Download best pose (SDF)",
                                best["sdf"],
                                file_name=f"diffdock_pose_{i + 1}.sdf",
                                key=f"dock_dl_{i}",
                            )
                else:
                    st.json(result)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Protein Binder Design (Proteina-Complexa)
# ══════════════════════════════════════════════════════════════════════════

with binder_design_tab:
    st.subheader("Protein Binder Design — Proteina-Complexa")
    st.caption(
        "Design novel proteins that bind to a target protein of interest using "
        "generative flow-matching."
    )

    # Target protein input
    st.markdown("**Target Protein**")
    target_method = st.radio(
        "Provide target via:",
        ["Upload PDB file", "Paste PDB text"],
        horizontal=True,
        key="binder_target_method",
    )

    target_pdb = ""
    if target_method == "Upload PDB file":
        target_file = st.file_uploader("Upload target PDB", type=["pdb"], key="binder_pdb_upload")
        if target_file is not None:
            target_pdb = target_file.getvalue().decode("utf-8")
            st.text_area("PDB Preview", target_pdb[:500] + "…", height=100, disabled=True)
    else:
        target_pdb = st.text_area(
            "Paste PDB text",
            height=150,
            placeholder="Paste target protein PDB content…",
            key="binder_pdb_text",
        )

    with st.form("binder_design_form", enter_to_submit=False):
        st.markdown("**Design Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            min_length = st.number_input("Min binder length (residues)", min_value=30, max_value=500, value=60, step=10, key="binder_min_len")
            n_designs = st.slider("Number of designs", 1, 20, 5, key="binder_n_designs")
        with col2:
            max_length = st.number_input("Max binder length (residues)", min_value=30, max_value=500, value=120, step=10, key="binder_max_len")
            validate_esmfold = st.checkbox("Validate designs with ESMFold", value=False, key="binder_validate")

        binder_submitted = st.form_submit_button("Design Binders")

    if binder_submitted:
        if not target_pdb.strip():
            st.warning("Please provide a target protein structure (PDB).")
        elif min_length > max_length:
            st.warning("Min length must be ≤ max length.")
        else:
            with st.spinner("Running Proteina-Complexa binder design…"):
                try:
                    inputs = [{
                        "pdb": target_pdb,
                        "min_length": min_length,
                        "max_length": max_length,
                        "n_designs": n_designs,
                    }]
                    binder_results = _query_endpoint(
                        "gwb_proteina_complexa_endpoint",
                        inputs,
                    )
                    st.session_state["binder_results"] = binder_results
                    st.session_state["binder_validate"] = validate_esmfold
                except Exception as e:
                    st.error(f"Proteina-Complexa error: {e}")

    if "binder_results" in st.session_state:
        binder_results = st.session_state["binder_results"]
        do_validate = st.session_state.get("binder_validate", False)

        designs = binder_results if isinstance(binder_results, list) else [binder_results]
        # Flatten if nested
        if len(designs) == 1 and isinstance(designs[0], list):
            designs = designs[0]

        st.success(f"Generated {len(designs)} binder design(s)")

        for i, design in enumerate(designs):
            label = f"Design {i + 1}"
            if isinstance(design, dict):
                score = design.get("reward_score", design.get("score", ""))
                seq = design.get("sequence", "")
                if score:
                    label += f" — reward: {score}"
            else:
                seq = str(design) if not isinstance(design, dict) else ""

            with st.expander(label):
                if isinstance(design, dict):
                    st.json(design)
                else:
                    st.code(str(design), language=None)

                if do_validate and seq:
                    if st.button(f"Validate with ESMFold", key=f"binder_val_{i}"):
                        with st.spinner("Running ESMFold validation…"):
                            try:
                                from utils.protein_design import hit_esmfold
                                pdb_str = hit_esmfold(seq)
                                st.session_state[f"binder_val_pdb_{i}"] = pdb_str
                            except Exception as e:
                                st.error(f"ESMFold error: {e}")

                    if f"binder_val_pdb_{i}" in st.session_state:
                        from utils.molstar_tools import molstar_html_singlebody
                        st.components.v1.html(
                            molstar_html_singlebody(st.session_state[f"binder_val_pdb_{i}"]),
                            height=400,
                        )
                        st.download_button(
                            "Download validated PDB",
                            st.session_state[f"binder_val_pdb_{i}"],
                            file_name=f"binder_design_{i + 1}.pdb",
                            key=f"binder_dl_{i}",
                        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Ligand Binder Design (Proteina-Complexa Ligand)
# ══════════════════════════════════════════════════════════════════════════

with ligand_binder_tab:
    st.subheader("Ligand Binder Design — Proteina-Complexa Ligand")
    st.caption(
        "Design proteins that bind a specified small molecule. Input a SMILES "
        "string and the model generates novel protein sequences."
    )

    with st.form("ligand_binder_form", enter_to_submit=False):
        ligand_smiles = st.text_input(
            "Ligand SMILES",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
            key="lig_smiles",
        )

        st.markdown("**Design Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            lig_min_len = st.number_input("Min protein length", min_value=30, max_value=500, value=60, step=10, key="lig_min_len")
            lig_n_designs = st.slider("Number of designs", 1, 20, 5, key="lig_n_designs")
        with col2:
            lig_max_len = st.number_input("Max protein length", min_value=30, max_value=500, value=120, step=10, key="lig_max_len")

        st.markdown("**Validation Options**")
        col3, col4 = st.columns(2)
        with col3:
            lig_validate_esmfold = st.checkbox("ESMFold structure validation", value=False, key="lig_val_esm")
        with col4:
            lig_validate_diffdock = st.checkbox("DiffDock binding validation", value=False, key="lig_val_dock")

        lig_submitted = st.form_submit_button("Design Ligand Binders")

    if lig_submitted:
        if not ligand_smiles.strip():
            st.warning("Please enter a SMILES string.")
        elif lig_min_len > lig_max_len:
            st.warning("Min length must be ≤ max length.")
        else:
            with st.spinner("Running Proteina-Complexa Ligand design…"):
                try:
                    inputs = [{
                        "smiles": ligand_smiles.strip(),
                        "min_length": lig_min_len,
                        "max_length": lig_max_len,
                        "n_designs": lig_n_designs,
                    }]
                    lig_results = _query_endpoint(
                        "gwb_proteina_complexa_ligand_endpoint",
                        inputs,
                    )
                    st.session_state["lig_results"] = lig_results
                    st.session_state["lig_smiles_input"] = ligand_smiles.strip()
                    st.session_state["lig_val_esm_flag"] = lig_validate_esmfold
                    st.session_state["lig_val_dock_flag"] = lig_validate_diffdock
                except Exception as e:
                    st.error(f"Proteina-Complexa Ligand error: {e}")

    if "lig_results" in st.session_state:
        lig_results = st.session_state["lig_results"]
        do_esm = st.session_state.get("lig_val_esm_flag", False)
        do_dock = st.session_state.get("lig_val_dock_flag", False)
        lig_smi = st.session_state.get("lig_smiles_input", "")

        designs = lig_results if isinstance(lig_results, list) else [lig_results]
        if len(designs) == 1 and isinstance(designs[0], list):
            designs = designs[0]

        st.success(f"Generated {len(designs)} ligand binder design(s)")

        for i, design in enumerate(designs):
            label = f"Design {i + 1}"
            seq = ""
            if isinstance(design, dict):
                seq = design.get("sequence", "")
                score = design.get("reward_score", design.get("score", ""))
                if score:
                    label += f" — score: {score}"

            with st.expander(label):
                if isinstance(design, dict):
                    st.json(design)
                else:
                    seq = str(design) if isinstance(design, str) else ""
                    st.code(str(design), language=None)

                if do_esm and seq:
                    if st.button("Validate with ESMFold", key=f"lig_esm_{i}"):
                        with st.spinner("Running ESMFold…"):
                            try:
                                from utils.protein_design import hit_esmfold
                                pdb_str = hit_esmfold(seq)
                                st.session_state[f"lig_esm_pdb_{i}"] = pdb_str
                            except Exception as e:
                                st.error(f"ESMFold error: {e}")

                    if f"lig_esm_pdb_{i}" in st.session_state:
                        from utils.molstar_tools import molstar_html_singlebody
                        st.components.v1.html(
                            molstar_html_singlebody(st.session_state[f"lig_esm_pdb_{i}"]),
                            height=400,
                        )

                if do_dock and seq and f"lig_esm_pdb_{i}" in st.session_state:
                    if st.button("Validate binding with DiffDock", key=f"lig_dock_{i}"):
                        with st.spinner("Running DiffDock validation…"):
                            try:
                                designed_pdb = st.session_state[f"lig_esm_pdb_{i}"]
                                esm_emb = _query_endpoint(
                                    "gwb_diffdock_esm_embeddings_endpoint",
                                    [{"pdb": designed_pdb}],
                                )
                                emb = esm_emb[0] if isinstance(esm_emb, list) else esm_emb
                                dock_result = _query_endpoint(
                                    "gwb_diffdock_endpoint",
                                    [{
                                        "pdb": designed_pdb,
                                        "smiles": lig_smi,
                                        "embeddings": emb,
                                        "n_poses": 3,
                                    }],
                                )
                                st.session_state[f"lig_dock_result_{i}"] = dock_result
                            except Exception as e:
                                st.error(f"DiffDock error: {e}")

                    if f"lig_dock_result_{i}" in st.session_state:
                        st.markdown("**DiffDock Binding Validation**")
                        st.json(st.session_state[f"lig_dock_result_{i}"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Motif Scaffolding (Proteina-Complexa AME — batch)
# ══════════════════════════════════════════════════════════════════════════

with motif_scaffolding_tab:
    st.subheader("Motif Scaffolding — Proteina-Complexa AME")
    st.caption(
        "Transplant a functional motif (e.g., enzyme active site, binding loop) "
        "into a new, stable protein scaffold. This runs as a batch job."
    )

    motif_file = st.file_uploader("Upload motif PDB file", type=["pdb"], key="motif_pdb_upload")
    motif_pdb_text = ""
    if motif_file is not None:
        motif_pdb_text = motif_file.getvalue().decode("utf-8")
        st.text_area("PDB Preview", motif_pdb_text[:500] + "…", height=100, disabled=True)

    with st.form("motif_scaffolding_form", enter_to_submit=False):
        st.markdown("**Scaffolding Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            motif_chain_id = st.text_input("Motif chain ID", value="A", max_chars=1, key="motif_chain")
            scaffold_min_len = st.number_input("Min scaffold length", min_value=50, max_value=500, value=80, step=10, key="motif_min_len")
            n_scaffolds = st.slider("Number of scaffold designs", 1, 20, 5, key="motif_n_designs")
        with col2:
            scaffold_max_len = st.number_input("Max scaffold length", min_value=50, max_value=500, value=150, step=10, key="motif_max_len")
            motif_validate_mpnn = st.checkbox("ProteinMPNN sequence optimization", value=False, key="motif_val_mpnn")
            motif_validate_esm = st.checkbox("ESMFold structure validation", value=False, key="motif_val_esm")

        st.markdown("**MLflow Tracking**")
        col3, col4 = st.columns(2)
        with col3:
            motif_experiment = st.text_input("MLflow Experiment Name", value="motif_scaffolding", key="motif_exp")
        with col4:
            motif_run_name = st.text_input("MLflow Run Name", placeholder="e.g. active_site_scaffold_v1", key="motif_run")

        motif_submitted = st.form_submit_button("Generate Scaffolds")

    if motif_submitted:
        if not motif_pdb_text.strip():
            st.warning("Please upload a motif PDB file.")
        elif not motif_run_name.strip():
            st.warning("MLflow run name is required.")
        elif scaffold_min_len > scaffold_max_len:
            st.warning("Min length must be ≤ max length.")
        else:
            with st.spinner("Submitting Proteina-Complexa AME batch job…"):
                try:
                    job_id = os.environ.get("PROTEINA_COMPLEXA_AME_JOB_ID", "")
                    if not job_id:
                        st.error(
                            "Motif scaffolding job ID not configured. "
                            "Please check PROTEINA_COMPLEXA_AME_JOB_ID in the app environment."
                        )
                    else:
                        job_run_id = execute_workflow(
                            job_id=job_id,
                            params={
                                "catalog": os.environ["CORE_CATALOG_NAME"],
                                "schema": os.environ["CORE_SCHEMA_NAME"],
                                "motif_pdb": motif_pdb_text,
                                "motif_chain_id": motif_chain_id,
                                "min_length": str(scaffold_min_len),
                                "max_length": str(scaffold_max_len),
                                "n_designs": str(n_scaffolds),
                                "optimize_mpnn": str(motif_validate_mpnn),
                                "validate_esmfold": str(motif_validate_esm),
                                "mlflow_experiment_name": motif_experiment,
                                "mlflow_run_name": motif_run_name,
                                "user_email": user_info.user_email,
                            },
                        )
                        st.success(f"Batch job submitted! Run ID: **{job_run_id}**")
                        st.info(
                            "This job may take several minutes. Monitor progress in the "
                            "**Monitoring and Alerts** tab or check MLflow for results."
                        )
                except Exception as e:
                    st.error(f"Error submitting scaffolding job: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — ADMET Prediction (Chemprop)
# ══════════════════════════════════════════════════════════════════════════

with admet_tab:
    st.subheader("ADMET Prediction — Chemprop")
    st.caption(
        "Predict drug-likeness and toxicity properties for small molecules using "
        "directed message-passing neural networks (D-MPNN)."
    )

    with st.form("admet_form", enter_to_submit=False):
        admet_smiles = st.text_area(
            "SMILES string(s)",
            height=100,
            placeholder="Enter one SMILES per line, e.g.:\nCC(=O)Oc1ccccc1C(=O)O\nCCO\nc1ccccc1",
            key="admet_smiles",
        )

        st.markdown("**Select Prediction Types**")
        col1, col2, col3 = st.columns(3)
        with col1:
            run_bbbp = st.checkbox("Blood-Brain Barrier Penetration (BBBP)", value=True, key="admet_bbbp")
        with col2:
            run_clintox = st.checkbox("Clinical Toxicity (ClinTox)", value=True, key="admet_clintox")
        with col3:
            run_admet = st.checkbox("Multi-task ADMET (10 properties)", value=True, key="admet_multi")

        admet_submitted = st.form_submit_button("Predict")

    if admet_submitted:
        if not admet_smiles.strip():
            st.warning("Please enter at least one SMILES string.")
        elif not (run_bbbp or run_clintox or run_admet):
            st.warning("Please select at least one prediction type.")
        else:
            smiles_list = [s.strip() for s in admet_smiles.strip().splitlines() if s.strip()]
            all_results = {}

            if run_bbbp:
                with st.spinner("Running BBBP prediction…"):
                    try:
                        bbbp_preds = _query_endpoint(
                            "gwb_chemprop_bbbp_endpoint",
                            smiles_list,
                        )
                        all_results["BBBP"] = bbbp_preds
                    except Exception as e:
                        st.error(f"BBBP error: {e}")

            if run_clintox:
                with st.spinner("Running ClinTox prediction…"):
                    try:
                        clintox_preds = _query_endpoint(
                            "gwb_chemprop_clintox_endpoint",
                            smiles_list,
                        )
                        all_results["ClinTox"] = clintox_preds
                    except Exception as e:
                        st.error(f"ClinTox error: {e}")

            if run_admet:
                with st.spinner("Running multi-task ADMET prediction…"):
                    try:
                        admet_preds = _query_endpoint(
                            "gwb_chemprop_admet_endpoint",
                            smiles_list,
                        )
                        all_results["ADMET"] = admet_preds
                    except Exception as e:
                        st.error(f"ADMET error: {e}")

            if all_results:
                st.session_state["admet_results"] = all_results
                st.session_state["admet_smiles_list"] = smiles_list

    if "admet_results" in st.session_state:
        results = st.session_state["admet_results"]
        smiles_list = st.session_state.get("admet_smiles_list", [])

        st.success(f"Predictions complete for {len(smiles_list)} molecule(s)")

        for model_name, preds in results.items():
            st.markdown(f"#### {model_name}")

            if isinstance(preds, list) and len(preds) == len(smiles_list):
                rows = []
                for smi, pred in zip(smiles_list, preds):
                    row = {"SMILES": smi}
                    if isinstance(pred, dict):
                        row.update(pred)
                    elif isinstance(pred, (list, tuple)):
                        # Multi-task output: enumerate properties
                        for j, val in enumerate(pred):
                            row[f"Property_{j + 1}"] = val
                    else:
                        row["Prediction"] = pred
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                # Fallback: display raw predictions
                st.json(preds)

            st.divider()
