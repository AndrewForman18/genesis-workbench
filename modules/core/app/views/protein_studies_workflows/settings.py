import streamlit as st
import pandas as pd
from utils.streamlit_helper import (display_import_model_uc_dialog,
                                    display_deploy_model_dialog)
from genesis_workbench.models import ModelCategory


def render(available_protein_models_df, deployed_protein_models_df):
    """Render the Deployed Models / Settings tab."""

    st.subheader("Deployed Models")

    if deployed_protein_models_df is not None and not deployed_protein_models_df.empty:
        display_cols = [c for c in ["Name", "Description", "Type", "Endpoint Name",
                                     "Model Name", "Source Version", "UC Name/Version"]
                        if c in deployed_protein_models_df.columns]
        st.dataframe(deployed_protein_models_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No deployed protein models found.")

    st.divider()
    st.subheader("Available Models")

    if available_protein_models_df is not None and not available_protein_models_df.empty:
        st.dataframe(available_protein_models_df, use_container_width=True, hide_index=True)
    else:
        st.info("No available protein models found.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button(":material/add: Import Model from UC", use_container_width=True):
            display_import_model_uc_dialog(
                ModelCategory.PROTEIN_STUDIES,
                success_callback=lambda: st.session_state.pop("available_protein_models_df", None),
            )

    with col2:
        if available_protein_models_df is not None and not available_protein_models_df.empty:
            selected = st.selectbox(
                "Select model to deploy",
                options=available_protein_models_df["model_labels"].tolist(),
                label_visibility="collapsed",
            )
            if st.button(":material/rocket_launch: Deploy Selected Model", use_container_width=True):
                display_deploy_model_dialog(
                    selected,
                    success_callback=lambda: st.session_state.pop("deployed_protein_models_df", None),
                )
