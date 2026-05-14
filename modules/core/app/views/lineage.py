"""
Genesis Workbench — Lineage & Provenance
Interactive visualization and exploration of the metadata tier DAG.
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from databricks.sdk import WorkspaceClient
from databricks import sql as databricks_sql

# ─── Page config ─────────────────────────────────────────────────────────────
st.title(":material/account_tree: Lineage and Provenance")
st.caption("Explore the Genesis Workbench metadata tier — assets, lineage edges, and run history.")

# ─── Constants ────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "bronze": "#CD7F32",
    "silver": "#C0C0C0",
    "gold": "#FFD700",
    "platinum": "#E5E4E2",
    "metadata": "#87CEEB",
}
TIER_ORDER = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3, "metadata": 4}

EDGE_COLORS = {
    "consumed_by": "#333333",
    "derived_from": "#666666",
    "annotated_by": "#0066CC",
    "triggered_by": "#CC6600",
    "registered_as": "#009900",
}
EDGE_DASHES = {
    "consumed_by": "solid",
    "derived_from": "dash",
    "annotated_by": "dot",
    "triggered_by": "dashdot",
    "registered_as": "solid",
}

CATALOG = os.environ.get("CORE_CATALOG_NAME", "dhbl_discovery_us_dev")
SCHEMA = os.environ.get("CORE_SCHEMA_NAME", "genesis_schema")
FQ_SCHEMA = f"{CATALOG}.{SCHEMA}"


# ─── SQL Connection ───────────────────────────────────────────────────────────
def get_sql_connection():
    w = WorkspaceClient()
    host = (os.environ.get("DATABRICKS_HOST") or w.config.host or "").replace("https://", "")
    return databricks_sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{os.environ['SQL_WAREHOUSE']}",
        credentials_provider=lambda: w.config.authenticate,
    )


def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame."""
    with get_sql_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)


# ─── Cached data loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_assets() -> pd.DataFrame:
    return run_query(f"""
        SELECT asset_id, asset_path, display_name, medallion_tier, module,
               asset_type, format, CAST(tags AS STRING) AS tags,
               row_count, is_active, created_at, updated_at
        FROM {FQ_SCHEMA}.asset_registry
        WHERE is_active = true
        ORDER BY medallion_tier, module, display_name
    """)


@st.cache_data(ttl=300)
def load_edges() -> pd.DataFrame:
    return run_query(f"""
        SELECT source_asset_id, target_asset_id, relationship_type,
               step_name, run_id, module, created_at
        FROM {FQ_SCHEMA}.workflow_lineage
    """)


@st.cache_data(ttl=300)
def load_runs() -> pd.DataFrame:
    return run_query(f"""
        SELECT run_id, run_source, module, status,
               CAST(parameters AS STRING) AS parameters,
               CAST(metrics AS STRING) AS metrics,
               owner_email, started_at, ended_at
        FROM {FQ_SCHEMA}.run_catalog_vw
        ORDER BY started_at DESC
    """)


# ─── Refresh button ──────────────────────────────────────────────────────────
col_refresh, col_spacer = st.columns([1, 5])
with col_refresh:
    if st.button("↻ Refresh", help="Clear cached data and reload from metadata tables"):
        st.cache_data.clear()
        st.rerun()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_dag, tab_browser, tab_runs, tab_trace = st.tabs(
    ["🔗 Lineage DAG", "📦 Asset Browser", "🕒 Run History", "🔍 Trace Explorer"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Lineage DAG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dag:
    assets_df = load_assets()
    edges_df = load_edges()

    if assets_df.empty:
        st.info("No assets found in asset_registry. Run the seed data in `00_metadata_tier_ddl` first.")
    else:
        # Module filter
        all_modules = sorted(assets_df["module"].dropna().unique().tolist())
        selected_modules = st.multiselect(
            "Filter by module", all_modules, default=all_modules, key="dag_module_filter"
        )

        # Filter data
        filtered_assets = assets_df[assets_df["module"].isin(selected_modules)]
        valid_ids = set(filtered_assets["asset_id"].tolist())
        filtered_edges = edges_df[
            (edges_df["source_asset_id"].isin(valid_ids))
            & (edges_df["target_asset_id"].isin(valid_ids))
        ]

        if filtered_assets.empty:
            st.warning("No assets match the selected module filter.")
        else:
            # Build node positions — tiered left-to-right layout
            # X: tier order; Y: spread within tier
            tier_counts = {}
            node_x, node_y = {}, {}
            for _, row in filtered_assets.iterrows():
                tier = row["medallion_tier"] or "metadata"
                x = TIER_ORDER.get(tier, 2)
                tier_counts.setdefault(x, 0)
                node_x[row["asset_id"]] = x
                node_y[row["asset_id"]] = tier_counts[x]
                tier_counts[x] += 1

            # Normalize Y to spread evenly
            for x_val in tier_counts:
                count = tier_counts[x_val]
                if count > 1:
                    for aid, y_val in node_y.items():
                        if node_x[aid] == x_val:
                            node_y[aid] = (y_val - (count - 1) / 2) * 1.2

            # Build edge traces (one per relationship type for legend)
            edge_traces = []
            for rel_type in filtered_edges["relationship_type"].unique():
                rel_edges = filtered_edges[filtered_edges["relationship_type"] == rel_type]
                edge_x, edge_y = [], []
                for _, e in rel_edges.iterrows():
                    src, tgt = e["source_asset_id"], e["target_asset_id"]
                    if src in node_x and tgt in node_x:
                        edge_x += [node_x[src], node_x[tgt], None]
                        edge_y += [node_y[src], node_y[tgt], None]
                edge_traces.append(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(
                            color=EDGE_COLORS.get(rel_type, "#999"),
                            width=1.5,
                            dash=EDGE_DASHES.get(rel_type, "solid"),
                        ),
                        hoverinfo="text",
                        hovertext=rel_type,
                        name=rel_type,
                        legendgroup="edges",
                    )
                )

            # Build node traces (one per tier for legend)
            node_traces = []
            for tier, color in TIER_COLORS.items():
                tier_assets = filtered_assets[filtered_assets["medallion_tier"] == tier]
                if tier_assets.empty:
                    continue
                nx_list, ny_list, labels, hovers = [], [], [], []
                for _, row in tier_assets.iterrows():
                    aid = row["asset_id"]
                    if aid in node_x:
                        nx_list.append(node_x[aid])
                        ny_list.append(node_y[aid])
                        name = row["display_name"] or aid[:20]
                        labels.append(name[:25])
                        hovers.append(
                            f"<b>{name}</b><br>"
                            f"Type: {row['asset_type']}<br>"
                            f"Module: {row['module']}<br>"
                            f"Format: {row['format']}<br>"
                            f"Tier: {tier}<br>"
                            f"Rows: {row['row_count'] or 'N/A'}"
                        )
                node_traces.append(
                    go.Scatter(
                        x=nx_list,
                        y=ny_list,
                        mode="markers+text",
                        marker=dict(size=18, color=color, line=dict(width=2, color="#333")),
                        text=labels,
                        textposition="bottom center",
                        textfont=dict(size=9),
                        hoverinfo="text",
                        hovertext=hovers,
                        name=f"{tier.title()}",
                        legendgroup="tiers",
                    )
                )

            # Assemble figure
            fig = go.Figure(data=edge_traces + node_traces)
            fig.update_layout(
                title="Genesis Workbench — Lineage DAG",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=True,
                    tickmode="array",
                    tickvals=list(TIER_ORDER.values()),
                    ticktext=[t.title() for t in TIER_ORDER.keys()],
                    title="Medallion Tier",
                ),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                margin=dict(l=40, r=40, t=80, b=40),
                plot_bgcolor="#FAFAFA",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Assets", len(filtered_assets))
            c2.metric("Edges", len(filtered_edges))
            c3.metric("Modules", len(selected_modules))
            c4.metric("Tiers", filtered_assets["medallion_tier"].nunique())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Asset Browser
# ═══════════════════════════════════════════════════════════════════════════════
with tab_browser:
    assets_df = load_assets()

    if assets_df.empty:
        st.info("No assets in registry.")
    else:
        # Filters
        col_tier, col_mod = st.columns(2)
        with col_tier:
            tier_options = sorted(assets_df["medallion_tier"].dropna().unique().tolist())
            tier_filter = st.multiselect(
                "Medallion Tier", tier_options, default=tier_options, key="browser_tier"
            )
        with col_mod:
            mod_options = sorted(assets_df["module"].dropna().unique().tolist())
            mod_filter = st.multiselect(
                "Module", mod_options, default=mod_options, key="browser_module"
            )

        # Apply filters
        display_df = assets_df[
            (assets_df["medallion_tier"].isin(tier_filter))
            & (assets_df["module"].isin(mod_filter))
        ][["display_name", "medallion_tier", "module", "asset_type", "format", "row_count", "updated_at"]].copy()

        # Display
        st.dataframe(
            display_df,
            column_config={
                "display_name": st.column_config.TextColumn("Name", width="large"),
                "medallion_tier": st.column_config.TextColumn("Tier", width="small"),
                "module": st.column_config.TextColumn("Module", width="medium"),
                "asset_type": st.column_config.TextColumn("Type", width="small"),
                "format": st.column_config.TextColumn("Format", width="small"),
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
                "updated_at": st.column_config.DatetimeColumn("Updated", format="YYYY-MM-DD HH:mm"),
            },
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Showing {len(display_df)} of {len(assets_df)} active assets")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Run History
# ═══════════════════════════════════════════════════════════════════════════════
with tab_runs:
    runs_df = load_runs()

    if runs_df.empty:
        st.info("No runs found in run_catalog_vw.")
    else:
        # Filters
        col_rmod, col_status = st.columns(2)
        with col_rmod:
            run_modules = sorted(runs_df["module"].dropna().unique().tolist())
            run_mod_filter = st.multiselect(
                "Module", run_modules, default=run_modules, key="runs_module"
            )
        with col_status:
            statuses = sorted(runs_df["status"].dropna().unique().tolist())
            status_filter = st.multiselect(
                "Status", statuses, default=statuses, key="runs_status"
            )

        # Filter and compute duration
        filtered_runs = runs_df[
            (runs_df["module"].isin(run_mod_filter)) & (runs_df["status"].isin(status_filter))
        ].copy()

        # Compute duration
        if not filtered_runs.empty:
            filtered_runs["started_at"] = pd.to_datetime(filtered_runs["started_at"], errors="coerce")
            filtered_runs["ended_at"] = pd.to_datetime(filtered_runs["ended_at"], errors="coerce")
            filtered_runs["duration"] = (
                filtered_runs["ended_at"] - filtered_runs["started_at"]
            ).apply(lambda x: str(x).split(".")[0] if pd.notna(x) else "—")

        display_runs = filtered_runs[
            ["run_id", "run_source", "module", "status", "owner_email", "started_at", "ended_at", "duration"]
        ]

        st.dataframe(
            display_runs,
            column_config={
                "run_id": st.column_config.TextColumn("Run ID", width="medium"),
                "run_source": st.column_config.TextColumn("Source", width="small"),
                "module": st.column_config.TextColumn("Module", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "owner_email": st.column_config.TextColumn("Owner", width="medium"),
                "started_at": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm"),
                "ended_at": st.column_config.DatetimeColumn("Ended", format="YYYY-MM-DD HH:mm"),
                "duration": st.column_config.TextColumn("Duration", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Showing {len(display_runs)} runs")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Trace Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trace:
    assets_df = load_assets()
    edges_df = load_edges()

    if assets_df.empty:
        st.info("No assets to trace.")
    else:
        # Build lookup
        asset_lookup = assets_df.set_index("asset_id").to_dict("index")
        asset_names = {
            row["asset_id"]: row["display_name"] or row["asset_id"][:20]
            for _, row in assets_df.iterrows()
        }

        # Asset selector
        options = sorted(asset_names.items(), key=lambda x: x[1])
        selected_label = st.selectbox(
            "Select an asset to trace",
            options=[f"{name} ({aid[:8]}...)" for aid, name in options],
            key="trace_asset",
        )

        if selected_label:
            # Extract asset_id from selection
            selected_aid = [aid for aid, name in options if f"{name} ({aid[:8]}...)" == selected_label][0]

            st.markdown(f"**Tracing:** `{asset_names[selected_aid]}`")
            asset_info = asset_lookup.get(selected_aid, {})
            st.caption(
                f"Tier: {asset_info.get('medallion_tier', '?')} · "
                f"Module: {asset_info.get('module', '?')} · "
                f"Type: {asset_info.get('asset_type', '?')} · "
                f"Format: {asset_info.get('format', '?')}"
            )

            st.divider()

            col_up, col_down = st.columns(2)

            # Upstream: edges where this asset is the TARGET
            with col_up:
                st.subheader("⬆️ Upstream Dependencies")
                upstream = edges_df[edges_df["target_asset_id"] == selected_aid]
                if upstream.empty:
                    st.caption("No upstream dependencies (root node).")
                else:
                    for _, edge in upstream.iterrows():
                        src_id = edge["source_asset_id"]
                        src_info = asset_lookup.get(src_id, {})
                        src_name = src_info.get("display_name", src_id[:20])
                        src_tier = src_info.get("medallion_tier", "?")
                        rel = edge["relationship_type"]
                        step = edge["step_name"] or ""
                        st.markdown(
                            f"**{src_name}** &nbsp; "
                            f"`{src_tier}` · _{rel}_ · {step}"
                        )

            # Downstream: edges where this asset is the SOURCE
            with col_down:
                st.subheader("⬇️ Downstream Consumers")
                downstream = edges_df[edges_df["source_asset_id"] == selected_aid]
                if downstream.empty:
                    st.caption("No downstream consumers (leaf node).")
                else:
                    for _, edge in downstream.iterrows():
                        tgt_id = edge["target_asset_id"]
                        tgt_info = asset_lookup.get(tgt_id, {})
                        tgt_name = tgt_info.get("display_name", tgt_id[:20])
                        tgt_tier = tgt_info.get("medallion_tier", "?")
                        rel = edge["relationship_type"]
                        step = edge["step_name"] or ""
                        st.markdown(
                            f"**{tgt_name}** &nbsp; "
                            f"`{tgt_tier}` · _{rel}_ · {step}"
                        )
