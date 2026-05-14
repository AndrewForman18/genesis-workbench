"""
Genesis Workbench - Module Authorization

Per-module access control based on workspace group membership.
Groups are configured in genesis_env.yml under permissions.module_access.

Usage in Streamlit views:
    from utils.authorization import require_module_access, check_module_access

    # Hard gate (shows error and stops rendering)
    require_module_access("disease_biology", user_info)

    # Soft check (returns bool for conditional UI)
    if check_module_access("protein_studies", user_info):
        render_protein_ui()
"""
import streamlit as st
from typing import List, Optional
from genesis_config import GenesisConfig


@st.cache_resource
def _get_cfg() -> GenesisConfig:
    return GenesisConfig.load()


def _get_module_groups(module_name: str) -> List[str]:
    """Get the list of groups allowed to access a module.
    Returns empty list if no restriction is configured (open access).
    """
    cfg = _get_cfg()
    module_access = cfg.raw.get("permissions", {}).get("module_access", {})
    groups = module_access.get(module_name, [])
    if isinstance(groups, str):
        groups = [groups]
    return groups


def check_module_access(module_name: str, user_info) -> bool:
    """Check if user has access to a module.
    Returns True if no restriction configured, user is admin, or user is in allowed group.
    """
    cfg = _get_cfg()
    admin_group = cfg.raw.get("permissions", {}).get("admin_group", "")
    allowed_groups = _get_module_groups(module_name)
    if not allowed_groups:
        return True
    user_groups = getattr(user_info, "user_groups", []) or []
    if admin_group and admin_group in user_groups:
        return True
    return bool(set(user_groups) & set(allowed_groups))


def require_module_access(module_name: str, user_info, message: Optional[str] = None):
    """Gate a Streamlit page - stops rendering if user lacks access.
    Call at the top of a view. Shows error and calls st.stop() if denied.
    """
    if check_module_access(module_name, user_info):
        return
    cfg = _get_cfg()
    allowed_groups = _get_module_groups(module_name)
    admin_group = cfg.raw.get("permissions", {}).get("admin_group", "")
    if not message:
        groups_str = "`, `".join(allowed_groups)
        message = (
            f"**Access Denied** - You do not have permission to use the "
            f"**{module_name.replace('_', ' ').title()}** module.\n\n"
            f"Required group membership: `{groups_str}`\n\n"
            f"Contact a workspace admin or member of `{admin_group}` to request access."
        )
    st.error(message, icon=":lock:")
    st.stop()


def get_accessible_modules(user_info) -> List[str]:
    """Return list of module names the user can access.
    Useful for building dynamic navigation.
    """
    cfg = _get_cfg()
    module_access = cfg.raw.get("permissions", {}).get("module_access", {})
    all_modules = list(module_access.keys())
    return [m for m in all_modules if check_module_access(m, user_info)]
