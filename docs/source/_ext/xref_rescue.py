"""Hook missing Python references into intersphinx inventories."""

from __future__ import annotations

from docutils import nodes
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


def setup(app):
    app.connect("missing-reference", on_missing)
    return {"version": "1.0", "parallel_read_safe": True}


def on_missing(app, env, node, contnode):
    """Attempt to resolve unresolved cross-references via intersphinx inventories."""

    target = node.get("reftarget") or ""
    reftype = node.get("reftype") or ""

    invs = getattr(app.builder, "intersphinx_named_inventory", {})
    for _inv_name, inventory in invs.items():
        for role, mapping in inventory.items():
            if reftype and role.split(":")[0] != reftype.split(":")[0]:
                continue
            if target in mapping:
                return _make_ref(app, mapping[target], contnode)
            for fullname, uri in mapping.items():
                if fullname.endswith("." + target):
                    return _make_ref(app, uri, contnode)
    return None


def _make_ref(app, uri, contnode):
    target_uri = uri
    if "#" not in target_uri:
        target_uri = target_uri.rstrip("/") + "/index.html"
    ref = nodes.reference("", "", internal=False)
    ref["refuri"] = app.builder.get_relative_uri(app.env.docname, target_uri)
    ref.append(contnode)
    return ref
