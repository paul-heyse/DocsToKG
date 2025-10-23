# docs/_ext/xref_rescue.py
from docutils import nodes
from sphinx.util import logging

logger = logging.getLogger(__name__)


def setup(app):
    app.connect("missing-reference", on_missing)
    return {"version": "1.0", "parallel_read_safe": True}


def on_missing(app, env, node, contnode):
    """
    Try to resolve missing refs by probing intersphinx inventories.
    Return a reference node to silence warnings, or None to let Sphinx warn.
    """
    target = node.get("reftarget") or ""
    reftype = node.get("reftype") or ""
    # Use Sphinx's intersphinx inventories (already loaded via intersphinx_mapping)
    invs = getattr(app.builder, "intersphinx_named_inventory", {})
    # Probe across all domains/roles in external inventories
    for invname, inv in invs.items():
        # inv is like {'py:class': {'pkg.mod.Name': '...'}, ...}
        for role, mapping in inv.items():
            # prefer python roles when our reftype looks pythonic
            if reftype and not role.startswith(reftype.split(":")[0]):
                pass
            # exact, then suffix match
            if target in mapping:
                return _ref(app, mapping[target], contnode)
            for fullname, uri in mapping.items():
                if fullname.endswith("." + target):
                    return _ref(app, uri, contnode)
    return None


def _ref(app, uri, contnode):
    ref = nodes.reference("", "", internal=False)
    ref["refuri"] = app.builder.get_relative_uri(app.env.docname, uri)
    ref.append(contnode)
    return ref
