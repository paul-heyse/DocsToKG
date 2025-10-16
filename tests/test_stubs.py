# === NAVMAP v1 ===
# {
#   "module": "tests.test_stubs",
#   "purpose": "Pytest coverage for stubs scenarios",
#   "sections": [
#     {
#       "id": "testpromotesimplenamespace",
#       "name": "TestPromoteSimpleNamespace",
#       "anchor": "class-testpromotesimplenamespace",
#       "kind": "class"
#     },
#     {
#       "id": "testdependencystubs",
#       "name": "TestDependencyStubs",
#       "anchor": "class-testdependencystubs",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Tests for DocParsing test stub utilities."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests._stubs import dependency_stubs, promote_simple_namespace_modules


class TestPromoteSimpleNamespace:
    """Test suite for :func:`promote_simple_namespace_modules`."""

    def teardown_method(self) -> None:  # Cleanup to avoid leakage
        for name in list(sys.modules):
            if name.startswith("_test_stub") or name.startswith("_test_hashable"):
                sys.modules.pop(name, None)

    def test_promotes_simple_namespace_to_module_type(self) -> None:
        """SimpleNamespace stubs should be promoted to ModuleType instances."""
        sys.modules["_test_stub_module"] = SimpleNamespace(test_attr="value")

        promote_simple_namespace_modules()

        assert isinstance(sys.modules["_test_stub_module"], ModuleType)
        assert sys.modules["_test_stub_module"].test_attr == "value"

    def test_preserves_attributes(self) -> None:
        """Promoted modules must preserve attributes from the stub."""
        sys.modules["_test_stub_attrs"] = SimpleNamespace(
            func=lambda x: x * 2,
            const=42,
            nested=SimpleNamespace(deep="value"),
        )

        promote_simple_namespace_modules()

        promoted = sys.modules["_test_stub_attrs"]
        assert promoted.func(5) == 10
        assert promoted.const == 42
        assert promoted.nested.deep == "value"

    def test_makes_hashable(self) -> None:
        """Promoted modules should be hashable to satisfy Hypothesis internals."""
        sys.modules["_test_hashable"] = SimpleNamespace()

        with pytest.raises(TypeError):
            hash(sys.modules["_test_hashable"])

        promote_simple_namespace_modules()

        hash(sys.modules["_test_hashable"])


class TestDependencyStubs:
    """Test suite for :func:`dependency_stubs`."""

    def teardown_method(self) -> None:
        for name in list(sys.modules):
            if name.startswith("_test_dep"):
                sys.modules.pop(name, None)

    def test_installs_stubs(self) -> None:
        """Installing a SimpleNamespace stub should populate ``sys.modules``."""
        dependency_stubs(_test_dep=SimpleNamespace(version="1.0"))

        assert "_test_dep" in sys.modules
        assert sys.modules["_test_dep"].version == "1.0"

    def test_converts_dict_to_namespace(self) -> None:
        """dict stubs should be converted to :class:`SimpleNamespace`."""
        dependency_stubs(_test_dep_dict={"key": "value"})

        assert hasattr(sys.modules["_test_dep_dict"], "key")
        assert sys.modules["_test_dep_dict"].key == "value"

    def test_callable_stub_invoked(self) -> None:
        """Callable stubs should be invoked to obtain the replacement object."""

        def factory() -> SimpleNamespace:
            return SimpleNamespace(created=True)

        dependency_stubs(_test_dep_callable=factory)

        assert sys.modules["_test_dep_callable"].created is True
