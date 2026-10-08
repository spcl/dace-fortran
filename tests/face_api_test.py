"""Static check that what dace-fortran takes from FaCe dace still exists there.

Every ``from dace... import name`` in the package must resolve, and so must every ``blas_nodes.<Node>`` /
``lapack_nodes.<Node>`` the library emitters reach for: FaCe drops library nodes and helpers, and the
frontend only finds out when a kernel that needs one is built.
"""

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "dace_fortran"

#: The emitters bind the node modules to these local names with ``importlib.import_module``.
NODE_MODULES = {"blas_nodes": "dace.libraries.blas.nodes", "lapack_nodes": "dace.libraries.lapack.nodes"}


def python_files() -> Iterator[Path]:
    return (path for path in sorted(PACKAGE.rglob("*.py")) if "build" not in path.relative_to(PACKAGE).parts)


def dace_imports() -> list[tuple[Path, int, str, str]]:
    """``(file, line, module, name)`` for every ``from dace[.x] import name`` in the package."""
    found = []
    for path in python_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and (node.module == "dace" or node.module.startswith("dace."))
            ):
                found.extend((path, node.lineno, node.module, alias.name) for alias in node.names)
    return found


def resolves(module: str, name: str) -> bool:
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return True
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return False
    return True


def test_every_dace_import_resolves():
    missing = []
    for path, line, module, name in dace_imports():
        try:
            ok = resolves(module, name)
        except ImportError as exc:
            missing.append(f"{path.relative_to(PACKAGE)}:{line}: {module} does not import ({exc})")
            continue
        if not ok:
            missing.append(f"{path.relative_to(PACKAGE)}:{line}: {module} has no {name}")
    assert not missing, "\n".join(missing)


def test_every_library_node_the_emitters_use_exists():
    tree = ast.parse((PACKAGE / "builder" / "emit_library.py").read_text())
    used = {
        (node.value.id, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in NODE_MODULES
    }
    assert used, "no library node is reached through blas_nodes / lapack_nodes any more"
    missing = sorted(
        f"{NODE_MODULES[local]}.{attr}"
        for local, attr in used
        if not hasattr(importlib.import_module(NODE_MODULES[local]), attr)
    )
    assert not missing, f"library nodes FaCe no longer has: {missing}"


def test_every_intrinsic_library_node_exists():
    from dace_fortran.intrinsics.linalg import LINALG, STANDARD

    missing = sorted(
        f"dace.libraries.{spec.module}.nodes.{spec.node_cls}"
        for spec in [*LINALG.values(), *STANDARD.values()]
        if not hasattr(importlib.import_module(f"dace.libraries.{spec.module}.nodes"), spec.node_cls)
    )
    assert not missing, f"library nodes FaCe no longer has: {missing}"


if __name__ == "__main__":
    test_every_dace_import_resolves()
    test_every_library_node_the_emitters_use_exists()
    test_every_intrinsic_library_node_exists()
