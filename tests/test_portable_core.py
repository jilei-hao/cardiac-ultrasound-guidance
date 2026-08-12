"""Guard the property that makes ``src/guidance`` droppable into vigil-system.

``src/guidance`` is vendored verbatim into vigil-system at
``src/cardiac/cardiac/guidance``. The copy is a plain ``cp -r`` with no edits, so
it only keeps working while the tree is genuinely self-contained: every import
must resolve either inside the tree (relatively) or to a third-party package
that both hosts install.

The failure this prevents is quiet. Adding ``from ui.image_window import ...``
to a renderer costs nothing here -- CUSG has ``src`` on its path, so it imports
fine and the app keeps working -- and then breaks only after the next sync, in
vigil-system, as an ImportError at node startup with nothing pointing back at
the commit that caused it. That is exactly the class of drift that produced the
hand-curated divergence this restructure exists to remove.

So: assert it here, where the mistake is made.
"""
import ast
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(_HERE, "..", "src", "guidance"))

# Top-level module names that belong to CUSG but NOT to the portable core.
# Importing any of them is what makes the tree un-droppable.
CUSG_ONLY = {
    "ui",              # desktop windows kept out of the core
    "inference",       # the legacy landmark model kept out of the core
    "data_access",     # Clarius/screen-capture sources: CUSG app only
    "common",
    "guidance_ui",
    "legacy_main",
    "probe_render",
    "record",
}

# The vendored DINOv3 checkout imports itself absolutely (``dinov3.layers`` etc).
# It ships inside the core at guidance/dinov3_repo and is put on sys.path by
# whoever loads the backbone, so it is self-contained despite looking absolute.
VENDORED_ABSOLUTE = {"dinov3"}


def _core_files():
    for root, dirs, files in os.walk(CORE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def _depth_from_core(path):
    """How many package levels ``path`` sits below the core root."""
    rel = os.path.relpath(path, CORE)
    return len(rel.split(os.sep)) - 1


@pytest.mark.parametrize("path", sorted(_core_files()), ids=lambda p: os.path.relpath(p, CORE))
def test_core_module_is_self_contained(path):
    """No import may reach outside src/guidance."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    inside_dinov3 = os.path.relpath(path, CORE).startswith("dinov3_repo")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in CUSG_ONLY, (
                    f"{os.path.relpath(path, CORE)} imports CUSG-only module "
                    f"'{alias.name}'. The core must not depend on the app; move "
                    f"the shared part into src/guidance or invert the dependency."
                )

        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                top = (node.module or "").split(".")[0]
                if inside_dinov3 and top in VENDORED_ABSOLUTE:
                    continue
                assert top not in CUSG_ONLY, (
                    f"{os.path.relpath(path, CORE)} imports CUSG-only module "
                    f"'{node.module}'. See the module docstring in this test."
                )
            else:
                # A relative import may not climb past the core root: `..` from
                # guidance/inference/ lands on guidance/ (fine), one more escapes.
                depth = _depth_from_core(path)
                assert node.level <= depth, (
                    f"{os.path.relpath(path, CORE)} has a relative import "
                    f"climbing {node.level} levels but sits only {depth} below "
                    f"src/guidance -- it escapes the portable core."
                )


@pytest.mark.parametrize("path", sorted(_core_files()), ids=lambda p: os.path.relpath(p, CORE))
def test_core_relative_imports_resolve(path):
    """Every relative import must point at a file that is actually in the tree.

    Reach and resolvability are different properties, and only checking reach is
    not enough: moving a helper OUT of the core leaves the importer with a
    perfectly well-formed `from .mesh_helper import ...` that simply has no
    target. CUSG surfaces that as a collection error in its own test run, but
    vigil-system would surface it as an ImportError at node startup after a sync
    -- far from the change that caused it. (This is not hypothetical; it is what
    happened while building this restructure.)
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    package_dir = os.path.dirname(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        base = package_dir
        for _ in range(node.level - 1):
            base = os.path.dirname(base)
        target = os.path.join(base, *node.module.split(".")) if node.module else base
        assert os.path.isdir(target) or os.path.isfile(target + ".py"), (
            f"{os.path.relpath(path, CORE)} imports "
            f"'{'.' * node.level}{node.module or ''}', which does not exist in "
            f"the core. Either it was moved out and should come back, or this "
            f"import is dead and the module belongs outside src/guidance."
        )


def test_core_has_no_desktop_dependencies():
    """The core must not pull a GUI toolkit in.

    vigil-system runs these modules headless inside ROS nodes. PySide6/PyQt6 are
    CUSG-app dependencies; importing one here would make the vendored copy fail
    to import on a box that never installed Qt bindings.
    """
    offenders = []
    for path in _core_files():
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                if name.split(".")[0] in {"PySide6", "PyQt6", "PyQt5", "tkinter"}:
                    offenders.append(f"{os.path.relpath(path, CORE)}: {name}")
    assert not offenders, "GUI toolkit imported inside the portable core:\n  " + \
        "\n  ".join(offenders)
