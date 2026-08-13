"""
RED-phase contract tests for the eight approved hello-gpu notebooks.

These tests validate structural and content contracts for operational notebooks.
They MUST fail when notebook files do not yet exist, with no import/syntax errors.

Approved notebook paths (flat layout, no chapter0/README/index):
  notebooks/part0-intro/chapter1.ipynb
  notebooks/part0-intro/chapter2.ipynb
  notebooks/part0-intro/chapter3.ipynb
  notebooks/part0-intro/chapter4.ipynb
  notebooks/part1-profiling/chapter5.ipynb
  notebooks/part1-profiling/chapter6.ipynb
  notebooks/part1-profiling/chapter7.ipynb
  notebooks/part2-kernels/chapter8.ipynb
"""

import json
import os
import pathlib
import re
import subprocess
import unittest

# Resolve repo root: code/part0-intro/tests/ -> 3 levels up
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

APPROVED_NOTEBOOKS = [
    "notebooks/part0-intro/chapter1.ipynb",
    "notebooks/part0-intro/chapter2.ipynb",
    "notebooks/part0-intro/chapter3.ipynb",
    "notebooks/part0-intro/chapter4.ipynb",
    "notebooks/part1-profiling/chapter5.ipynb",
    "notebooks/part1-profiling/chapter6.ipynb",
    "notebooks/part1-profiling/chapter7.ipynb",
    "notebooks/part2-kernels/chapter8.ipynb",
]

REJECTED_PATHS = [
    "notebooks/part0-intro/chapter0.ipynb",
    "notebooks/README.md",
    "notebooks/index.ipynb",
]

# Docs cross-reference: each chapter must mention its docs index.md
DOCS_CROSS_REF = {
    1: "docs/part0-intro/chapter1/index.md",
    2: "docs/part0-intro/chapter2/index.md",
    3: "docs/part0-intro/chapter3/index.md",
    4: "docs/part0-intro/chapter4/index.md",
    5: "docs/part1-profiling/chapter5/index.md",
    6: "docs/part1-profiling/chapter6/index.md",
    7: "docs/part1-profiling/chapter7/index.md",
    8: "docs/part2-kernels/chapter8/index.md",
}

# Code cross-reference: each chapter must mention its code directory
CODE_CROSS_REF = {
    1: "code/part0-intro/chapter1/",
    2: "code/part0-intro/chapter2/",
    3: "code/part0-intro/chapter3/",
    4: "code/part0-intro/chapter4/",
    5: "code/part1-profiling/chapter5/",
    6: "code/part1-profiling/chapter6/",
    7: "code/part1-profiling/chapter7/",
    8: "code/part2-kernels/chapter8/",
}

# Chapter-specific operational tokens that MUST appear in notebook source
CHAPTER_TOKENS = {
    1: ["rocminfo", "check_torch_rocm", "vector_add", "status PASS"],
    2: ["N=10", "block=4", "thread mapping", "branch_divergence"],
    3: [
        "global_memory_access",
        "LDS",
        "WMMA",
        "rdna3_wmma",
        "rdna4_wmma",
        "three offload arches",
    ],
    4: [
        "benchmark_vector_add",
        "warmup",
        "repeat",
        "arithmetic intensity",
        "effective bandwidth",
    ],
    5: [
        "bench_ch4",
        "bench_torch_op",
        "bench_triton_copy",
        "min",
        "median",
        "mean",
        "p95",
        "std",
    ],
    6: [
        "coalesced",
        "linecross",
        "stride",
        "rocprofv3",
        "reference fallback",
    ],
    7: [
        "plot_roofline_ch6",
        "arithmetic intensity",
        "memory ceiling",
        "compute ceiling",
        "source attribution",
    ],
    8: [
        "HIP",
        "Triton",
        "N=31",
        "N=32",
        "N=33",
        "N=1027",
        "BLOCK_SIZE",
        "offsets",
        "mask",
        "inline visualization",
        "correctness before performance",
    ],
}

# Forbidden content patterns
FORBIDDEN_PATTERNS = [
    # Mojibake indicators
    (r"[\ufffd\u00e3\u00c3][\u0080-\u00bf]", "mojibake detected"),
    # HSA_OVERRIDE_GFX_VERSION must never appear
    (r"HSA_OVERRIDE_GFX_VERSION", "forbidden HSA_OVERRIDE_GFX_VERSION"),
    # Claims that gfx1151 is unsupported or unstable
    (
        r"gfx1151.{0,40}(unsupported|unstable|not supported|experimental)",
        "forbidden claim that gfx1151 is unsupported/unstable",
    ),
    # Live localhost or triton-viz service startup
    (r"localhost:\d+", "forbidden live localhost service reference"),
    (r"triton[-_]viz.*serv", "forbidden triton-viz service startup"),
    # %%writefile duplication
    (r"%%writefile", "forbidden %%writefile magic (use code/ references instead)"),
]

# Semantic flow headings/content keywords required in every notebook
SEMANTIC_FLOW_KEYWORDS = [
    "goal",
    "prerequisite",
    "platform",
    "parameter",
    "execution",
    "expected",
    "pass criteria",
]


def _load_notebook(rel_path):
    """Load notebook JSON from repo-root-relative path."""
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Notebook does not exist: {rel_path} "
            f"(resolved: {full_path})"
        )
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_all_source(nb):
    """Concatenate all cell sources into one string."""
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.extend(src)
        else:
            parts.append(src)
    return "".join(parts)


def _get_cells_by_type(nb, cell_type):
    """Return cells of a given type."""
    return [c for c in nb.get("cells", []) if c.get("cell_type") == cell_type]


def _chapter_number(rel_path):
    """Extract chapter number from path like notebooks/partX/chapterN.ipynb."""
    m = re.search(r"chapter(\d+)\.ipynb$", rel_path)
    return int(m.group(1)) if m else None


class TestNotebookExistence(unittest.TestCase):
    """Verify exactly the approved notebooks exist and rejected ones don't."""

    def test_approved_notebooks_exist(self):
        """All eight approved notebooks must exist as files."""
        missing = []
        for nb_path in APPROVED_NOTEBOOKS:
            full = REPO_ROOT / nb_path
            if not full.exists():
                missing.append(nb_path)
        self.assertEqual(
            missing,
            [],
            f"Missing approved notebooks: {missing}",
        )

    def test_rejected_paths_do_not_exist(self):
        """Rejected paths must NOT exist."""
        for rejected in REJECTED_PATHS:
            full = REPO_ROOT / rejected
            self.assertFalse(
                full.exists(),
                f"Rejected path must not exist: {rejected}",
            )

    def test_no_nested_chapter_directories(self):
        """Notebooks must be flat .ipynb files, not chapter/ directories."""
        notebooks_dir = REPO_ROOT / "notebooks"
        if not notebooks_dir.exists():
            self.skipTest("notebooks/ directory does not exist yet")
        for part_dir in notebooks_dir.iterdir():
            if not part_dir.is_dir():
                continue
            for item in part_dir.iterdir():
                if item.is_dir() and item.name.startswith("chapter"):
                    self.fail(
                        f"Nested chapter directory forbidden: {item.relative_to(REPO_ROOT)}"
                    )

    def test_approved_notebooks_are_not_gitignored(self):
        ignored = []
        for nb_path in APPROVED_NOTEBOOKS:
            result = subprocess.run(
                ["git", "check-ignore", "-q", nb_path],
                cwd=REPO_ROOT,
                check=False,
            )
            if result.returncode == 0:
                ignored.append(nb_path)
        self.assertEqual(
            ignored,
            [],
            f"Approved notebooks are still ignored by Git: {ignored}",
        )


class TestNotebookFormat(unittest.TestCase):
    """Validate nbformat 4, Python 3 kernelspec, cell structure."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_nbformat_version_4(self):
        """All notebooks must use nbformat 4."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                self.assertEqual(
                    nb.get("nbformat"),
                    4,
                    f"{path}: nbformat must be 4",
                )

    def test_python3_kernelspec(self):
        """All notebooks must declare Python 3 kernel."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                ks = nb.get("metadata", {}).get("kernelspec", {})
                lang = ks.get("language", "")
                self.assertEqual(
                    lang.lower(),
                    "python",
                    f"{path}: kernelspec language must be python",
                )
                # Name should indicate python3
                name = ks.get("name", "")
                self.assertIn(
                    "python3",
                    name.lower().replace("python 3", "python3"),
                    f"{path}: kernelspec name must indicate python3",
                )

    def test_non_empty_markdown_cells(self):
        """Every notebook must have at least one non-empty markdown cell."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                md_cells = _get_cells_by_type(nb, "markdown")
                non_empty = [
                    c
                    for c in md_cells
                    if "".join(c.get("source", [])).strip()
                ]
                self.assertGreater(
                    len(non_empty),
                    0,
                    f"{path}: must have non-empty markdown cells",
                )

    def test_non_empty_code_cells(self):
        """Every notebook must have at least one non-empty code cell."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                code_cells = _get_cells_by_type(nb, "code")
                non_empty = [
                    c
                    for c in code_cells
                    if "".join(c.get("source", [])).strip()
                ]
                self.assertGreater(
                    len(non_empty),
                    0,
                    f"{path}: must have non-empty code cells",
                )

    def test_no_stored_outputs(self):
        """Code cells must not have stored outputs."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                for i, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    outputs = cell.get("outputs", [])
                    self.assertEqual(
                        outputs,
                        [],
                        f"{path} cell[{i}]: must not store outputs",
                    )

    def test_no_execution_count(self):
        """Code cells must have null execution_count."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                for i, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    ec = cell.get("execution_count")
                    self.assertIsNone(
                        ec,
                        f"{path} cell[{i}]: execution_count must be null",
                    )


class TestCrossReferences(unittest.TestCase):
    """Each notebook must reference its corresponding docs and code paths."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_docs_cross_reference(self):
        """Each notebook mentions its docs/.../index.md."""
        for path, nb in self.notebooks.items():
            ch = _chapter_number(path)
            with self.subTest(notebook=path, chapter=ch):
                source = _get_all_source(nb)
                expected_ref = DOCS_CROSS_REF[ch]
                self.assertIn(
                    expected_ref,
                    source,
                    f"{path}: must reference {expected_ref}",
                )

    def test_code_cross_reference(self):
        """Each notebook mentions its code/.../chapterN/ directory."""
        for path, nb in self.notebooks.items():
            ch = _chapter_number(path)
            with self.subTest(notebook=path, chapter=ch):
                source = _get_all_source(nb)
                expected_ref = CODE_CROSS_REF[ch]
                self.assertIn(
                    expected_ref,
                    source,
                    f"{path}: must reference {expected_ref}",
                )

    def test_relative_paths_from_repo_root(self):
        """Paths in notebooks must be repo-root-relative, not cwd-relative."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                source = _get_all_source(nb)
                # Should not assume cwd is notebook dir with ../
                # Code/docs refs should start from repo root
                self.assertNotRegex(
                    source,
                    r"\.\./\.\./code/",
                    f"{path}: must use repo-root-relative paths, not ../../",
                )


class TestSemanticFlow(unittest.TestCase):
    """Notebooks must follow goal→prerequisite→platform→parameters→execution→expected→pass criteria flow."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_semantic_flow_present(self):
        """Each notebook contains all semantic flow keywords in headings/content."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                source = _get_all_source(nb).lower()
                missing = [
                    kw
                    for kw in SEMANTIC_FLOW_KEYWORDS
                    if kw.lower() not in source
                ]
                self.assertEqual(
                    missing,
                    [],
                    f"{path}: missing semantic flow keywords: {missing}",
                )


class TestPlatformGuidance(unittest.TestCase):
    """gfx1201 is narrative baseline; gfx1100/gfx1151 included for platform commands where GPU compilation applies."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_gfx1201_baseline(self):
        """Every notebook must mention gfx1201 as narrative baseline."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                source = _get_all_source(nb)
                self.assertIn(
                    "gfx1201",
                    source,
                    f"{path}: must mention gfx1201 as narrative baseline",
                )

    def test_multi_arch_where_compilation_applies(self):
        """Chapters with GPU compilation must include gfx1100, gfx1151, gfx1201."""
        # Chapters 3, 4, 5, 6, 7, 8 involve GPU compilation
        compilation_chapters = [3, 4, 5, 6, 7, 8]
        for path, nb in self.notebooks.items():
            ch = _chapter_number(path)
            if ch not in compilation_chapters:
                continue
            with self.subTest(notebook=path, chapter=ch):
                source = _get_all_source(nb)
                for arch in ["gfx1100", "gfx1151", "gfx1201"]:
                    self.assertIn(
                        arch,
                        source,
                        f"{path}: must include {arch} in platform command guidance",
                    )


class TestForbiddenContent(unittest.TestCase):
    """Notebooks must not contain forbidden patterns."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_no_forbidden_patterns(self):
        """No mojibake, HSA_OVERRIDE_GFX_VERSION, gfx1151-unstable claims, localhost services, %%writefile."""
        for path, nb in self.notebooks.items():
            source = _get_all_source(nb)
            for pattern, description in FORBIDDEN_PATTERNS:
                with self.subTest(notebook=path, check=description):
                    self.assertIsNone(
                        re.search(pattern, source, re.IGNORECASE),
                        f"{path}: {description}",
                    )


class TestChapterTokens(unittest.TestCase):
    """Each chapter must contain its specific operational tokens."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_chapter_specific_tokens(self):
        """Each notebook contains all required operational tokens."""
        for path, nb in self.notebooks.items():
            ch = _chapter_number(path)
            with self.subTest(notebook=path, chapter=ch):
                source = _get_all_source(nb)
                tokens = CHAPTER_TOKENS[ch]
                missing = [t for t in tokens if t not in source]
                self.assertEqual(
                    missing,
                    [],
                    f"{path}: missing operational tokens: {missing}",
                )


class TestCodeEmbeddingPolicy(unittest.TestCase):
    """C/HIP code must be referenced/compiled in place, not embedded as long cells.
    Short Python/Triton snippets are allowed inline."""

    def setUp(self):
        self.notebooks = {}
        for path in APPROVED_NOTEBOOKS:
            self.notebooks[path] = _load_notebook(path)

    def test_no_long_hip_code_cells(self):
        """Code cells must not contain long C/HIP code (>20 lines of C-like code)."""
        # Heuristic: if a code cell has many lines with C indicators (includes, {}, ;)
        c_indicators = re.compile(
            r"(#include|__global__|__device__|hipLaunchKernelGGL|int main\()"
        )
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                for i, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    src = "".join(cell.get("source", []))
                    lines = src.strip().split("\n")
                    c_lines = [l for l in lines if c_indicators.search(l)]
                    # Allow references/short snippets, forbid >20 C-like lines
                    if len(c_lines) > 20:
                        self.fail(
                            f"{path} cell[{i}]: contains {len(c_lines)} C/HIP lines. "
                            f"C/HIP code must be referenced from code/ directory, "
                            f"not embedded as long cells."
                        )

    def test_no_writefile_duplication(self):
        """%%writefile must not be used (covered by forbidden patterns, double-check here)."""
        for path, nb in self.notebooks.items():
            with self.subTest(notebook=path):
                source = _get_all_source(nb)
                self.assertNotIn(
                    "%%writefile",
                    source,
                    f"{path}: must not use %%writefile (use code/ references)",
                )


class TestChapter8Specifics(unittest.TestCase):
    """Chapter 8 specific: HIP+Triton, boundary testing, no live service."""

    def setUp(self):
        path = "notebooks/part2-kernels/chapter8.ipynb"
        self.nb = _load_notebook(path)
        self.path = path

    def test_hip_and_triton_implementations(self):
        """Chapter 8 must reference both HIP and Triton implementations."""
        source = _get_all_source(self.nb)
        self.assertIn("HIP", source, f"{self.path}: must reference HIP implementation")
        self.assertIn(
            "Triton", source, f"{self.path}: must reference Triton implementation"
        )

    def test_boundary_values(self):
        """Chapter 8 must test N=31,32,33,1027 boundary values."""
        source = _get_all_source(self.nb)
        for val in ["N=31", "N=32", "N=33", "N=1027"]:
            self.assertIn(val, source, f"{self.path}: must test boundary {val}")

    def test_block_size_and_masking(self):
        """Chapter 8 must cover BLOCK_SIZE, offsets, mask, inline visualization."""
        source = _get_all_source(self.nb)
        for token in ["BLOCK_SIZE", "offsets", "mask", "inline visualization"]:
            self.assertIn(token, source, f"{self.path}: must cover {token}")

    def test_correctness_before_performance(self):
        """Chapter 8 must emphasize correctness before performance."""
        source = _get_all_source(self.nb)
        self.assertIn(
            "correctness before performance",
            source,
            f"{self.path}: must state correctness before performance principle",
        )

    def test_no_live_service(self):
        """Chapter 8 must not start live localhost/triton-viz services."""
        source = _get_all_source(self.nb)
        self.assertIsNone(
            re.search(r"localhost:\d+", source),
            f"{self.path}: must not reference live localhost services",
        )
        self.assertIsNone(
            re.search(r"triton[-_]viz.*serv", source, re.IGNORECASE),
            f"{self.path}: must not start triton-viz service",
        )


if __name__ == "__main__":
    unittest.main()
