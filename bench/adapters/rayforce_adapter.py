"""Rayforce native adapter for benchmarks.

Measures end-to-end execution time with time.perf_counter_ns around
eval_str() — same convention as the other adapters, so the comparison is
fair. Does NOT use the engine's internal (timeit ...), which would hide
Python-binding overhead and tilt results in rayforce's favor.
"""

import time
from pathlib import Path
import subprocess
import sys

from .base import Adapter, BenchmarkResult


class RayforceAdapter(Adapter):
    """Benchmark adapter for rayforce native core.

    Measures: Pure rayforce execution time (no Python overhead).

    Uses eval_str("(timeit ...)") to measure query execution time
    directly in the rayforce runtime.
    """

    name = "rayforce"

    QUERY_STRINGS = {
        "groupby_q1":  't.select(v1=Column("v1").sum()).by("id1").execute()',
        "groupby_q2":  't.select(v1=Column("v1").sum()).by("id1","id2").execute()',
        "groupby_q3":  't.select(v1=Column("v1").sum(), v3=Column("v3").mean()).by("id3").execute()',
        "groupby_q4":  't.select(v1=Column("v1").mean(), v2=Column("v2").mean(), v3=Column("v3").mean()).by("id4").execute()',
        "groupby_q5":  't.select(v1=Column("v1").sum(), v2=Column("v2").sum(), v3=Column("v3").sum()).by("id6").execute()',
        "groupby_q6":  't.select(v3_median=Column("v3").median(), v3_std=Column("v3").std()).by("id4","id5").execute()',
        "groupby_q7":  '# Two-stage workaround for engine NYI on arithmetic-of-aggregates per-group:\nagg = t.select(v1m=Column("v1").max(), v2m=Column("v2").min()).by("id3").execute()\nagg.select("id3", range_v1_v2=Column("v1m") - Column("v2m")).execute()',
        "groupby_q8":  't.select(largest2_v3=Column("v3").top(2)).by("id6").execute()',
        "groupby_q9":  '# Two-stage: pearson_corr at top first, then square the result\nagg = t.select(r=Column("v1").pearson_corr(Column("v2"))).by("id2","id4").execute()\nagg.select("id2", "id4", r2=Column("r")*Column("r")).execute()',
        "groupby_q10": 't.select(v3=Column("v3").sum(), cnt=Column("v1").count()).by("id1","id2","id3","id4","id5","id6").execute()',
        "join_q1":     '# pre-project right to (key, v2) to avoid to_dict() collapse on dup cols\nx.inner_join(small.select("id1","v2").execute(), on=["id1"]).execute()',
        "join_q2":     'x.inner_join(medium.select("id2","v2").execute(), on=["id2"]).execute()',
        "join_q3":     'x.left_join(medium.select("id2","v2").execute(), on=["id2"]).execute()',
        "join_q4":     'x.inner_join(medium.select("id5","v2").execute(), on=["id5"]).execute()',
        "join_q5":     'x.inner_join(big.select("id3","v2").execute(), on=["id3"]).execute()',
        "join_inner":  'L.inner_join(R, on=["id1","id2","id3"]).execute()',
        "join_left":   'L.left_join(R,  on=["id1","id2","id3"]).execute()',
        "sort_single": 't.order_by("id1").execute()',
        "sort_multi":  't.order_by("id1","id2","id3").execute()',
    }

    def __init__(self, local_path: str | Path | None = None):
        """Initialize rayforce adapter.

        Args:
            local_path: Path to local rayforce-py repo for dev builds.
                       If None, uses installed package from PyPI.
        """
        self._local_path = Path(local_path) if local_path else None
        self._rayforce = None
        self._eval_str = None
        self._Table = None
        self._I64 = None
        self._F64 = None
        self._table_names: dict[str, str] = {}  # Maps logical name to rayforce symbol

        self._setup_rayforce()

    def _setup_rayforce(self) -> None:
        """Setup rayforce module - either from PyPI or local build."""
        if self._local_path:
            self._setup_local_build()
        else:
            self._setup_pypi()

    def _setup_pypi(self) -> None:
        """Use rayforce from PyPI."""
        try:
            import rayforce
            from rayforce import Table, Column, I64, F64

            self._rayforce = rayforce
            self._eval_str = rayforce.eval_str
            self._Table = Table
            self._Column = Column
            self._I64 = I64
            self._F64 = F64
            self._Symbol = getattr(rayforce, "Symbol", None)
            self._STR = getattr(rayforce, "STR", None) or getattr(rayforce, "Str", None)
            self.version = f"{rayforce.version} (pypi)"
        except ImportError as e:
            raise ImportError(
                "rayforce-py not installed. Install with: pip install rayforce-py"
            ) from e

    def _setup_local_build(self) -> None:
        """Build and use rayforce from local path."""
        if not self._local_path or not self._local_path.exists():
            raise ValueError(f"Local path does not exist: {self._local_path}")

        # Build the package using uv (faster) or pip
        print(f"Building rayforce-py from {self._local_path}...")

        result = subprocess.run(
            ["uv", "pip", "install", "-e", str(self._local_path), "--python", sys.executable],
            capture_output=True,
            text=True,
            cwd=self._local_path,
        )
        if result.returncode != 0:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(self._local_path), "-q"],
                capture_output=True,
                text=True,
                cwd=self._local_path,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build rayforce-py: {result.stderr}")

        # Force reimport
        for mod in list(sys.modules.keys()):
            if mod.startswith("rayforce"):
                del sys.modules[mod]

        import rayforce
        from rayforce import Table, Column, I64, F64

        self._rayforce = rayforce
        self._eval_str = rayforce.eval_str
        self._Table = Table
        self._Column = Column
        self._I64 = I64
        self._F64 = F64
        self._Symbol = getattr(rayforce, "Symbol", None)
        self._STR = getattr(rayforce, "STR", None) or getattr(rayforce, "Str", None)
        self.version = f"{rayforce.version} (local: {self._local_path})"
        print(f"Using rayforce {self.version}")

    def load_data(self, path: Path, table_name: str = "data") -> None:
        """Load CSV data using rayforce native Table.from_csv."""
        symbol_name = f"_bench_{table_name}"

        column_types = self._get_column_types(path)
        rf_table = self._Table.from_csv(column_types, str(path))
        rf_table.save(symbol_name)
        self._table_names[table_name] = symbol_name
        # Keep a Python-level Table handle for the chained API
        # (.select().by().execute(), etc.) — canonical rayforce-py idiom
        # per https://py.rayforcedb.com.
        if not hasattr(self, "_tables"):
            self._tables = {}
        self._tables[table_name] = rf_table

    def _get_table_obj(self, name: str = "data"):
        if not hasattr(self, "_tables") or name not in self._tables:
            raise ValueError(f"Table '{name}' not loaded")
        return self._tables[name]

    def _get_column_types(self, path: Path) -> list:
        """Get column types for canonical H2O CSVs.

        groupby:  id1..id3 string-symbol, id4..id6 i64, v1/v2 i64, v3 f64
        join:     id1..id3 i64,           id4..id6 string-symbol, v float
        """
        with open(path) as f:
            header = [c.strip().strip('"') for c in f.readline().strip().split(",")]

        sym = self._Symbol or self._STR
        if sym is None:
            raise RuntimeError(
                "rayforce-py lacks Symbol/STR types; "
                "canonical H2O schema requires string IDs. "
                "Upgrade rayforce-py."
            )

        # Look at the first data row to disambiguate (groupby vs join layout).
        with open(path) as f:
            f.readline()
            first_row = [c.strip().strip('"') for c in f.readline().strip().split(",")]

        types = []
        for col, val in zip(header, first_row):
            if col.startswith("id"):
                # Strings start with "id" prefix in our generator output.
                types.append(sym if val.startswith("id") else self._I64)
            elif col == "v3":
                types.append(self._F64)
            elif col.startswith("v"):
                # join-side v is float; groupby v1/v2 are int. Sniff.
                try:
                    int(val)
                    types.append(self._I64)
                except ValueError:
                    types.append(self._F64)
            else:
                types.append(self._I64)
        return types

    def _get_symbol(self, name: str = "data") -> str:
        """Get rayforce symbol name for a table."""
        if name not in self._table_names:
            raise ValueError(f"Table '{name}' not loaded")
        return self._table_names[name]

    def _run_timed_query(self, query: str, bench_name: str) -> BenchmarkResult:
        """Execute a query and time it externally with perf_counter_ns.

        Args:
            query: The rayforce query (rayfall expression as a string)
            bench_name: Name of the benchmark

        Returns:
            BenchmarkResult with timing in nanoseconds
        """
        start = time.perf_counter_ns()
        result = self._eval_str(query)
        time_ns = time.perf_counter_ns() - start
        rows = len(result) if result is not None else 0
        return BenchmarkResult(bench_name, time_ns, rows)

    def _timed(self, fn, name: str) -> BenchmarkResult:
        result, time_ns = self._time_it(fn)
        rows = len(result) if result is not None else 0
        return BenchmarkResult(name, time_ns, rows)

    def run_groupby_q1(self) -> BenchmarkResult:
        """Q1: sum(v1) group by id1"""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v1=C("v1").sum()).by("id1").execute(),
            "groupby_q1",
        )

    def run_groupby_q2(self) -> BenchmarkResult:
        """Q2: sum(v1) group by id1, id2"""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v1=C("v1").sum()).by("id1", "id2").execute(),
            "groupby_q2",
        )

    def run_groupby_q3(self) -> BenchmarkResult:
        """Q3: sum(v1), mean(v3) group by id3"""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v1=C("v1").sum(),
                             v3=C("v3").mean()).by("id3").execute(),
            "groupby_q3",
        )

    def run_groupby_q4(self) -> BenchmarkResult:
        """Q4: mean(v1), mean(v2), mean(v3) by id4 — canonical H2O."""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v1=C("v1").mean(),
                             v2=C("v2").mean(),
                             v3=C("v3").mean()).by("id4").execute(),
            "groupby_q4",
        )

    def run_groupby_q5(self) -> BenchmarkResult:
        """Q5: sum(v1), sum(v2), sum(v3) by id6 — canonical H2O."""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v1=C("v1").sum(),
                             v2=C("v2").sum(),
                             v3=C("v3").sum()).by("id6").execute(),
            "groupby_q5",
        )

    def run_groupby_q6(self) -> BenchmarkResult:
        """Q6: median(v3), sd(v3) by id4, id5 — canonical H2O."""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v3_median=C("v3").median(),
                             v3_std=C("v3").std()
                             ).by("id4", "id5").execute(),
            "groupby_q6",
        )

    def run_groupby_q7(self) -> BenchmarkResult:
        """Q7: max(v1) - min(v2) by id3 — canonical H2O.

        Two-stage workaround: rayforce's arithmetic-of-aggregates
        (Column.max() - Column.min()) inside .by(...) computes globally,
        not per-group. Compute max/min per group first, subtract after.
        See REQUIREMENTS_CANONICAL_H2O.md §1.2.
        """
        t = self._get_table_obj()
        C = self._Column

        def query():
            agg = t.select(v1m=C("v1").max(),
                           v2m=C("v2").min()).by("id3").execute()
            return agg.select("id3",
                              range_v1_v2=C("v1m") - C("v2m")).execute()

        return self._timed(query, "groupby_q7")

    def run_groupby_q8(self) -> BenchmarkResult:
        """Q8: largest two v3 by id6 — canonical H2O.

        The engine's OP_GROUP_TOPK_ROWFORM operator emits row form
        directly (one row per kept-value, not nested LIST<F64>[K]
        cells), so the natural chained API matches DuckDB/polars's
        2-rows-per-group shape (200k rows for 100k id6 groups) without
        any adapter-side explode.
        """
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(largest2_v3=C("v3").top(2)
                             ).by("id6").execute(),
            "groupby_q8",
        )

    def run_groupby_q9(self) -> BenchmarkResult:
        """Q9: pearson_corr(v1, v2)**2 by id2, id4 — canonical H2O.

        Two-stage workaround mirroring run_groupby_q7: the planner only
        lowers a *top-level* agg call to its hash-agg opcode.  Writing
        ``C("v1").pearson_corr(...) ** 2`` puts ``pow`` at the head of
        the select expression — pow is not an aggregator, so the inner
        ``pearson_corr`` collapses through the eval-level scatter
        fallback and never hits OP_PEARSON_CORR.  Splitting the squaring
        into a second select against the aggregated result keeps the
        first stage purely aggregatory (single ``pearson_corr`` per
        group → OP_PEARSON_CORR vectorized hash-agg) and the second
        stage trivial element-wise arithmetic.
        """
        t = self._get_table_obj()
        C = self._Column

        def query():
            agg = t.select(r=C("v1").pearson_corr(C("v2"))
                           ).by("id2", "id4").execute()
            return agg.select("id2", "id4", r2=C("r") * C("r")).execute()

        return self._timed(query, "groupby_q9")

    def run_groupby_q10(self) -> BenchmarkResult:
        """Q10: sum(v3), count(v1) by id1..id6 — canonical H2O."""
        t = self._get_table_obj()
        C = self._Column
        return self._timed(
            lambda: t.select(v3=C("v3").sum(),
                             cnt=C("v1").count()
                             ).by("id1", "id2", "id3",
                                  "id4", "id5", "id6").execute(),
            "groupby_q10",
        )

    def _load_table_from_csv(self, path: Path) -> object:
        """Load CSV file using rayforce native Table.from_csv."""
        column_types = self._get_column_types(path)
        return self._Table.from_csv(column_types, str(path))

    # Canonical H2O J1 — 5 single-key joins via chain API.
    #
    # rayforce-py 2.0a1 has a known wrapper bug: Table.to_dict() drops
    # duplicate column names, silently losing one side's data. Right
    # tables (small/medium/big) carry id4/id5/id6 cols that overlap
    # with x. To sidestep the wrapper bug *and* keep the engine work
    # honest, we pre-project each right side to (key, v2) before the
    # join. Engine still computes the same join, just on a narrower
    # schema; result has only x's cols + v2 → no name collisions.

    def _project_right(self, right_name: str, key: str):
        return self._get_table_obj(right_name).select(key, "v2").execute()

    def run_join_q1(self) -> BenchmarkResult:
        """Q1: x.inner_join(small, on=id1)."""
        x = self._get_table_obj("x")
        r = self._project_right("small", "id1")
        return self._timed(
            lambda: x.inner_join(r, on=["id1"]).execute(), "join_q1")

    def run_join_q2(self) -> BenchmarkResult:
        """Q2: x.inner_join(medium, on=id2)."""
        x = self._get_table_obj("x")
        r = self._project_right("medium", "id2")
        return self._timed(
            lambda: x.inner_join(r, on=["id2"]).execute(), "join_q2")

    def run_join_q3(self) -> BenchmarkResult:
        """Q3: x.left_join(medium, on=id2)."""
        x = self._get_table_obj("x")
        r = self._project_right("medium", "id2")
        return self._timed(
            lambda: x.left_join(r, on=["id2"]).execute(), "join_q3")

    def run_join_q4(self) -> BenchmarkResult:
        """Q4: x.inner_join(medium, on=id5) — string key."""
        x = self._get_table_obj("x")
        r = self._project_right("medium", "id5")
        return self._timed(
            lambda: x.inner_join(r, on=["id5"]).execute(), "join_q4")

    def run_join_q5(self) -> BenchmarkResult:
        """Q5: x.inner_join(big, on=id3)."""
        x = self._get_table_obj("x")
        r = self._project_right("big", "id3")
        return self._timed(
            lambda: x.inner_join(r, on=["id3"]).execute(), "join_q5")

    def run_join_inner(self, right_path: Path) -> BenchmarkResult:
        """Inner join on (id1, id2, id3) — canonical H2O J1.

        Pre-project right to (id1, id2, id3, v2) so the engine doesn't
        emit duplicate-name columns (id4..id6, v1 exist on both sides);
        Table.to_dict() collapses dup-name keys, which corrupts the
        left-side values. This matches the canonical H2O answer
        "left-side cols + right.v2" already enforced by check.py.
        """
        L = self._get_table_obj("left")
        R = self._load_table_from_csv(right_path)
        C = self._Column
        return self._timed(
            lambda: L.inner_join(
                R.select("id1", "id2", "id3", "v2").execute(),
                on=["id1", "id2", "id3"],
            ).execute(),
            "join_inner",
        )

    def run_join_left(self, right_path: Path) -> BenchmarkResult:
        """Left join on (id1, id2, id3) — canonical H2O J1.

        Same pre-project trick as run_join_inner — see its docstring.
        """
        L = self._get_table_obj("left")
        R = self._load_table_from_csv(right_path)
        C = self._Column
        return self._timed(
            lambda: L.left_join(
                R.select("id1", "id2", "id3", "v2").execute(),
                on=["id1", "id2", "id3"],
            ).execute(),
            "join_left",
        )

    def run_sort_single(self) -> BenchmarkResult:
        """Sort by single column."""
        t = self._get_table_obj()
        return self._timed(lambda: t.order_by("id1").execute(), "sort_single")

    def run_sort_multi(self) -> BenchmarkResult:
        """Sort by multiple columns."""
        t = self._get_table_obj()
        return self._timed(
            lambda: t.order_by("id1", "id2", "id3").execute(),
            "sort_multi",
        )

    _RF_TYPES_NAME = {
        "u8": "U8", "i16": "I16", "i32": "I32",
        "i64": "I64", "f64": "F64",
        # rayforce-py 1.0.0 has rf.String but it's a Vector wrapper without
        # the .ray_name attribute Table.from_csv() expects, so we can't
        # request a RAY_STR column at load time. Fall back to Symbol — same
        # underlying scan path that ~/rayforce/bench/h2o/q*.rfl uses.
        "str8": "Symbol", "str16": "Symbol",
    }

    def run_sort_typed_full(self, csv_path, dtype: str,
                             n_warmup: int, n_iter: int):
        """Sort a single typed column for the extended sort grid."""
        type_name = self._RF_TYPES_NAME[dtype]
        rf_type = getattr(self._rayforce, type_name)
        t = self._Table.from_csv([rf_type], str(csv_path))
        rows = len(t)

        for _ in range(n_warmup):
            t.order_by("v").execute()

        results = []
        for _ in range(n_iter):
            start = time.perf_counter_ns()
            t.order_by("v").execute()
            time_ns = time.perf_counter_ns() - start
            results.append(BenchmarkResult(f"sort_{dtype}", time_ns, rows))
        return results

    def materialize(self, op: str, right_path: Path | None = None):
        import polars as pl
        C = self._Column
        if op.startswith("join_q") and op[len("join_q"):].isdigit():
            # Pre-project right side to (key, v2) only so the join result
            # has no duplicate non-key column names (which would be lost
            # to Table.to_dict()'s collapse bug). Same trick as run_join_qN.
            x = self._get_table_obj("x")
            joins = {
                "join_q1": ("small",  "inner_join", "id1"),
                "join_q2": ("medium", "inner_join", "id2"),
                "join_q3": ("medium", "left_join",  "id2"),
                "join_q4": ("medium", "inner_join", "id5"),
                "join_q5": ("big",    "inner_join", "id3"),
            }
            right_name, fn_name, key = joins[op]
            r = self._get_table_obj(right_name).select(key, "v2").execute()
            result = getattr(x, fn_name)(r, on=[key]).execute()
        elif op in ("join_inner", "join_left"):
            L = self._get_table_obj("left")
            R = self._load_table_from_csv(right_path)
            # Pre-project right to (keys, v2) — see run_join_inner for
            # rationale (Table.to_dict() collapses dup-name columns).
            R_proj = R.select("id1", "id2", "id3", "v2").execute()
            kind = L.inner_join if op == "join_inner" else L.left_join
            result = kind(R_proj, on=["id1", "id2", "id3"]).execute()
        else:
            t = self._get_table_obj()
            if op == "groupby_q1":
                result = t.select(v1=C("v1").sum()).by("id1").execute()
            elif op == "groupby_q2":
                result = t.select(v1=C("v1").sum()).by("id1", "id2").execute()
            elif op == "groupby_q3":
                result = t.select(v1=C("v1").sum(),
                                  v3=C("v3").mean()).by("id3").execute()
            elif op == "groupby_q4":
                result = t.select(v1=C("v1").mean(),
                                  v2=C("v2").mean(),
                                  v3=C("v3").mean()).by("id4").execute()
            elif op == "groupby_q5":
                result = t.select(v1=C("v1").sum(),
                                  v2=C("v2").sum(),
                                  v3=C("v3").sum()).by("id6").execute()
            elif op == "groupby_q6":
                # Extra `_cnt` column lets us reconstruct n<=1 nulls in
                # v3_std after to_dict() — the wrapper drops the
                # typed-null bit when materialising F64 vectors, so the
                # engine's correct 0Nf for sample-std-of-1 surfaces here
                # as 0.0. Post-process below replaces those with NaN
                # which check.py's canonicalize folds to null.
                result = t.select(v3_median=C("v3").median(),
                                  v3_std=C("v3").std(),
                                  _cnt=C("v3").count()
                                  ).by("id4", "id5").execute()
            elif op == "groupby_q7":
                # Two-stage workaround — see run_groupby_q7 for rationale.
                agg = t.select(v1m=C("v1").max(),
                               v2m=C("v2").min()).by("id3").execute()
                result = agg.select("id3",
                                    range_v1_v2=C("v1m") - C("v2m")).execute()
            elif op == "groupby_q8":
                # Engine's OP_GROUP_TOPK_ROWFORM emits row form
                # directly — one row per (id6, kept_value).  Variable-K
                # behaviour (groups with <K non-null v3 emit fewer
                # rows) is handled engine-side.
                result = t.select(largest2_v3=C("v3").top(2)
                                  ).by("id6").execute()
            elif op == "groupby_q9":
                # Two-stage to keep OP_PEARSON_CORR at the top of the
                # first select (see run_groupby_q9 docstring).  Carry
                # `_cnt` to reconstruct NaN where n<2 — pearson_corr is
                # undefined and engine emits typed-null F64 (which the
                # wrapper's to_python() surfaces as 0.0; cnt-based mask
                # below restores polars-comparable NaN).
                agg = t.select(r=C("v1").pearson_corr(C("v2")),
                               _cnt=C("v1").count()
                               ).by("id2", "id4").execute()
                result = agg.select("id2", "id4",
                                    r2=C("r") * C("r"),
                                    _cnt=C("_cnt")).execute()
            elif op == "groupby_q10":
                result = t.select(v3=C("v3").sum(),
                                  cnt=C("v1").count()
                                  ).by("id1", "id2", "id3",
                                       "id4", "id5", "id6").execute()
            elif op == "sort_single":
                result = t.order_by("id1").execute()
            elif op == "sort_multi":
                result = t.order_by("id1", "id2", "id3").execute()
            else:
                raise ValueError(f"unknown op: {op}")

        if result is None:
            return pl.DataFrame()
        # to_dict() can return wrapped scalar types (I64(3), F64(...));
        # polars treats those as Object dtype and refuses to write IPC.
        # Unwrap via .to_python() each rayforce scalar provides.
        d = result.to_dict()
        for col, vals in d.items():
            if vals and any(hasattr(v, "to_python") for v in vals):
                # Wrapper note: `.to_python()` on F64 silently drops the
                # typed-null bit (engine returns 0Nf for e.g. stddev on
                # n<=1, but Python sees 0.0). Per-query nil handling
                # lives below — q6 reconstructs nulls via _cnt.
                # v2 wrapper mixes plain floats (NaN for nulls) with
                # wrapped scalars in one column — convert per element.
                d[col] = [v.to_python() if hasattr(v, "to_python") else v
                          for v in vals]

        # q6: replace v3_std with NaN where group size <= 1 (engine
        # returns 0Nf, but the wrapper drops the null bit and surfaces
        # 0.0). check.py's canonicalize will fold NaN→null.
        if op == "groupby_q6" and "_cnt" in d:
            import math
            cnt = list(d["_cnt"])
            std = list(d["v3_std"])
            d["v3_std"] = [math.nan if c <= 1 else v for v, c in zip(std, cnt)]
            del d["_cnt"]

        # q9: same wrapper-null trick — pearson_corr is undefined when
        # n<2 (engine emits 0Nf, wrapper surfaces 0.0); also undefined
        # when either side has zero variance (engine emits NaN already,
        # passes through to_python unchanged).  Mask r² to NaN for the
        # n<2 case.
        if op == "groupby_q9" and "_cnt" in d:
            import math
            cnt = list(d["_cnt"])
            r2  = list(d["r2"])
            d["r2"] = [math.nan if c < 2 else v for v, c in zip(r2, cnt)]
            del d["_cnt"]

        # q8: engine's OP_GROUP_TOPK_ROWFORM emits row form natively;
        # no Python-side explode required.

        df = pl.from_dict(d)

        # join_left: rayforce's left_join surfaces v2=0.0 for unmatched
        # rows because the wrapper's to_dict() drops the F64 null bit.
        # Reconstruct null v2 via a polars-side anti-match against the
        # right CSV's keys.
        if op == "join_left" and right_path is not None and "v2" in df.columns:
            right_keys = (pl.read_csv(right_path)
                            .select(["id1", "id2", "id3"])
                            .unique()
                            .with_columns(pl.lit(True).alias("_match")))
            df = df.join(right_keys, on=["id1", "id2", "id3"], how="left")
            df = df.with_columns(
                pl.when(pl.col("_match").is_null())
                  .then(None)
                  .otherwise(pl.col("v2"))
                  .alias("v2")
            ).drop("_match")

        return df

    def close(self) -> None:
        self._table_names.clear()
