"""
Full geolocalization pipeline — runs all steps in order.

Each run gets its own timestamped folder: results/run_YYYYMMDD_HHMMSS/
A results/latest.txt pointer is updated on every new run so --from X
and the dashboard always know where the current results are.

Usage:
    python src/run_pipeline.py                   # full run
    python src/run_pipeline.py --from vo         # skip extract + tile, reuse latest run dir
    python src/run_pipeline.py --skip-sat        # skip automated satellite anchors
    python src/run_pipeline.py --no-mark         # skip interactive anchor marking
    python src/run_pipeline.py --max-seconds 5   # test run (first 5 seconds only)
    python src/run_pipeline.py --force           # re-run all steps in a new run dir

Steps:
    1. extract  — extract frames from video
    2. tile     — download satellite tile
    3. vo + sat — visual odometry + satellite anchors (parallel)
    5. mark     — interactive manual anchor marking
    6. fuse     — fit + drift-correct + GPS benchmark + plots
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
CODE         = ROOT / "src"
DATA         = ROOT / "data"
FRAMES       = DATA / "frames_15hz"
RESULTS_ROOT = ROOT / "results"
LATEST_TXT   = RESULTS_ROOT / "latest.txt"

STEPS = ["extract", "tile", "vo", "sat", "mark", "fuse"]


# ── Run-dir helpers ───────────────────────────────────────────────────────────

def get_latest_run_dir() -> Path | None:
    if LATEST_TXT.exists():
        p = Path(LATEST_TXT.read_text().strip())
        if p.exists():
            return p
    # Fallback: most recent run_* folder
    runs = sorted(RESULTS_ROOT.glob("run_*/"))
    return runs[-1] if runs else None


def new_run_dir() -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    d   = RESULTS_ROOT / f"run_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    LATEST_TXT.write_text(str(d))
    return d


# ── Skip-check helpers (all take explicit run_dir) ────────────────────────────

def frames_exist() -> bool:
    return len(list(FRAMES.glob("*.jpg"))) >= 100


def tile_exists() -> bool:
    return any(DATA.glob("satellite_*.bbox.json"))


def vo_csv_exists(run_dir: Path) -> bool:
    return (run_dir / "dl_vo_log_full.csv").exists()


def sat_anchors_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("sat_anchors_*.csv"))
    return candidates[-1] if candidates else None


def manual_anchors_exist() -> bool:
    p = DATA / "manual_anchors.json"
    if not p.exists():
        return False
    return len(json.loads(p.read_text())) >= 2


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run(label: str, cmd: list[str], cwd: Path | None = None,
        extra_env: dict | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0  = time.perf_counter()
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label} — exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[done] {label}  ({elapsed:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full dronomy pipeline")
    ap.add_argument("--from", dest="from_step", choices=STEPS, default=None,
                    help="resume from this step using the latest run dir")
    ap.add_argument("--skip-sat",    action="store_true",
                    help="skip automated satellite anchor matching")
    ap.add_argument("--no-mark",     action="store_true",
                    help="skip interactive anchor marking (use existing JSON)")
    ap.add_argument("--force",       action="store_true",
                    help="re-run all steps in a fresh run dir")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only extract/process this many seconds (test mode)")
    args = ap.parse_args()

    start_idx = STEPS.index(args.from_step) if args.from_step else 0
    py        = sys.executable

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Determine run dir:
    #   --from X  → reuse latest (adding results to the same folder)
    #   fresh/--force → create new timestamped folder
    if args.from_step and not args.force:
        run_dir = get_latest_run_dir() or new_run_dir()
        print(f"\nResuming into existing run dir: {run_dir.name}")
    else:
        run_dir = new_run_dir()
        print(f"\nNew run dir: {run_dir.name}")

    # Clean up stale VO sentinel so --watch-vo exits cleanly on re-runs
    sentinel = run_dir / "dl_vo_log_full.done"
    if sentinel.exists() and (args.force or not vo_csv_exists(run_dir)):
        sentinel.unlink()

    print(f"Python: {py}")
    print(f"Root:   {ROOT}\n")

    # ── 1. Extract frames ─────────────────────────────────────────────────────
    if start_idx <= STEPS.index("extract"):
        if not args.force and frames_exist():
            n = len(list(FRAMES.glob("*.jpg")))
            print(f"[skip] extract — {n} frames already in {FRAMES.relative_to(ROOT)}")
        else:
            extract_cmd = [py, str(CODE / "extract_frames.py")]
            if args.max_seconds:
                extract_cmd += ["--max-seconds", str(args.max_seconds)]
            run("Step 1 / 6 — Extract frames", extract_cmd)

    # ── 2. Satellite tile ─────────────────────────────────────────────────────
    if start_idx <= STEPS.index("tile"):
        if not args.force and tile_exists():
            tile = next(DATA.glob("satellite_*.png"))
            print(f"[skip] tile — {tile.name} already exists")
        else:
            run("Step 2 / 6 — Download satellite tile",
                [py, str(CODE / "fetch_satellite.py")])

    # ── 3+4. VO + satellite anchors (parallel when both needed) ──────────────
    vo_needed  = start_idx <= STEPS.index("vo")  and (args.force or not vo_csv_exists(run_dir))
    sat_needed = (start_idx <= STEPS.index("sat") and not args.skip_sat
                  and (args.force or not sat_anchors_csv(run_dir)))

    if not vo_needed:
        print("[skip] vo — dl_vo_log_full.csv already exists")
    if not sat_needed and start_idx <= STEPS.index("sat"):
        if args.skip_sat:
            print("[skip] sat — --skip-sat flag set")
        elif sat_anchors_csv(run_dir):
            print(f"[skip] sat — {sat_anchors_csv(run_dir).name} already exists")

    vo_env = {"VO_OUT_CSV": str(run_dir / "dl_vo_log_full.csv")}

    if vo_needed and sat_needed:
        print("\nRunning VO + satellite anchors in parallel.")
        print("Open a second terminal: streamlit run src/dashboard.py\n")
        t0       = time.perf_counter()
        env      = {**os.environ, **vo_env}
        vo_proc  = subprocess.Popen([py, str(CODE / "vo_run.py")],
                                    cwd=str(CODE), env=env)
        sat_proc = subprocess.Popen([py, str(CODE / "sat_anchors.py"),
                                     "--watch-vo",
                                     "--vo-csv", str(run_dir / "dl_vo_log_full.csv"),
                                     "--out-dir", str(run_dir)])
        vo_proc.wait()
        if vo_proc.returncode != 0:
            sat_proc.terminate()
            print(f"\n[FAILED] VO — exit code {vo_proc.returncode}")
            sys.exit(vo_proc.returncode)
        sat_proc.wait()
        if sat_proc.returncode != 0:
            print(f"\n[FAILED] Satellite anchors — exit code {sat_proc.returncode}")
            sys.exit(sat_proc.returncode)
        print(f"\n[done] VO + satellite anchors  ({time.perf_counter()-t0:.0f}s)")

    elif vo_needed:
        print("\nNOTE: VO runs on CPU (~45–90 min). "
              "Open a second terminal: streamlit run src/dashboard.py\n")
        run("Step 3 / 6 — Visual odometry (SuperPoint + LightGlue)",
            [py, str(CODE / "vo_run.py")], cwd=CODE, extra_env=vo_env)

    elif sat_needed:
        run("Step 4 / 6 — Automated satellite anchors",
            [py, str(CODE / "sat_anchors.py"),
             "--vo-csv", str(run_dir / "dl_vo_log_full.csv"),
             "--out-dir", str(run_dir)])

    # ── 5. Interactive anchor marking ─────────────────────────────────────────
    if start_idx <= STEPS.index("mark"):
        if args.no_mark:
            print("[skip] mark — --no-mark flag set")
        elif not args.force and manual_anchors_exist():
            n = len(json.loads((DATA / "manual_anchors.json").read_text()))
            print(f"[skip] mark — manual_anchors.json already has {n} marks "
                  "(use --force to re-open)")
        else:
            print("\nNOTE: Left-click on the satellite tile to mark each frame. "
                  "Right-click to skip. Close window to finish early.\n")
            run("Step 5 / 6 — Interactive anchor marking",
                [py, str(CODE / "mark_anchors.py")])

    # ── 6. Fuse + benchmark ───────────────────────────────────────────────────
    if start_idx <= STEPS.index("fuse"):
        sat_csv = sat_anchors_csv(run_dir)
        cmd = [py, str(CODE / "fuse.py"),
               "--vo-csv",  str(run_dir / "dl_vo_log_full.csv"),
               "--out-dir", str(run_dir)]
        if sat_csv:
            cmd += ["--sat-anchors", str(sat_csv)]
        run("Step 6 / 6 — Fuse VO + anchors + GPS benchmark", cmd)

    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"  Results in: {run_dir.relative_to(ROOT)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
