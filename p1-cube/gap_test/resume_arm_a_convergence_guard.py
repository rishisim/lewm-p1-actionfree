#!/usr/bin/env python3
"""Resume Arm A on Vast until validation curves converge, then back up results.

Run this locally under caffeinate. Cloud credentials stay local; remote work is
done only through SSH and rsync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


P1_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = P1_ROOT.parent
OUTPUT_ROOT = P1_ROOT / "gap_test" / "outputs"
STATE_PATH = OUTPUT_ROOT / "arm_a_convergence_guard_state.json"
LOCAL_BUNDLE_ROOT = OUTPUT_ROOT / "vast_8xh100_guard"
README_PATH = P1_ROOT / "gap_test" / "README.md"
README_START = "<!-- gap-test-converged-results:start -->"
README_END = "<!-- gap-test-converged-results:end -->"

REMOTE_REPO = "/workspace/lewm-p1-inversedynamics"
REMOTE_PYTHON = "/venv/main/bin/python"
PRETRAIN_HDF5 = f"{REMOTE_REPO}/p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5"
READOUT_HDF5 = f"{REMOTE_REPO}/p1-cube/data/lewm_hdf5/visual_cube_single_play_readout_heldout.h5"
SPLIT_JSON = f"{REMOTE_REPO}/p1-cube/gap_test/outputs/heldout_readout_split_seed2026.json"
ARM_A_OUTPUT = "lewm_cube_play_arm_a_conditioned_8xh100_ddp_bs768"
ARM_A_CHECKPOINT_DIR = f"/root/.stable_worldmodel/checkpoints/{ARM_A_OUTPUT}"
ORIGINAL_METRICS_CSV = "/root/.cache/stable-pretraining/runs/20260618/014031/75af218b74f8/metrics.csv"
ORIGINAL_LAST_CKPT = "/root/.cache/stable-pretraining/runs/20260618/014031/75af218b74f8/checkpoints/last.ckpt"
LOCAL_100_EPOCH_BUNDLE = OUTPUT_ROOT / "vast_8xh100_guard" / "arm_a_20260618T074538Z"
CONVERGED_OUT_DIR = f"{REMOTE_REPO}/p1-cube/gap_test/outputs/arm_a_converged"
CONVERGED_CKPT_DIR = f"{CONVERGED_OUT_DIR}/resume_lightning_checkpoints"
COMBINED_METRICS_CSV = f"{CONVERGED_OUT_DIR}/arm_a_converged_combined_metrics.csv"
ARTIFACT_PREFIX = "arm_a_converged"
BASELINE_100_EPOCH_R2_NO_ACTION_3 = 0.87806636095047


@dataclass(frozen=True)
class VastTarget:
    host: str
    port: int
    ssh_key: Path
    instance_id: int
    remote_repo: str
    remote_python: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run_local(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ssh_base(target: VastTarget, *, connect_timeout: int | None = None) -> list[str]:
    parts = [
        "ssh",
        "-i",
        str(target.ssh_key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=10",
    ]
    if connect_timeout is not None:
        parts.extend(["-o", f"ConnectTimeout={connect_timeout}"])
    parts.extend(["-p", str(target.port), target.host])
    return parts


def run_remote(
    target: VastTarget,
    command: str,
    *,
    check: bool = True,
    timeout: int | None = None,
    connect_timeout: int | None = None,
) -> str:
    proc = run_local(
        ssh_base(target, connect_timeout=connect_timeout) + [command],
        check=False,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"remote command failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
    return proc.stdout


def vast_set_state(target: VastTarget, state: str) -> None:
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise RuntimeError("VAST_API_KEY is not set locally")
    url = f"https://console.vast.ai/api/v0/instances/{target.instance_id}/"
    body = json.dumps({"state": state}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            log(f"Vast state={state} response status={resp.status} body={payload[:300]}")
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vast state={state} failed status={exc.code} body={payload[:500]}") from exc


def wait_for_ssh(target: VastTarget, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        out = run_remote(target, "echo ssh_ready", check=False, timeout=20, connect_timeout=8)
        if "ssh_ready" in out:
            log("SSH is ready")
            return
        time.sleep(20)
    raise TimeoutError("Vast SSH did not become ready before timeout")


def rsync_file_to_remote(target: VastTarget, local_path: Path, remote_path: str) -> None:
    run_remote(target, f"mkdir -p {q(str(Path(remote_path).parent))}")
    remote = f"{target.host}:{remote_path}"
    cmd = [
        "rsync",
        "-az",
        "--partial",
        "-e",
        " ".join(shlex.quote(part) for part in ssh_base(target)[:-1]),
        str(local_path),
        remote,
    ]
    proc = run_local(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"rsync to remote failed rc={proc.returncode}\n{proc.stdout[-2000:]}")


def sync_sources(target: VastTarget) -> None:
    files = [
        (REPO_ROOT / "le-wm" / "train.py", f"{target.remote_repo}/le-wm/train.py"),
        (
            P1_ROOT / "gap_test" / "summarize_training_curves.py",
            f"{target.remote_repo}/p1-cube/gap_test/summarize_training_curves.py",
        ),
        (
            P1_ROOT / "gap_test" / "build_encoder_readout_cache.py",
            f"{target.remote_repo}/p1-cube/gap_test/build_encoder_readout_cache.py",
        ),
        (
            P1_ROOT / "gap_test" / "run_arm_readout.py",
            f"{target.remote_repo}/p1-cube/gap_test/run_arm_readout.py",
        ),
        (
            P1_ROOT / "gap_test" / "histogram_action3.py",
            f"{target.remote_repo}/p1-cube/gap_test/histogram_action3.py",
        ),
    ]
    for local_path, remote_path in files:
        rsync_file_to_remote(target, local_path, remote_path)
    compile_cmd = (
        f"cd {q(target.remote_repo)} && {q(target.remote_python)} -m py_compile "
        "le-wm/train.py "
        "p1-cube/gap_test/summarize_training_curves.py "
        "p1-cube/gap_test/build_encoder_readout_cache.py "
        "p1-cube/gap_test/run_arm_readout.py "
        "p1-cube/gap_test/histogram_action3.py"
    )
    run_remote(target, compile_cmd)
    log("synced and py-compiled resume helper files on Vast")


def remote_exists(target: VastTarget, path: str) -> bool:
    out = run_remote(target, f"test -e {q(path)} && echo yes || echo no")
    return out.strip().endswith("yes")


def ensure_remote_seed_artifacts(target: VastTarget) -> None:
    mappings = [
        (LOCAL_100_EPOCH_BUNDLE / "run" / "metrics.csv", ORIGINAL_METRICS_CSV),
        (LOCAL_100_EPOCH_BUNDLE / "run" / "checkpoints" / "last.ckpt", ORIGINAL_LAST_CKPT),
        (LOCAL_100_EPOCH_BUNDLE / "checkpoint" / "weights_epoch_100.pt", f"{ARM_A_CHECKPOINT_DIR}/weights_epoch_100.pt"),
        (LOCAL_100_EPOCH_BUNDLE / "checkpoint" / "config.json", f"{ARM_A_CHECKPOINT_DIR}/config.json"),
        (LOCAL_100_EPOCH_BUNDLE / "checkpoint" / "config.yaml", f"{ARM_A_CHECKPOINT_DIR}/config.yaml"),
        (OUTPUT_ROOT / "heldout_readout_split_seed2026.json", SPLIT_JSON),
    ]
    for local_path, remote_path in mappings:
        if remote_exists(target, remote_path):
            continue
        if not local_path.exists():
            raise FileNotFoundError(f"missing local seed artifact: {local_path}")
        log(f"restoring missing remote seed artifact: {remote_path}")
        rsync_file_to_remote(target, local_path, remote_path)


def latest_weight_epoch(target: VastTarget) -> int:
    script = f"""
python3 - <<'PY'
import re
from pathlib import Path
p=Path({ARM_A_CHECKPOINT_DIR!r})
best=0
if p.exists():
    for f in p.glob("weights_epoch_*.pt"):
        m=re.search(r"_(\\d+)\\.pt$", f.name)
        if m:
            best=max(best, int(m.group(1)))
print(best)
PY
"""
    return int(run_remote(target, script).strip().splitlines()[-1])


def training_process_count(target: VastTarget) -> int:
    pattern = f"output_model_name={ARM_A_OUTPUT}"
    cmd = f"pgrep -af -- {q(pattern)} | grep -v pgrep | wc -l | tr -d ' '"
    out = run_remote(target, cmd, check=False)
    try:
        return int(out.strip().splitlines()[-1])
    except Exception:
        return 0


def newest_resume_ckpt(target: VastTarget) -> str:
    if remote_exists(target, f"{CONVERGED_CKPT_DIR}/last.ckpt"):
        return f"{CONVERGED_CKPT_DIR}/last.ckpt"
    return ORIGINAL_LAST_CKPT


def launch_training_block(target: VastTarget, target_epoch: int, resume_ckpt: str) -> str:
    logs = f"{CONVERGED_OUT_DIR}/logs"
    pid_path = f"{logs}/resume_to_{target_epoch}.pid"
    log_path = f"{logs}/resume_to_{target_epoch}.log"
    csv_dir = f"{CONVERGED_OUT_DIR}/csv_logs/to_{target_epoch}"
    inner = (
        "set -euo pipefail\n"
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "export WANDB_MODE=disabled\n"
        "export PYTHONUNBUFFERED=1\n"
        "export NCCL_DEBUG=WARN\n"
        "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
        f"cd {q(target.remote_repo + '/le-wm')}\n"
        f"{q(target.remote_python)} -u train.py "
        f"data=ogb "
        f"data.dataset.name={q(PRETRAIN_HDF5)} "
        f"action_free=false "
        f"trainer.accelerator=gpu "
        f"trainer.devices=8 "
        f"trainer.precision=bf16 "
        f"+trainer.strategy=ddp "
        f"trainer.max_epochs={target_epoch} "
        f"loader.batch_size=768 "
        f"loader.num_workers=4 "
        f"loader.persistent_workers=true "
        f"loader.prefetch_factor=2 "
        f"loader.pin_memory=true "
        f"output_model_name={ARM_A_OUTPUT} "
        f"subdir={ARM_A_OUTPUT} "
        f"+resume_fit_ckpt_path={q(resume_ckpt)} "
        f"+resume_checkpoint_dir={q(CONVERGED_CKPT_DIR)} "
        f"+resume_csv_log_dir={q(csv_dir)}\n"
    )
    command = (
        "set -euo pipefail\n"
        f"mkdir -p {q(logs)} {q(csv_dir)} {q(CONVERGED_CKPT_DIR)}\n"
        f"if pgrep -af -- {q('output_model_name=' + ARM_A_OUTPUT)} | grep -v pgrep; then "
        "echo 'matching training process already exists' >&2; exit 12; fi\n"
        f"nohup bash -lc {q(inner)} > {q(log_path)} 2>&1 < /dev/null &\n"
        f"echo $! > {q(pid_path)}\n"
        "sleep 20\n"
        f"kill -0 $(cat {q(pid_path)})\n"
        f"echo launched_pid=$(cat {q(pid_path)}) target_epoch={target_epoch} log={q(log_path)}"
    )
    line = run_remote(target, command, timeout=120).strip().splitlines()[-1]
    log(line)
    return log_path


def wait_for_block(target: VastTarget, target_epoch: int, poll_seconds: int) -> None:
    last_seen = -1
    while True:
        epoch = latest_weight_epoch(target)
        running = training_process_count(target)
        if epoch != last_seen:
            log(f"arm_a_resume: latest weights_epoch_{epoch}.pt; target={target_epoch}; train_processes={running}")
            last_seen = epoch
        if epoch >= target_epoch and running == 0:
            log(f"arm_a_resume: target epoch {target_epoch} complete")
            return
        if running == 0 and epoch < target_epoch:
            raise RuntimeError(f"training exited before target epoch {target_epoch}; latest epoch={epoch}")
        time.sleep(poll_seconds)


def combine_metrics(target: VastTarget) -> str:
    script = f"""
set -euo pipefail
mkdir -p {q(CONVERGED_OUT_DIR)}
{q(target.remote_python)} - <<'PY'
from pathlib import Path
import pandas as pd
base = Path({CONVERGED_OUT_DIR!r})
files = [Path({ORIGINAL_METRICS_CSV!r})]
csv_root = base / "csv_logs"
if csv_root.exists():
    files.extend(sorted(csv_root.rglob("metrics.csv")))
frames = []
for path in files:
    if not path.exists():
        continue
    df = pd.read_csv(path)
    df["__source_file"] = str(path)
    frames.append(df)
if not frames:
    raise SystemExit("no metrics files found")
combined = pd.concat(frames, ignore_index=True, sort=False)
out = Path({COMBINED_METRICS_CSV!r})
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out, index=False)
print(out)
PY
"""
    return run_remote(target, script, timeout=60 * 10).strip().splitlines()[-1]


def summarize_curves(target: VastTarget) -> dict[str, Any]:
    combined = combine_metrics(target)
    cmd = (
        f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
        "p1-cube/gap_test/summarize_training_curves.py "
        f"--metrics-csv {q(combined)} "
        f"--out-dir {q(CONVERGED_OUT_DIR)} "
        f"--artifact-prefix {q(ARTIFACT_PREFIX)} "
        "--window-epochs 20 --mean-change-threshold 0.03 --floor-threshold 0.05"
    )
    out = run_remote(target, cmd, timeout=60 * 15)
    log(out.strip().splitlines()[-1])
    summary_path = f"{CONVERGED_OUT_DIR}/{ARTIFACT_PREFIX}_training_curves_summary.json"
    raw = run_remote(target, f"cat {q(summary_path)}", timeout=60)
    return json.loads(raw)


def recommended_budget(plateau_epoch: int) -> int:
    margin = max(50, math.ceil(0.25 * plateau_epoch))
    return int(math.ceil((plateau_epoch + margin) / 25.0) * 25)


def run_readout_pipeline(target: VastTarget, weight_epoch: int) -> None:
    weights = f"{ARM_A_CHECKPOINT_DIR}/weights_epoch_{weight_epoch}.pt"
    config = f"{ARM_A_CHECKPOINT_DIR}/config.json"
    cache_npz = f"{CONVERGED_OUT_DIR}/{ARTIFACT_PREFIX}_encoder_transitions.npz"
    cache_summary = f"{CONVERGED_OUT_DIR}/{ARTIFACT_PREFIX}_encoder_transitions_summary.json"
    commands = [
        (
            f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
            "p1-cube/gap_test/build_encoder_readout_cache.py "
            f"--hdf5 {q(READOUT_HDF5)} --split {q(SPLIT_JSON)} "
            f"--config {q(config)} --weights {q(weights)} "
            f"--out {q(cache_npz)} --summary {q(cache_summary)} "
            "--batch-size 512 --device cuda "
            f"--encoder-label {q(f'Cube LeWM Arm A validation-converged encoder epoch {weight_epoch}')}"
        ),
        (
            f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
            "p1-cube/gap_test/run_arm_readout.py "
            f"--data {q(cache_npz)} --summary {q(cache_summary)} "
            f"--out-dir {q(CONVERGED_OUT_DIR)} --artifact-prefix {q(ARTIFACT_PREFIX)} --device cuda"
        ),
        (
            f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
            "p1-cube/gap_test/histogram_action3.py "
            f"--data {q(cache_npz)} --out-dir {q(CONVERGED_OUT_DIR)} "
            f"--artifact-prefix {q(ARTIFACT_PREFIX)}"
        ),
    ]
    for i, command in enumerate(commands, start=1):
        log(f"arm_a_converged: running readout/hist step {i}/3")
        out = run_remote(target, command, timeout=60 * 60 * 4)
        if out.strip():
            log(out.strip().splitlines()[-1])


def create_remote_bundle(target: VastTarget, weight_epoch: int, plateau_epoch: int, budget: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = f"{target.remote_repo}/p1-cube/gap_test/outputs/vast_backups/arm_a_converged_{stamp}"
    script = f"""
set -euo pipefail
BUNDLE={q(bundle)}
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/checkpoint" "$BUNDLE/run/checkpoints" "$BUNDLE/outputs/arm_a_converged" "$BUNDLE/logs"
copy_if_exists() {{
  src="$1"; dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}}
copy_if_exists {q(ARM_A_CHECKPOINT_DIR + f'/weights_epoch_{weight_epoch}.pt')} "$BUNDLE/checkpoint/weights_epoch_{weight_epoch}.pt"
copy_if_exists {q(ARM_A_CHECKPOINT_DIR + '/config.json')} "$BUNDLE/checkpoint/config.json"
copy_if_exists {q(ARM_A_CHECKPOINT_DIR + '/config.yaml')} "$BUNDLE/checkpoint/config.yaml"
copy_if_exists {q(COMBINED_METRICS_CSV)} "$BUNDLE/run/combined_metrics.csv"
copy_if_exists {q(CONVERGED_CKPT_DIR + '/last.ckpt')} "$BUNDLE/run/checkpoints/last.ckpt"
if [ -d {q(CONVERGED_OUT_DIR)} ]; then
  cp -a {q(CONVERGED_OUT_DIR)}/. "$BUNDLE/outputs/arm_a_converged/"
fi
copy_if_exists {q(SPLIT_JSON)} "$BUNDLE/outputs/heldout_readout_split_seed2026.json"
if [ -d {q(CONVERGED_OUT_DIR + '/logs')} ]; then
  cp -a {q(CONVERGED_OUT_DIR + '/logs')}/. "$BUNDLE/logs/"
fi
cat > "$BUNDLE/RUN_LABEL.txt" <<EOF
Arm A validation-converged continuation
plateau_epoch={plateau_epoch}
readout_weight_epoch={weight_epoch}
recommended_fixed_epoch_budget={budget}
EOF
(cd "$BUNDLE" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
echo "$BUNDLE"
"""
    remote_bundle = run_remote(target, script, timeout=60 * 30).strip().splitlines()[-1]
    log(f"arm_a_converged: remote bundle staged at {remote_bundle}")
    return remote_bundle


def rsync_bundle(target: VastTarget, remote_bundle: str) -> Path:
    local_dir = LOCAL_BUNDLE_ROOT / Path(remote_bundle).name
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        log(f"rsync attempt {attempt}: {remote_bundle} -> {local_dir}")
        cmd = [
            "rsync",
            "-az",
            "--partial",
            "-e",
            " ".join(shlex.quote(part) for part in ssh_base(target)[:-1]),
            f"{target.host}:{remote_bundle}/",
            f"{local_dir}/",
        ]
        proc = run_local(cmd, check=False)
        if proc.returncode != 0:
            log(f"rsync failed rc={proc.returncode}: {proc.stdout[-1000:]}")
            time.sleep(30)
            continue
        if verify_remote_manifest(local_dir):
            return local_dir
        log("remote manifest verification failed locally; retrying rsync")
        time.sleep(30)
    raise RuntimeError(f"could not verify rsync bundle after retries: {remote_bundle}")


def verify_remote_manifest(local_dir: Path) -> bool:
    manifest = local_dir / "SHA256SUMS.txt"
    if not manifest.exists():
        return False
    proc = run_local(["shasum", "-a", "256", "-c", "SHA256SUMS.txt"], cwd=local_dir, check=False)
    ok = proc.returncode == 0
    log(f"remote manifest verification {'passed' if ok else 'failed'} for {local_dir.name}")
    if not ok:
        log(proc.stdout[-2000:])
    return ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_local_manifest(local_dir: Path) -> None:
    files = [
        p
        for p in sorted(local_dir.rglob("*"))
        if p.is_file() and p.name not in {"LOCAL_SHA256SUMS.txt"}
    ]
    lines = [f"{sha256_file(p)}  {p.relative_to(local_dir)}" for p in files]
    manifest = local_dir / "LOCAL_SHA256SUMS.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        digest, rel = line.split("  ", 1)
        if sha256_file(local_dir / rel) != digest:
            raise RuntimeError(f"local manifest verification failed for {rel}")
    log(f"local manifest verification passed for {local_dir.name}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_record(bundle: Path, plateau_epoch: int, final_epoch: int, budget: int) -> dict[str, Any]:
    out = bundle / "outputs" / "arm_a_converged"
    curves = load_json(out / f"{ARTIFACT_PREFIX}_training_curves_summary.json")
    readout = load_json(out / f"{ARTIFACT_PREFIX}_readout_summary.json")
    histogram = load_json(out / f"{ARTIFACT_PREFIX}_action3_histogram_summary.json")
    return {
        "created_at_utc": utc_now(),
        "bundle": str(bundle),
        "plateau_epoch": plateau_epoch,
        "final_trained_epoch": final_epoch,
        "recommended_fixed_epoch_budget": budget,
        "curves": curves,
        "readout": readout,
        "histogram": histogram,
    }


def metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_convergence_record(record: dict[str, Any]) -> str:
    curves = record["curves"]
    val = curves["plateau"]
    readout = record["readout"]
    headline = readout["headline"]
    baseline = headline["mean_action_baseline"]
    hist_all = record["histogram"]["splits"]["all"]
    delta = headline["test_r2_excluding_action_3"] - BASELINE_100_EPOCH_R2_NO_ACTION_3
    per_dim = readout["metrics"]["test"]["readout_mlp"]["per_dim"]
    lines = [
        "## Arm A Validation-Converged Ceiling",
        "",
        f"- Local verified bundle: `{Path(record['bundle']).relative_to(REPO_ROOT)}`",
        f"- Validation plateau epoch: `{record['plateau_epoch']}`",
        f"- Final trained epoch in this continuation block: `{record['final_trained_epoch']}`",
        f"- Recommended fixed epoch budget for the full matrix: `{record['recommended_fixed_epoch_budget']}`",
        f"- Validation pred: last `{metric_value(val['validate_pred_loss']['last'])}`, "
        f"last-window mean `{metric_value(val['validate_pred_loss']['last_window_mean'])}`, "
        f"previous-window mean `{metric_value(val['validate_pred_loss']['previous_window_mean'])}`, "
        f"near floor `{val['validate_pred_loss']['at_running_floor']}`",
        f"- Validation SIGReg: last `{metric_value(val['validate_sigreg_loss']['last'])}`, "
        f"last-window mean `{metric_value(val['validate_sigreg_loss']['last_window_mean'])}`, "
        f"previous-window mean `{metric_value(val['validate_sigreg_loss']['previous_window_mean'])}`, "
        f"near floor `{val['validate_sigreg_loss']['at_running_floor']}`",
        f"- Test R2 all dims: `{headline['test_r2_all_dims']:.6f}`; excluding action_3: `{headline['test_r2_excluding_action_3']:.6f}`",
        f"- Ceiling movement vs 100-epoch no-action_3 R2 `0.878066`: `{delta:+.6f}`",
        f"- Test normalized MSE excluding action_3: readout `{headline['test_train_variance_normalized_mse_excluding_action_3']:.6f}` "
        f"vs mean baseline `{baseline['test_train_variance_normalized_mse_excluding_action_3']:.6f}`",
        f"- action_3 separately: R2 `{headline['action_3']['r2']:.6f}`, "
        f"normalized MSE `{headline['action_3']['train_variance_normalized_mse']:.6f}`",
        f"- action_3 histogram: min `{hist_all['min']:.6f}`, max `{hist_all['max']:.6f}`, "
        f"near-extremes fraction `{hist_all['near_extremes_fraction']:.3f}`, "
        f"middle-80%-span fraction `{hist_all['middle_80pct_span_fraction']:.3f}`",
        "",
        "| dim | R2 | normalized MSE |",
        "| --- | ---: | ---: |",
    ]
    for row in per_dim:
        lines.append(
            f"| {row['name']} | {row['r2']:.6f} | {row['train_variance_normalized_mse']:.6f} |"
        )
    return "\n".join(lines)


def update_readme(record: dict[str, Any]) -> None:
    existing = README_PATH.read_text(encoding="utf-8")
    replacement = f"{README_START}\n\n{render_convergence_record(record)}\n\n{README_END}\n"
    if README_START in existing and README_END in existing:
        before = existing.split(README_START, 1)[0].rstrip()
        after = existing.split(README_END, 1)[1].lstrip()
        updated = f"{before}\n\n{replacement}\n{after}"
    else:
        updated = existing.rstrip() + "\n\n" + replacement
    README_PATH.write_text(updated, encoding="utf-8")
    STATE_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"updated README: {README_PATH}")


def copy_readme_into_bundle(bundle: Path) -> None:
    (bundle / "README.local.md").write_text(README_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="root@212.13.234.26")
    parser.add_argument("--port", type=int, default=29769)
    parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "vast_inversedynamics")
    parser.add_argument("--instance-id", type=int, default=41405880)
    parser.add_argument("--remote-repo", default=REMOTE_REPO)
    parser.add_argument("--remote-python", default=REMOTE_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--block-epochs", type=int, default=25)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--start-instance", action="store_true")
    parser.add_argument("--dry-run-stop", action="store_true")
    args = parser.parse_args()

    target = VastTarget(
        host=args.host,
        port=args.port,
        ssh_key=args.ssh_key,
        instance_id=args.instance_id,
        remote_repo=args.remote_repo,
        remote_python=args.remote_python,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    log("Arm A convergence guard started")
    if args.start_instance:
        vast_set_state(target, "running")
    wait_for_ssh(target, timeout_seconds=60 * 30)
    sync_sources(target)
    ensure_remote_seed_artifacts(target)

    summary = summarize_curves(target)
    current_epoch = latest_weight_epoch(target)
    while not summary.get("both_validate_converged_at_floor", False):
        if current_epoch >= args.max_epoch:
            raise RuntimeError(f"validation convergence not reached by max_epoch={args.max_epoch}")
        target_epoch = min(args.max_epoch, current_epoch + args.block_epochs)
        resume_ckpt = newest_resume_ckpt(target)
        launch_training_block(target, target_epoch, resume_ckpt)
        wait_for_block(target, target_epoch, args.poll_seconds)
        current_epoch = latest_weight_epoch(target)
        summary = summarize_curves(target)

    plateau_epoch = int(summary["validation_plateau"]["epoch_1based"])
    final_epoch = current_epoch
    budget = recommended_budget(plateau_epoch)
    log(f"validation plateau epoch={plateau_epoch}; final_epoch={final_epoch}; recommended_budget={budget}")

    run_readout_pipeline(target, plateau_epoch)
    remote_bundle = create_remote_bundle(target, plateau_epoch, plateau_epoch, budget)
    local_bundle = rsync_bundle(target, remote_bundle)
    record = result_record(local_bundle, plateau_epoch, final_epoch, budget)
    update_readme(record)
    copy_readme_into_bundle(local_bundle)
    write_local_manifest(local_bundle)
    if not args.dry_run_stop:
        vast_set_state(target, "stopped")
    else:
        log("dry-run stop requested; leaving Vast instance running")
    log("Arm A convergence guard finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
