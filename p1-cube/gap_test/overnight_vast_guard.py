#!/usr/bin/env python3
"""Local overnight guard for the Vast 8x H100 Cube Gap runs.

This script is meant to run on the local Mac under caffeinate. It never writes
cloud credentials to the Vast box; it only uses SSH/rsync for remote work and
uses VAST_API_KEY locally if the instance should be stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
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
README_PATH = P1_ROOT / "gap_test" / "README.md"
STATE_PATH = OUTPUT_ROOT / "overnight_guard_state.json"
LOCAL_BUNDLE_ROOT = OUTPUT_ROOT / "vast_8xh100_guard"
README_START = "<!-- gap-test-overnight-results:start -->"
README_END = "<!-- gap-test-overnight-results:end -->"
PROBE_R2_ALL = 0.4354148805141449
PROBE_R2_NO_ACTION_3 = (0.790639 + 0.556731 + 0.806556 + 0.694505) / 4.0


@dataclass(frozen=True)
class VastTarget:
    host: str
    port: int
    ssh_key: Path
    instance_id: int
    remote_repo: str
    remote_python: str


@dataclass(frozen=True)
class ArmRun:
    key: str
    title: str
    output_model: str
    action_free: bool
    out_dir_name: str
    artifact_prefix: str
    encoder_label: str
    metrics_csv: str | None = None

    @property
    def checkpoint_dir(self) -> str:
        return f"/root/.stable_worldmodel/checkpoints/{self.output_model}"

    @property
    def remote_output_dir(self) -> str:
        return f"{REMOTE_REPO}/p1-cube/gap_test/outputs/{self.out_dir_name}"


REMOTE_REPO = "/workspace/lewm-p1-inversedynamics"
SPLIT_JSON = f"{REMOTE_REPO}/p1-cube/gap_test/outputs/heldout_readout_split_seed2026.json"
PRETRAIN_HDF5 = f"{REMOTE_REPO}/p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5"
READOUT_HDF5 = f"{REMOTE_REPO}/p1-cube/data/lewm_hdf5/visual_cube_single_play_readout_heldout.h5"

ARM_A = ArmRun(
    key="arm_a",
    title="Arm A action-conditioned",
    output_model="lewm_cube_play_arm_a_conditioned_8xh100_ddp_bs768",
    action_free=False,
    out_dir_name="arm_a_conditioned",
    artifact_prefix="arm_a",
    encoder_label="Cube LeWM Arm A action-conditioned encoder",
    metrics_csv="/root/.cache/stable-pretraining/runs/20260618/014031/75af218b74f8/metrics.csv",
)
ARM_B = ArmRun(
    key="arm_b",
    title="Arm B action-free",
    output_model="lewm_cube_play_arm_b_action_free_8xh100_ddp_bs768",
    action_free=True,
    out_dir_name="arm_b_action_free",
    artifact_prefix="arm_b",
    encoder_label="Cube LeWM Arm B action-free encoder",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run_local(cmd: list[str], *, cwd: Path | None = None, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ssh_base(target: VastTarget) -> list[str]:
    return [
        "ssh",
        "-i",
        str(target.ssh_key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=10",
        "-p",
        str(target.port),
        target.host,
    ]


def run_remote(target: VastTarget, command: str, *, check: bool = True, timeout: int | None = None) -> str:
    proc = run_local(ssh_base(target) + [command], check=False, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"remote command failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
    return proc.stdout


def remote_exists(target: VastTarget, path: str) -> bool:
    out = run_remote(target, f"test -e {q(path)} && echo yes || echo no")
    return out.strip().endswith("yes")


def latest_weight_epoch(target: VastTarget, arm: ArmRun) -> int:
    script = f"""
python3 - <<'PY'
import re
from pathlib import Path
p=Path({arm.checkpoint_dir!r})
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


def training_process_count(target: VastTarget, arm: ArmRun) -> int:
    pattern = f"output_model_name={arm.output_model}"
    cmd = f"pgrep -af -- {q(pattern)} | grep -v pgrep | wc -l | tr -d ' '"
    out = run_remote(target, cmd, check=False)
    try:
        return int(out.strip().splitlines()[-1])
    except Exception:
        return 0


def newest_metrics_csv(target: VastTarget) -> str:
    cmd = (
        "find /root/.cache/stable-pretraining/runs -name metrics.csv "
        "-printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-"
    )
    out = run_remote(target, cmd).strip()
    if not out:
        raise RuntimeError("could not identify newest metrics.csv")
    return out.splitlines()[-1]


def wait_for_clean_training(target: VastTarget, arm: ArmRun, poll_seconds: int) -> str:
    final_weights = f"{arm.checkpoint_dir}/weights_epoch_100.pt"
    last_seen = -1
    while True:
        epoch = latest_weight_epoch(target, arm)
        running = training_process_count(target, arm)
        if epoch != last_seen:
            remaining = max(0, 100 - epoch)
            log(f"{arm.key}: weights_epoch_{epoch}.pt observed; train_processes={running}; approx epochs remaining={remaining}")
            last_seen = epoch

        if remote_exists(target, final_weights):
            if running == 0:
                log(f"{arm.key}: final weights exist and trainer process has exited")
                return arm.metrics_csv or newest_metrics_csv(target)
            log(f"{arm.key}: final weights exist; waiting for trainer cleanup ({running} processes)")
            time.sleep(min(120, poll_seconds))
            continue

        if running == 0:
            raise RuntimeError(f"{arm.key}: training exited before {final_weights} existed; latest epoch={epoch}")

        time.sleep(poll_seconds)


def run_remote_pipeline(target: VastTarget, arm: ArmRun, metrics_csv: str) -> None:
    out_dir = f"{target.remote_repo}/p1-cube/gap_test/outputs/{arm.out_dir_name}"
    weights = f"{arm.checkpoint_dir}/weights_epoch_100.pt"
    config = f"{arm.checkpoint_dir}/config.json"
    cache_npz = f"{out_dir}/{arm.artifact_prefix}_encoder_transitions.npz"
    cache_summary = f"{out_dir}/{arm.artifact_prefix}_encoder_transitions_summary.json"
    commands = [
        (
            f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
            f"p1-cube/gap_test/summarize_training_curves.py "
            f"--metrics-csv {q(metrics_csv)} --out-dir {q(out_dir)} "
            f"--artifact-prefix {q(arm.artifact_prefix)}"
        ),
        (
            f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
            f"p1-cube/gap_test/build_encoder_readout_cache.py "
            f"--hdf5 {q(READOUT_HDF5)} --split {q(SPLIT_JSON)} "
            f"--config {q(config)} --weights {q(weights)} "
            f"--out {q(cache_npz)} --summary {q(cache_summary)} "
            f"--batch-size 512 --device cuda --encoder-label {q(arm.encoder_label)}"
        ),
        (
            f"cd {q(target.remote_repo)} && {q(target.remote_python)} "
            f"p1-cube/gap_test/run_arm_readout.py "
            f"--data {q(cache_npz)} --summary {q(cache_summary)} "
            f"--out-dir {q(out_dir)} --artifact-prefix {q(arm.artifact_prefix)} --device cuda"
        ),
    ]
    for i, command in enumerate(commands, start=1):
        log(f"{arm.key}: running post step {i}/3")
        out = run_remote(target, command, timeout=60 * 60 * 4)
        log(out.strip().splitlines()[-1] if out.strip() else f"{arm.key}: post step {i} complete")


def launch_arm_b(target: VastTarget) -> None:
    arm = ARM_B
    out = f"{target.remote_repo}/p1-cube/gap_test/outputs"
    log_path = f"{out}/{arm.output_model}.log"
    pid_path = f"{out}/{arm.output_model}.pid"
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
        f"action_free=true "
        f"trainer.accelerator=gpu "
        f"trainer.devices=8 "
        f"trainer.precision=bf16 "
        f"+trainer.strategy=ddp "
        f"trainer.max_epochs=100 "
        f"loader.batch_size=768 "
        f"loader.num_workers=4 "
        f"loader.persistent_workers=true "
        f"loader.prefetch_factor=2 "
        f"loader.pin_memory=true "
        f"output_model_name={arm.output_model} "
        f"subdir={arm.output_model}\n"
    )
    command = (
        "set -euo pipefail\n"
        f"mkdir -p {q(out)}\n"
        f"if pgrep -af -- {q('output_model_name=' + arm.output_model)} | grep -v pgrep; then "
        "echo 'matching training process already exists' >&2; exit 12; fi\n"
        f"nohup bash -lc {q(inner)} > {q(log_path)} 2>&1 < /dev/null &\n"
        f"echo $! > {q(pid_path)}\n"
        "sleep 20\n"
        f"kill -0 $(cat {q(pid_path)})\n"
        f"echo launched_pid=$(cat {q(pid_path)}) log={q(log_path)}"
    )
    log(run_remote(target, command).strip().splitlines()[-1])


def create_remote_bundle(target: VastTarget, arm: ArmRun, metrics_csv: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = f"{target.remote_repo}/p1-cube/gap_test/outputs/vast_backups/{arm.key}_{stamp}"
    out_dir = f"{target.remote_repo}/p1-cube/gap_test/outputs/{arm.out_dir_name}"
    train_log = f"{target.remote_repo}/p1-cube/gap_test/outputs/{arm.output_model}.log"
    script = f"""
set -euo pipefail
BUNDLE={q(bundle)}
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/checkpoint" "$BUNDLE/run/checkpoints" "$BUNDLE/outputs" "$BUNDLE/logs"
copy_if_exists() {{
  src="$1"; dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}}
copy_if_exists {q(arm.checkpoint_dir + '/weights_epoch_100.pt')} "$BUNDLE/checkpoint/weights_epoch_100.pt"
copy_if_exists {q(arm.checkpoint_dir + '/config.json')} "$BUNDLE/checkpoint/config.json"
copy_if_exists {q(arm.checkpoint_dir + '/config.yaml')} "$BUNDLE/checkpoint/config.yaml"
copy_if_exists {q(metrics_csv)} "$BUNDLE/run/metrics.csv"
copy_if_exists "$(dirname {q(metrics_csv)})/checkpoints/last.ckpt" "$BUNDLE/run/checkpoints/last.ckpt"
if [ -d {q(out_dir)} ]; then
  mkdir -p "$BUNDLE/outputs/{arm.out_dir_name}"
  cp -a {q(out_dir)}/. "$BUNDLE/outputs/{arm.out_dir_name}/"
fi
copy_if_exists {q(SPLIT_JSON)} "$BUNDLE/outputs/heldout_readout_split_seed2026.json"
copy_if_exists {q(train_log)} "$BUNDLE/logs/train.log"
copy_if_exists {q(target.remote_repo + '/p1-cube/gap_test/README.md')} "$BUNDLE/README.remote.md"
printf '%s\\n' {q(arm.title)} > "$BUNDLE/RUN_LABEL.txt"
(cd "$BUNDLE" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
echo "$BUNDLE"
"""
    remote_bundle = run_remote(target, script, timeout=60 * 30).strip().splitlines()[-1]
    log(f"{arm.key}: remote bundle staged at {remote_bundle}")
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
        log("local remote-manifest verification failed; retrying rsync")
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


def write_local_manifest(local_dir: Path) -> Path:
    manifest = local_dir / "LOCAL_SHA256SUMS.txt"
    files = [
        p
        for p in sorted(local_dir.rglob("*"))
        if p.is_file() and p.name not in {"LOCAL_SHA256SUMS.txt"}
    ]
    lines = [f"{sha256_file(p)}  {p.relative_to(local_dir)}" for p in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        digest, rel = line.split("  ", 1)
        if sha256_file(local_dir / rel) != digest:
            raise RuntimeError(f"local manifest verification failed for {rel}")
    log(f"local manifest verification passed for {local_dir.name}")
    return manifest


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_paths(bundle: Path, arm: ArmRun) -> dict[str, Path]:
    out = bundle / "outputs" / arm.out_dir_name
    return {
        "curves": out / f"{arm.artifact_prefix}_training_curves_summary.json",
        "readout": out / f"{arm.artifact_prefix}_readout_summary.json",
    }


def load_record(bundle: Path, arm: ArmRun) -> dict[str, Any]:
    paths = result_paths(bundle, arm)
    curves = load_json(paths["curves"]) if paths["curves"].exists() else {}
    readout = load_json(paths["readout"]) if paths["readout"].exists() else {}
    return {
        "arm": arm.key,
        "title": arm.title,
        "bundle": str(bundle),
        "created_at_utc": utc_now(),
        "curves": curves,
        "readout": readout,
    }


def save_state_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state = {"records": []}
    if STATE_PATH.exists():
        state = load_json(STATE_PATH)
    records = [r for r in state.get("records", []) if r.get("arm") != record.get("arm")]
    records.append(record)
    state["records"] = records
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return records


def metric_value(row: dict[str, Any] | None, key: str) -> str:
    if not row or row.get(key) is None:
        return "n/a"
    return f"{float(row[key]):.6f}"


def render_record(record: dict[str, Any]) -> str:
    curves = record.get("curves", {})
    readout = record.get("readout", {})
    headline = readout.get("headline", {})
    comparison = headline.get("probe_comparison", {})
    baseline = headline.get("mean_action_baseline", {})
    action_3 = headline.get("action_3")
    plateau = curves.get("plateau", {})
    per_dim = readout.get("metrics", {}).get("test", {}).get("readout_mlp", {}).get("per_dim", [])
    lines = [
        f"### {record['title']}",
        "",
        f"- Local verified bundle: `{Path(record['bundle']).relative_to(REPO_ROOT)}`",
        f"- Training epochs in CSV: `{curves.get('epochs_in_csv', 'n/a')}`",
        f"- Fit prediction plateaued: `{plateau.get('fit_pred_loss', {}).get('plateaued', False)}`; last `{metric_value(plateau.get('fit_pred_loss'), 'last')}`",
        f"- Fit SIGReg plateaued: `{plateau.get('fit_sigreg_loss', {}).get('plateaued', False)}`; at floor `{curves.get('sigreg_at_floor', False)}`; last `{metric_value(plateau.get('fit_sigreg_loss'), 'last')}`",
        f"- Validation prediction plateaued: `{plateau.get('validate_pred_loss', {}).get('plateaued', False)}`; validation SIGReg plateaued: `{plateau.get('validate_sigreg_loss', {}).get('plateaued', False)}`",
        f"- Test R2 all dims: `{headline.get('test_r2_all_dims', float('nan')):.6f}`; excluding action_3: `{headline.get('test_r2_excluding_action_3', float('nan')):.6f}`",
        f"- Test normalized MSE all dims: readout `{headline.get('test_train_variance_normalized_mse_all_dims', float('nan')):.6f}` vs mean baseline `{baseline.get('test_train_variance_normalized_mse_all_dims', float('nan')):.6f}`",
        f"- Test normalized MSE excluding action_3: readout `{headline.get('test_train_variance_normalized_mse_excluding_action_3', float('nan')):.6f}` vs mean baseline `{baseline.get('test_train_variance_normalized_mse_excluding_action_3', float('nan')):.6f}`",
    ]
    if comparison:
        lines.append(
            f"- Probe comparison: Probe all-dim R2 `0.435415`, delta `{comparison.get('delta_vs_probe_all_dims', float('nan')):.6f}`; "
            f"Probe excluding-action_3 R2 `{comparison.get('probe_test_r2_excluding_action_3', PROBE_R2_NO_ACTION_3):.6f}`, "
            f"delta `{comparison.get('delta_vs_probe_excluding_action_3', float('nan')):.6f}`; "
            f"large discrepancy excluding action_3 `{comparison.get('large_discrepancy_vs_probe_excluding_action_3', False)}`"
        )
    if action_3:
        lines.append(
            f"- action_3 separately: R2 `{action_3.get('r2', float('nan')):.6f}`, "
            f"normalized MSE `{action_3.get('train_variance_normalized_mse', float('nan')):.6f}`"
        )
    lines.extend(["", "| dim | R2 | normalized MSE |", "| --- | ---: | ---: |"])
    for row in per_dim:
        lines.append(
            f"| {row.get('name')} | {row.get('r2', float('nan')):.6f} | "
            f"{row.get('train_variance_normalized_mse', float('nan')):.6f} |"
        )
    return "\n".join(lines)


def update_readme(records: list[dict[str, Any]]) -> None:
    existing = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    block = "\n\n".join(render_record(r) for r in sorted(records, key=lambda x: x.get("arm", "")))
    replacement = f"{README_START}\n\n## Overnight Vast 8x H100 results\n\n{block}\n\n{README_END}\n"
    if README_START in existing and README_END in existing:
        before = existing.split(README_START, 1)[0].rstrip()
        after = existing.split(README_END, 1)[1].lstrip()
        updated = f"{before}\n\n{replacement}\n{after}"
    else:
        updated = existing.rstrip() + "\n\n" + replacement
    README_PATH.write_text(updated, encoding="utf-8")
    log(f"updated README: {README_PATH}")


def copy_readme_into_bundle(bundle: Path) -> None:
    dst = bundle / "README.local.md"
    dst.write_text(README_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def arm_a_allows_arm_b(record: dict[str, Any]) -> tuple[bool, str]:
    curves = record.get("curves", {})
    readout = record.get("readout", {})
    comparison = readout.get("headline", {}).get("probe_comparison", {})
    if not (curves.get("both_plateaued") and curves.get("sigreg_at_floor")):
        return False, "Arm A fit curves did not clearly plateau with SIGReg at floor"
    if not readout.get("gate", {}).get("passed", False):
        return False, "Arm A readout did not beat mean-action baseline"
    if comparison.get("large_discrepancy_vs_probe_excluding_action_3", True):
        return False, "Arm A readout is grossly discrepant from Probe excluding action_3"
    return True, "Arm A passed plateau/readout gates"


def stop_instance(target: VastTarget, *, dry_run: bool) -> None:
    if dry_run:
        log("dry-run stop requested; leaving Vast instance running")
        return
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise RuntimeError("VAST_API_KEY is not set locally; refusing to claim the instance was stopped")
    url = f"https://console.vast.ai/api/v0/instances/{target.instance_id}/"
    body = json.dumps({"state": "stopped"}).encode("utf-8")
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
            log(f"Vast stop response status={resp.status} body={payload[:300]}")
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vast stop failed status={exc.code} body={payload[:500]}") from exc


def backup_verify_update(target: VastTarget, arm: ArmRun, metrics_csv: str) -> tuple[Path, dict[str, Any]]:
    remote_bundle = create_remote_bundle(target, arm, metrics_csv)
    local_bundle = rsync_bundle(target, remote_bundle)
    record = load_record(local_bundle, arm)
    records = save_state_record(record)
    update_readme(records)
    copy_readme_into_bundle(local_bundle)
    write_local_manifest(local_bundle)
    return local_bundle, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="root@212.13.234.26")
    parser.add_argument("--port", type=int, default=29769)
    parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "vast_inversedynamics")
    parser.add_argument("--instance-id", type=int, default=41405880)
    parser.add_argument("--remote-repo", default=REMOTE_REPO)
    parser.add_argument("--remote-python", default="/venv/main/bin/python")
    parser.add_argument("--poll-seconds", type=int, default=600)
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

    log("overnight guard started")
    log("waiting for Arm A to finish without interrupting it")
    arm_a_metrics = wait_for_clean_training(target, ARM_A, args.poll_seconds)
    run_remote_pipeline(target, ARM_A, arm_a_metrics)
    arm_a_bundle, arm_a_record = backup_verify_update(target, ARM_A, arm_a_metrics)
    log(f"Arm A verified backup: {arm_a_bundle}")

    allow_b, reason = arm_a_allows_arm_b(arm_a_record)
    log(f"Arm B decision: {allow_b} ({reason})")
    if not allow_b:
        stop_instance(target, dry_run=args.dry_run_stop)
        return 0

    launch_arm_b(target)
    time.sleep(120)
    arm_b_metrics = wait_for_clean_training(target, ARM_B, args.poll_seconds)
    run_remote_pipeline(target, ARM_B, arm_b_metrics)
    arm_b_bundle, _ = backup_verify_update(target, ARM_B, arm_b_metrics)
    log(f"Arm B verified backup: {arm_b_bundle}")
    stop_instance(target, dry_run=args.dry_run_stop)
    log("overnight guard finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
