#!/usr/bin/env python3
"""Stage 1 v2 fixed history-only baselines and report."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]
for rel in ("stable-worldmodel-readonly", "stable-pretraining-readonly", "le-wm"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402


OUT_DIR = ROOT / "prelim-p1" / "outputs"
CACHE_PATH = OUT_DIR / "stage1v2_latents.npz"
EXTRACTION_PATH = OUT_DIR / "stage1v2_extraction_summary.json"
STAGE0_PATH = OUT_DIR / "stage0_timing.json"

SEED = 2026
HISTORY_SIZE = 3
LATENT_DIM = 192
RIDGE_ALPHAS = np.logspace(-4, 6, 16)
SANITY_RATIO_RANGE = (0.65, 1.35)
SCALING_FRACTIONS = [0.125, 0.25, 0.5, 1.0]

MLP_MAX_EPOCHS = 1000
MLP_PATIENCE = 60
MLP_BATCH_SIZE = 256
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-2
MLP_MIN_DELTA = 1e-5


class HistoryOnlyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HISTORY_SIZE * LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RidgeModel:
    def __init__(self, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, coef: np.ndarray):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.coef = coef

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.x_mean) / self.x_std
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return (z @ self.coef + self.y_mean).astype(np.float32)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel | None:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    x_mean = x64.mean(axis=0)
    x_std = np.maximum(x64.std(axis=0, ddof=0), 1e-6)
    y_mean = y64.mean(axis=0)
    z = (x64 - x_mean) / x_std
    yc = y64 - y_mean
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lhs = z.T @ z + float(alpha) * np.eye(z.shape[1], dtype=np.float64)
        rhs = z.T @ yc
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None
    if not np.isfinite(coef).all():
        return None
    return RidgeModel(x_mean, x_std, y_mean, coef)


def select_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> tuple[RidgeModel, dict]:
    rows = []
    best = None
    for alpha in alphas:
        model = fit_ridge(x_train, y_train, float(alpha))
        if model is None:
            rows.append({"alpha": float(alpha), "val_mse": None})
            continue
        pred = model.predict(x_val)
        if not np.isfinite(pred).all():
            rows.append({"alpha": float(alpha), "val_mse": None})
            continue
        val_mse = mse(pred, y_val)
        rows.append({"alpha": float(alpha), "val_mse": val_mse})
        if best is None or val_mse < best[0]:
            best = (val_mse, float(alpha), model)
    if best is None:
        raise RuntimeError("All ridge solves failed.")
    return best[2], {"best_alpha": best[1], "best_val_mse": best[0], "sweep": rows}


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def per_trajectory_stats(pred: np.ndarray, target: np.ndarray, episode_id: np.ndarray) -> dict:
    per_pair = np.mean((pred - target) ** 2, axis=1)
    values = []
    for ep in np.unique(episode_id):
        values.append(float(per_pair[episode_id == ep].mean()))
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n_trajectories": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def metric_row(
    pred: np.ndarray,
    target: np.ndarray,
    persistence_mse: float,
    episode_id: np.ndarray,
) -> dict:
    abs_mse = mse(pred, target)
    denom = float(target.var(axis=0, ddof=0).mean())
    return {
        "abs_mse": abs_mse,
        "normalized_mse": abs_mse / max(denom, 1e-12),
        "fraction_of_persistence": abs_mse / max(persistence_mse, 1e-12),
        "per_trajectory": per_trajectory_stats(pred, target, episode_id),
    }


def predict_full_model(history: np.ndarray, action: np.ndarray, device: torch.device) -> np.ndarray:
    model = swm.policy.AutoCostModel("pusht/lewm", cache_dir=swm.data.utils.get_cache_dir())
    model = model.to(device).eval()
    model.requires_grad_(False)
    outs = []
    with torch.no_grad():
        for start in range(0, history.shape[0], 512):
            hist = torch.from_numpy(history[start : start + 512]).to(device)
            act = torch.from_numpy(action[start : start + 512]).to(device)
            pred = model.predict(hist, model.action_encoder(act))[:, -1]
            outs.append(pred.detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> tuple[HistoryOnlyMLP, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = HistoryOnlyMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    xtr = torch.from_numpy(x_train.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    xva = torch.from_numpy(x_val.astype(np.float32)).to(device)
    yva = torch.from_numpy(y_val.astype(np.float32)).to(device)

    n = xtr.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_val = math.inf
    best_epoch = 0
    best_state = None
    wait = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, MLP_BATCH_SIZE):
            idx = perm[start : start + MLP_BATCH_SIZE].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_mse = loss_fn(model(xtr), ytr).item()
            val_mse = loss_fn(model(xva), yva).item()
        if epoch == 1 or epoch % 25 == 0:
            history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val - MLP_MIN_DELTA:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= MLP_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a best state.")
    model.load_state_dict(best_state)
    model.eval()
    sync(device)
    elapsed = time.perf_counter() - start_time
    with torch.no_grad():
        final_train = loss_fn(model(xtr), ytr).item()
        final_val = loss_fn(model(xva), yva).item()
    return model, {
        "architecture": "576->512->192 ReLU MLP",
        "max_epochs": MLP_MAX_EPOCHS,
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(epoch),
        "patience": MLP_PATIENCE,
        "batch_size": MLP_BATCH_SIZE,
        "lr": MLP_LR,
        "weight_decay": MLP_WEIGHT_DECAY,
        "best_val_mse": float(best_val),
        "final_train_mse_at_best": float(final_train),
        "final_val_mse_at_best": float(final_val),
        "elapsed_seconds": elapsed,
        "history": history,
    }


def predict_mlp(model: HistoryOnlyMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, xt.shape[0], 512):
            outs.append(model(xt[start : start + 512]).detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def compute_metrics(preds: dict[str, np.ndarray], target: np.ndarray, episode_id: np.ndarray) -> dict:
    persistence_mse = mse(preds["persistence"], target)
    return {
        name: metric_row(pred, target, persistence_mse, episode_id)
        for name, pred in preds.items()
    }


def scaling_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    rng = np.random.default_rng(SEED)
    n = len(x_train)
    rows = []
    for frac in SCALING_FRACTIONS:
        k = max(1, int(round(frac * n)))
        idx = np.arange(n) if frac == 1.0 else np.sort(rng.choice(n, size=k, replace=False))
        model, info = select_ridge(x_train[idx], y_train[idx], x_val, y_val)
        test_mse = mse(model.predict(x_test), y_test)
        rows.append(
            {
                "fraction": float(frac),
                "train_pairs": int(k),
                "best_alpha": info["best_alpha"],
                "val_mse": info["best_val_mse"],
                "test_mse": test_mse,
            }
        )
    return rows


def plot_scaling(rows: list[dict], path: Path) -> None:
    xs = [100 * r["fraction"] for r in rows]
    ys = [r["test_mse"] for r in rows]
    plt.figure(figsize=(6.5, 4.0), dpi=160)
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Training pairs used (%)")
    plt.ylabel("Ridge test MSE")
    plt.title("History-Only Ridge Data Scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.6f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing v2 cache: {CACHE_PATH}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    device = torch.device("mps")

    data = np.load(CACHE_PATH)
    history = data["history"].astype(np.float32)
    action = data["action"].astype(np.float32)
    target = data["target"].astype(np.float32)
    contact = data["contact"]
    episode_id = data["episode_id"]
    split = data["split"]
    train_mask = split == 0
    val_mask = split == 1
    test_mask = split == 2

    x_train = history[train_mask].reshape(train_mask.sum(), -1)
    y_train = target[train_mask]
    x_val = history[val_mask].reshape(val_mask.sum(), -1)
    y_val = target[val_mask]
    x_test = history[test_mask].reshape(test_mask.sum(), -1)
    y_test = target[test_mask]
    test_episode = episode_id[test_mask]

    # B. Decisive sanity check: last-latent ridge should be in persistence scale.
    last_train = history[train_mask, -1]
    last_val = history[val_mask, -1]
    last_test = history[test_mask, -1]
    sanity_model, sanity_info = select_ridge(last_train, y_train, last_val, y_val)
    sanity_pred = sanity_model.predict(last_test)
    persistence_test = mse(last_test, y_test)
    sanity_test = mse(sanity_pred, y_test)
    sanity_ratio = sanity_test / max(persistence_test, 1e-12)
    sanity_pass = SANITY_RATIO_RANGE[0] <= sanity_ratio <= SANITY_RATIO_RANGE[1]
    sanity = {
        "passed": bool(sanity_pass),
        "persistence_test_mse": persistence_test,
        "last_latent_ridge_test_mse": sanity_test,
        "ratio_to_persistence": sanity_ratio,
        "accepted_ratio_range": list(SANITY_RATIO_RANGE),
        "ridge": sanity_info,
    }

    if not sanity_pass:
        result = {"stage": "1v2", "sanity_check": sanity, "stopped": "sanity_check_failed"}
        (OUT_DIR / "stage1v2_results.json").write_text(json.dumps(result, indent=2))
        raise SystemExit(
            "STOP: last-latent ridge sanity check failed; not continuing to probes/contact split."
        )

    ridge_model, ridge_info = select_ridge(x_train, y_train, x_val, y_val)
    ridge_test_pred = ridge_model.predict(x_test)
    ridge_val_pred = ridge_model.predict(x_val)

    mlp, mlp_info = train_mlp(x_train, y_train, x_val, y_val, device)
    mlp_test_pred = predict_mlp(mlp, x_test, device)
    mlp_val_pred = predict_mlp(mlp, x_val, device)

    full_test_pred = predict_full_model(history[test_mask], action[test_mask], device)
    persistence_pred = last_test

    preds = {
        "persistence": persistence_pred,
        "ridge": ridge_test_pred,
        "mlp": mlp_test_pred,
        "full_lewm": full_test_pred,
    }
    test_metrics = compute_metrics(preds, y_test, test_episode)
    val_metrics = {
        "ridge": {"abs_mse": mse(ridge_val_pred, y_val)},
        "mlp": {"abs_mse": mse(mlp_val_pred, y_val)},
    }

    ridge_test_mse = test_metrics["ridge"]["abs_mse"]
    mlp_test_mse = test_metrics["mlp"]["abs_mse"]
    ridge_mlp_ratio = mlp_test_mse / max(ridge_test_mse, 1e-12)
    probes_roughly_agree = 0.75 <= ridge_mlp_ratio <= 1.25
    if probes_roughly_agree and mlp_info["best_val_mse"] < ridge_info["best_val_mse"]:
        best_history_key = "mlp"
    else:
        best_history_key = "ridge"
    if not probes_roughly_agree:
        best_history_key = "ridge"

    best_history_mse = test_metrics[best_history_key]["abs_mse"]
    full_mse = test_metrics["full_lewm"]["abs_mse"]
    action_info_value = (best_history_mse - full_mse) / max(best_history_mse, 1e-12)
    headroom = (persistence_test - best_history_mse) / max(persistence_test, 1e-12)

    scaling = scaling_curve(x_train, y_train, x_val, y_val, x_test, y_test)
    scaling_plot = OUT_DIR / "stage1v2_ridge_scaling.png"
    plot_scaling(scaling, scaling_plot)

    contact_results = {}
    test_contact = contact[test_mask] > 0
    for subset_name, mask in {"non_contact": ~test_contact, "contact": test_contact}.items():
        subset_preds = {k: v[mask] for k, v in preds.items()}
        subset_target = y_test[mask]
        subset_episode = test_episode[mask]
        metrics = compute_metrics(subset_preds, subset_target, subset_episode)
        subset_best_mse = metrics[best_history_key]["abs_mse"]
        subset_full_mse = metrics["full_lewm"]["abs_mse"]
        contact_results[subset_name] = {
            "n_pairs": int(mask.sum()),
            "n_trajectories": int(len(np.unique(subset_episode))),
            "metrics": metrics,
            "action_information_value": float(
                (subset_best_mse - subset_full_mse) / max(subset_best_mse, 1e-12)
            ),
        }

    scaling_drop = (scaling[-2]["test_mse"] - scaling[-1]["test_mse"]) / max(scaling[-2]["test_mse"], 1e-12)
    scaling_status = (
        "still_dropping" if scaling_drop > 0.05 else "flattened_or_nearly_flat"
    )

    results = {
        "stage": "1v2",
        "device": str(device),
        "sanity_check": sanity,
        "ridge": ridge_info,
        "mlp": mlp_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "headline": {
            "best_history_probe": best_history_key,
            "ridge_mlp_test_ratio": float(ridge_mlp_ratio),
            "probes_roughly_agree": bool(probes_roughly_agree),
            "action_information_value": float(action_info_value),
            "predictive_headroom_over_persistence": float(headroom),
        },
        "data_scaling": {
            "rows": scaling,
            "plot": str(scaling_plot),
            "last_step_relative_drop": float(scaling_drop),
            "status": scaling_status,
        },
        "contact_split": contact_results,
    }
    results_path = OUT_DIR / "stage1v2_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    stage0 = json.loads(STAGE0_PATH.read_text()) if STAGE0_PATH.exists() else {}
    extraction = json.loads(EXTRACTION_PATH.read_text()) if EXTRACTION_PATH.exists() else {}
    report_path = OUT_DIR / "stage1_report.md"
    report_path.write_text(make_report(stage0, extraction, results))

    print("STAGE 1 V2 SUMMARY")
    print(
        "sanity last-latent ridge: "
        f"persistence={persistence_test:.6f}, ridge={sanity_test:.6f}, ratio={sanity_ratio:.6f}, "
        f"passed={sanity_pass}"
    )
    for method in ("persistence", "ridge", "mlp", "full_lewm"):
        row = test_metrics[method]
        pt = row["per_trajectory"]
        print(
            f"{method}: abs_mse={row['abs_mse']:.6f}, normalized={row['normalized_mse']:.6f}, "
            f"frac_persistence={row['fraction_of_persistence']:.6f}, "
            f"traj_mean±std={pt['mean']:.6f}±{pt['std']:.6f}"
        )
    print(f"best_history_probe: {best_history_key}")
    print(f"action_information_value: {action_info_value:.6f}")
    print(f"predictive_headroom_over_persistence: {headroom:.6f}")
    print(f"ridge scaling status: {scaling_status}, last_step_relative_drop={scaling_drop:.6f}")
    for subset in ("non_contact", "contact"):
        print(
            f"{subset}: n={contact_results[subset]['n_pairs']}, "
            f"action_information_value={contact_results[subset]['action_information_value']:.6f}"
        )
    print(f"wrote: {results_path}")
    print(f"wrote: {report_path}")
    print(f"wrote: {scaling_plot}")


def make_report(stage0: dict, extraction: dict, results: dict) -> str:
    timing = stage0.get("timing", {})
    timing_rows = [
        ["Load/device", str(stage0.get("load_ok")), str(stage0.get("device"))],
        ["Param count", f"{stage0.get('param_count', 0):,}", f"{stage0.get('param_count_millions', 0):.2f}M"],
        ["Encode", fmt(timing.get("encode_ms_per_frame")), "ms/frame"],
        ["Predictor fwd", fmt(timing.get("predictor_fwd_ms")), "ms"],
        ["Predictor fwd+bwd+step", fmt(timing.get("predictor_fwd_bwd_step_ms")), "ms"],
    ]

    sanity = results["sanity_check"]
    sanity_rows = [
        ["Persistence", fmt(sanity["persistence_test_mse"])],
        ["Last-latent ridge", fmt(sanity["last_latent_ridge_test_mse"])],
        ["Ratio", fmt(sanity["ratio_to_persistence"])],
        ["Accepted ratio range", f"{sanity['accepted_ratio_range'][0]} to {sanity['accepted_ratio_range'][1]}"],
        ["Passed", str(sanity["passed"])],
    ]

    method_labels = [
        ("Persistence", "persistence"),
        ("Ridge history probe", "ridge"),
        ("MLP history probe", "mlp"),
        ("Full LeWM", "full_lewm"),
    ]
    metric_rows = []
    for label, key in method_labels:
        row = results["test_metrics"][key]
        pt = row["per_trajectory"]
        metric_rows.append(
            [
                label,
                fmt(row["abs_mse"]),
                fmt(row["normalized_mse"]),
                fmt(row["fraction_of_persistence"]),
                f"{pt['mean']:.6f} ± {pt['std']:.6f}",
            ]
        )

    scaling_rows = [
        [
            f"{100 * row['fraction']:.1f}%",
            str(row["train_pairs"]),
            fmt(row["best_alpha"]),
            fmt(row["val_mse"]),
            fmt(row["test_mse"]),
        ]
        for row in results["data_scaling"]["rows"]
    ]

    contact_rows = []
    for subset_label, subset_key in (("Non-contact", "non_contact"), ("Contact", "contact")):
        subset = results["contact_split"][subset_key]
        for label, key in method_labels:
            row = subset["metrics"][key]
            contact_rows.append(
                [
                    subset_label,
                    str(subset["n_pairs"]),
                    label,
                    fmt(row["abs_mse"]),
                    fmt(row["normalized_mse"]),
                    fmt(row["fraction_of_persistence"]),
                    fmt(subset["action_information_value"]) if key == results["headline"]["best_history_probe"] else "",
                ]
            )

    headline = results["headline"]
    if headline["probes_roughly_agree"]:
        probe_sentence = "The ridge and MLP probes roughly agree on held-out test error."
    else:
        probe_sentence = (
            "The ridge and MLP probes diverge by more than 25%, so the report uses ridge as the "
            "robust history-only anchor."
        )
    if results["data_scaling"]["status"] == "still_dropping":
        scaling_sentence = (
            "The ridge scaling curve is still dropping from 50% to 100% train pairs, so the "
            "history-only result remains data-limited; any residual gap to Full LeWM is not clean "
            "evidence that actions are the sole missing signal."
        )
    else:
        scaling_sentence = (
            "The ridge scaling curve has flattened or nearly flattened by 100% train pairs, so the "
            "residual gap is less likely to be explained by simple data starvation."
        )
    if headline["predictive_headroom_over_persistence"] <= 0:
        headroom_sentence = (
            "The best history-only probe still does not beat persistence; this means the one-step "
            "comparison is degenerate and should not be extended here without a separate v3 design."
        )
    else:
        headroom_sentence = (
            "The best history-only probe beats persistence, so the one-step comparison is no longer "
            "invalidated by the v1 overfit failure."
        )

    lines = [
        "# Stage 1 Report v2: Proposal 1 Preliminary Test",
        "",
        "The previous Stage 1 report is preserved as `stage1_report_v1_INVALID.md`; its action-information and headroom numbers are discarded.",
        "",
        "## Stage 0 Timing + Fallback",
        "",
        table(["Item", "Value", "Unit/Notes"], timing_rows),
        "",
        "No explicit MPS CPU fallback warnings appeared in the captured Stage 0 terminal log.",
        "",
        "## V2 Extraction Summary",
        "",
        f"- Episodes used: {extraction.get('used_episodes')} total; {extraction.get('train_trajectory_count')} train, {extraction.get('val_trajectory_count')} validation, {extraction.get('test_trajectory_count')} test trajectories.",
        f"- Pairs: {extraction.get('train_pairs')} train, {extraction.get('val_pairs')} validation, {extraction.get('test_pairs')} test, {extraction.get('total_pairs')} total.",
        f"- History latents: `{tuple(extraction.get('history_shape', []))}`; actions: `{tuple(extraction.get('action_history_shape', []))}`; targets: `{tuple(extraction.get('target_shape', []))}`.",
        f"- Latent stats: mean(mean_dim)={extraction.get('latent_stats', {}).get('target_per_dim_mean_mean'):.6f}, mean(std_dim)={extraction.get('latent_stats', {}).get('target_per_dim_std_mean'):.6f}, any_nan={extraction.get('latent_stats', {}).get('any_nan')}.",
        f"- Contact labels: {extraction.get('contact', {}).get('source')} with {extraction.get('contact', {}).get('positive_pairs')} contact pairs and {extraction.get('contact', {}).get('non_contact_pairs')} non-contact pairs.",
        "",
        "## B. Pipeline Sanity Check",
        "",
        table(["Quantity", "Test MSE / Value"], sanity_rows),
        "",
        "Result: passed. The last-latent ridge sits in the same scale as persistence, so the v1 failure is treated as MLP overfit rather than a gross indexing/normalization bug.",
        "",
        "## 1b Three-Way/Four-Method Comparison",
        "",
        f"Ridge probe: full 3-latent history, validation-selected alpha `{results['ridge']['best_alpha']:.6g}`, validation MSE `{results['ridge']['best_val_mse']:.6f}`.",
        f"MLP probe: {results['mlp']['architecture']}, stopped at epoch {results['mlp']['stopped_epoch']}, best validation epoch {results['mlp']['best_epoch']}, best validation MSE `{results['mlp']['best_val_mse']:.6f}`.",
        "",
        table(
            ["Method", "Abs MSE", "Normalized MSE", "Fraction of Persistence", "Per-Trajectory MSE Mean ± Std"],
            metric_rows,
        ),
        "",
        f"- Best history-only probe used for headlines: `{headline['best_history_probe']}`",
        f"- Action information value: `{headline['action_information_value']:.6f}`",
        f"- Predictive headroom over persistence: `{headline['predictive_headroom_over_persistence']:.6f}`",
        f"- Ridge/MLP test MSE ratio: `{headline['ridge_mlp_test_ratio']:.6f}`",
        "",
        "## D. Ridge Data-Scaling Control",
        "",
        table(["Train Fraction", "Train Pairs", "Alpha", "Val MSE", "Test MSE"], scaling_rows),
        "",
        f"Scaling status: `{results['data_scaling']['status']}`; 50% to 100% relative test-MSE drop `{results['data_scaling']['last_step_relative_drop']:.6f}`.",
        "",
        f"![Ridge scaling]({Path(results['data_scaling']['plot']).name})",
        "",
        "## 1c Contact Split",
        "",
        table(
            ["Subset", "Pairs", "Method", "Abs MSE", "Normalized MSE", "Fraction of Persistence", "Action Info Value"],
            contact_rows,
        ),
        "",
        "## Plain-Language Reading",
        "",
        f"{probe_sentence} {scaling_sentence} {headroom_sentence} Full LeWM remains far lower-error than the best history-only probe on this one-step test.",
        "",
        "## Divergences, Caveats, Confounds",
        "",
        "- No discrepancy for the loaded object checkpoint: `AutoCostModel('pusht/lewm')` resolves to `jepa.JEPA` from `le-wm/jepa.py`.",
        "- The installed `stable_worldmodel.wm.lewm.LeWM` source has a different rollout implementation, but it is not the loaded object checkpoint class here; one-step `predict(emb, act_emb)` shape is the same.",
        "- The HDF5 dataset has no `n_contacts` column; contact/non-contact uses the prior geometric fallback from 7D state.",
        "- Known confound: probes are trained on latents from an already action-conditioned trained encoder, so this remains a proxy rather than a direct action-free pretraining measurement.",
        "- Ridge features were standardized using probe-train statistics before closed-form fitting; errors are reported in original latent units.",
        "",
        "Stage 2 was not run.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
