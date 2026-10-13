"""Offline CN7 hardware identification and experimental inverse-response controller.

Train, select and evaluate on disjoint chronological ROUTES. All response tables are
binned medians, with no extrapolation. Recorded commands are the baseline plant
replay; candidate feedback uses the same frozen recorded targets. Neither replay
reruns the vision model, and neither establishes a future closed-loop road outcome.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from lateral_report import band_of

DT = .05
STEER_MAX = 409
MIN_CELL = 100
LAT_EDGES = np.array([-410, -330, -250, -170, -90, -30, 30, 90, 170, 250, 330, 410])
LONG_EDGES = np.array([-3.51, -2.5, -1.5, -.8, -.3, .1, .5, 1., 1.5, 2.01])


def route_split(routes):
    routes = sorted(set(routes))
    if len(routes) < 5:
        raise ValueError("at least five independent routes are required")
    a, b = max(3, int(len(routes) * .6)), max(4, int(len(routes) * .8))
    return {"train": routes[:a], "validation": routes[a:b], "test": routes[b:]}


def lookup(table, speed, command):
    cell = table.get(band_of(float(speed)))
    if cell is None or not np.isfinite(command):
        return float("nan")
    x, y = cell["command"], cell["response"]
    if command < x[0] or command > x[-1]:
        return float("nan")
    return float(np.interp(command, x, y))


def inverse(table, speed, response):
    cell = table.get(band_of(float(speed)))
    if cell is None or not np.isfinite(response):
        return float("nan")
    y, x = np.array(cell["response"]), np.array(cell["command"])
    if y[-1] < y[0]:
        y, x = y[::-1], x[::-1]
    if response < y[0] or response > y[-1]:
        return float("nan")
    return float(np.interp(response, y, x))


def usable_cell_table(command, response, axis):
    if len(command) < 3:
        return False
    delta = np.diff(response)
    if axis == "lat":
        return bool(np.all(delta < 0) and response[0] - response[-1] >= .3)
    return bool(np.all(delta > 0) and response[-1] - response[0] >= .3
                and min(response) < -.1 and max(response) > .1)


def limit_torque(request, previous, dt):
    """Driver-free Hyundai 100 Hz asymmetric limiter, aggregated to the sample grid.

    Driver input is excluded from these experiments. This does not substitute for
    panda safety or test its driver envelope. Ceiling/rates are never optimized.
    """
    if dt <= 0 or not np.isfinite([request, previous, dt]).all():
        raise ValueError("finite command and positive dt required")
    steps = int(round(dt / .01))
    if steps < 1 or abs(steps * .01 - dt) > 1e-8:
        raise ValueError("dt must be a positive multiple of the native 100 Hz period")
    for _ in range(steps):
        if previous > 0:
            low, high = max(previous - 7, -3), previous + 3
        else:
            low, high = previous - 3, min(previous + 7, 3)
        previous = int(round(np.clip(request, max(-409, low), min(409, high))))
    return previous


def integrate(speed, yaw, dt):
    speed, yaw = np.asarray(speed), np.asarray(yaw)
    xy = np.full((len(speed), 2), np.nan)
    xy[0] = 0
    heading = 0.
    for i in range(1, len(speed)):
        if not np.isfinite([speed[i - 1], yaw[i - 1]]).all():
            break
        mid = heading + yaw[i - 1] * dt / 2
        xy[i] = xy[i - 1] + speed[i - 1] * dt * np.array([np.cos(mid), np.sin(mid)])
        heading += yaw[i - 1] * dt
    return xy


def clean_mask(d, axis):
    fields = ["t", "v", "a", "yaw", "roll", "calibrated", "steering_pressed",
              "brake_pressed", "gas_pressed"]
    fields += ["torque", "desired_curvature", "lat_active"] if axis == "lat" else [
        "accel_command", "accel_target", "long_active"]
    m = np.ones(len(d["t"]), dtype=bool)
    for key in fields:
        m &= np.isfinite(d[key])
    for key in ("steering_pressed", "brake_pressed", "gas_pressed"):
        m &= d[key] == 0
    m &= (d["calibrated"] == 1) & (d["v"] >= 3)
    m &= d["lat_active" if axis == "lat" else "long_active"] == 1
    validity = d.get("valid_lat" if axis == "lat" else "valid_long", d.get("valid"))
    if validity is not None:
        m &= validity == 1
    if axis == "lat":
        m &= np.abs(d["torque"]) <= 409
    else:
        m &= (d["accel_command"] >= -3.5) & (d["accel_command"] <= 2)
    if "eps_fault" in d:
        m &= d["eps_fault"] == 0
    return m


def window_indices(d, axis, seconds=3, stride=10):
    n = int(round(seconds / DT)) + 1
    good = clean_mask(d, axis)
    gaps = np.r_[False, np.abs(np.diff(d["t"]) - DT) > .002]
    starts = []
    for i in range(0, max(0, len(good) - n + 1), stride):
        if i >= 12 and np.all(good[i-12:i+n]) and not np.any(gaps[i-11:i+n]):
            starts.append(i)
    return np.array(starts, dtype=int)


def fit_table(records, axis):
    edges = LAT_EDGES if axis == "lat" else LONG_EDGES
    bins = defaultdict(list)
    routes = defaultdict(set)
    excluded = {"unclean": 0, "not_settled": 0, "outside_command_bins": 0}
    for meta, d in records:
        command = d["torque" if axis == "lat" else "accel_command"]
        response = d["v"] * d["yaw"] - 9.81 * d["roll"] if axis == "lat" else d["a"]
        clean = clean_mask(d, axis)
        excluded["unclean"] += int(np.sum(~clean))
        for i in np.where(clean)[0]:
            # A whole .6 s command/effect window must be clean and contiguous.
            lo = i - 12
            if lo < 0 or not np.all(clean[lo:i+1]) or np.any(np.diff(d["t"][lo:i+1]) > .052):
                excluded["not_settled"] += 1
                continue
            if np.ptp(command[lo:i+1]) > (16 if axis == "lat" else .15):
                excluded["not_settled"] += 1
                continue
            b = int(np.searchsorted(edges, command[i], side="right") - 1)
            if not 0 <= b < len(edges) - 1:
                excluded["outside_command_bins"] += 1
                continue
            key = band_of(float(d["v"][i])), b
            bins[key].append((float(command[i]), float(response[i])))
            routes[key].add(meta["route"])
    table, support, rejected = {}, [], []
    by_band = defaultdict(list)
    for (band, b), values in sorted(bins.items()):
        a = np.asarray(values)
        row = {"band": band, "bin": b, "n": len(a), "routes": len(routes[band, b]),
               "command": float(np.median(a[:, 0])), "response": float(np.median(a[:, 1])),
               "response_p10": float(np.percentile(a[:, 1], 10)),
               "response_p90": float(np.percentile(a[:, 1], 90))}
        support.append(row)
        if len(a) >= MIN_CELL and len(routes[band, b]) >= 2:
            by_band[band].append(row)
    for band, rows in by_band.items():
        x, y = [r["command"] for r in rows], [r["response"] for r in rows]
        if usable_cell_table(x, y, axis):
            table[band] = {"command": x, "response": y,
                           "n": [r["n"] for r in rows], "routes": [r["routes"] for r in rows]}
        else:
            rejected.append(band)
    return table, {"cells": support, "excluded": excluded, "rejected_nonmonotonic_bands": rejected}


def rollout(d, start, model, axis, gain=None, seconds=3, speed_override=None):
    n = int(round(seconds / DT)) + 1
    z = slice(start, start+n)
    v = d["v"][z].copy() if speed_override is None else np.array(speed_override, copy=True)
    measured = d["v"][z] * d["yaw"][z] if axis == "lat" else d["a"][z]
    state = np.full(n, np.nan)
    command = np.full(n, np.nan)
    state[0] = measured[0]
    cmd_key = "torque" if axis == "lat" else "accel_command"
    command[0] = d[cmd_key][start]
    delay_n = int(round(model["delay_s"] / DT))
    transition = model.get("kind") == "median_transition"
    alpha = None if transition else DT / (model["tau_s"] + DT)
    history = list(d[cmd_key][max(0, start-delay_n):start])
    if len(history) < delay_n or not np.isfinite(history).all():
        return state, command
    for j in range(n - 1):
        i = start + j
        if gain is None:
            command[j] = d[cmd_key][i]
        else:
            target = d["desired_curvature"][i] * v[j]**2 if axis == "lat" else d["accel_target"][i]
            if transition:
                from controller_dynamics import inverse as transition_inverse, reachable

                corrected = state[j] - 9.81 * d["roll"][i]
                bounds = reachable(model, v[j], corrected)
                if bounds is None:
                    break
                # Saturate the desired state change to observed capacity; commands
                # remain inside measured support and the unchanged native limiter.
                target_next = np.clip(corrected + gain * (target - state[j]), *bounds)
                ask = transition_inverse(model, v[j], corrected, target_next)
            else:
                target += gain * (target - state[j])
                if axis == "lat":
                    target -= 9.81 * d["roll"][i]
                ask = inverse(model["table"], v[j], target)
            if not np.isfinite(ask):
                break
            command[j] = limit_torque(ask, command[max(0, j-1)], DT) if axis == "lat" else np.clip(
                ask, command[max(0, j-1)] - DT * 2.5, command[max(0, j-1)] + DT * 2.5)
        history.append(command[j])
        delayed = history[-delay_n-1]
        if transition:
            from controller_dynamics import forward

            corrected = state[j] - 9.81 * d["roll"][i]
            state[j+1] = forward(model, v[j], corrected, delayed) + 9.81 * d["roll"][i+1]
            if not np.isfinite(state[j+1]):
                break
            continue
        steady = lookup(model["table"], v[j], delayed)
        if not np.isfinite(steady):
            break
        if axis == "lat":
            steady += 9.81 * d["roll"][i]
        state[j+1] = state[j] + alpha * (steady - state[j])
        if axis == "long":
            v[j+1] = max(0, v[j] + state[j] * DT)
    command[-1] = command[-2]
    return state, command


def score_model(records, model, axis, gain=None, stride=40):
    windows = {}
    attempted = 0
    for meta, d in records:
        for i in window_indices(d, axis, stride=stride):
            if i < 12:
                continue
            attempted += 1
            state, _ = rollout(d, i, model, axis, gain)
            if not np.isfinite(state).all():
                continue
            z = slice(i, i+len(state))
            actual = d["v"][z] * d["yaw"][z] if axis == "lat" else d["a"][z]
            target = d["desired_curvature"][z] * d["v"][z]**2 if axis == "lat" else d["accel_target"][z]
            windows[f"{meta['segment']}:{i}"] = {"absolute": np.abs(state - actual).tolist(),
                "target": np.abs(state - target).tolist(), "endpoint": float(abs(state[-1] - actual[-1]))}
    return {"attempted_windows": attempted, "window_scores": windows}


def paired_scores(scores):
    """Compare only identical windows, retaining loss of support in the denominator."""
    common = set.intersection(*(set(s["window_scores"]) for s in scores))
    summaries = []
    for score in scores:
        attempted, used = score["attempted_windows"], len(common)
        if not used:
            summaries.append({"attempted_windows": attempted, "used_windows": 0, "coverage": 0.,
                "mae": None, "p95": None, "target_mae": None, "endpoint_p95": None}
            )
            continue
        residuals = [v for k in sorted(common) for v in score["window_scores"][k]["absolute"]]
        target = [v for k in sorted(common) for v in score["window_scores"][k]["target"]]
        endpoints = [score["window_scores"][k]["endpoint"] for k in sorted(common)]
        summaries.append({"attempted_windows": attempted, "used_windows": used, "coverage": used/max(1, attempted),
            "mae": float(np.mean(residuals)), "p95": float(np.percentile(residuals, 95)),
            "target_mae": float(np.mean(target)), "endpoint_p95": float(np.percentile(endpoints, 95)),
            "window_ids": sorted(common)})
    return summaries


def acceptance(baseline, candidate, identified):
    return bool(identified and baseline["mae"] is not None and candidate["mae"] is not None
                and candidate["mae"] < baseline["mae"] * .95 and candidate["p95"] <= baseline["p95"])


def identify(records, axis):
    table, support = fit_table(records, axis)
    if not table:
        return None, support
    grid, raw_scores = [], []
    for delay in (0., .1, .2, .3, .4, .5, .6):
        for tau in (.1, .2, .35, .5, .8):
            model = {"table": table, "delay_s": delay, "tau_s": tau}
            score = score_model(records, model, axis, stride=120)
            grid.append((delay, tau))
            raw_scores.append(score)
    scores = [(s["mae"], d, t, s) for (d, t), s in zip(grid, paired_scores(raw_scores), strict=True)
              if s["used_windows"] >= 10 and s["coverage"] >= .1]
    support["dynamic_grid"] = [{"delay_s": d, "tau_s": t, **s} for _, d, t, s in scores]
    if not scores:
        return None, support
    _, delay, tau, score = min(scores)
    boundary = delay == .6 or tau in (.1, .8)
    return {"table": table, "delay_s": delay, "tau_s": tau, "boundary_optimum": boundary,
            "training_prediction": score}, support


def identify_transient(records):
    from controller_dynamics import fit

    candidates, raw = [], []
    for delay in (0., .05, .1, .2, .3, .4, .5, .6):
        model = fit(records, delay)
        if model["cells"]:
            candidates.append(model)
            raw.append(score_model(records, model, "lat", stride=120))
    if not candidates:
        return None, {"status": "no_supported_transient_cells"}
    summaries = paired_scores(raw)
    eligible = [(s["mae"], i) for i, s in enumerate(summaries)
                if s["used_windows"] >= 10 and s["coverage"] >= .1]
    support = {"dynamic_grid": [{"delay_s": m["delay_s"], **s} for m, s in zip(candidates, summaries, strict=True)],
               "cell_support": [m["support"] for m in candidates]}
    if not eligible:
        return None, support
    _, i = min(eligible)
    model = candidates[i]
    model.update(boundary_optimum=model["delay_s"] == .6, training_prediction=summaries[i])
    return model, support


def cohort_key(meta):
    material = {k: meta.get(k) for k in ("git_commit", "dirty", "fingerprint", "op_long",
                                        "torque_factor", "torque_friction", "params", "opendbc_gitlink",
                                        "model_identity", "car_params", "car_params_sp")}
    # Refresh timestamps and the catalog of UNSELECTED downloadable models do not
    # identify the active driving model. Preserve the active bundle and runner.
    material["model_identity"] = {k: v for k, v in (meta.get("model_identity") or {}).items()
        if not k.startswith(("ModelManager_LastSyncTime", "ModelManager_ModelsCache", "ModelManager_PrevBundle"))}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:12]


def enrich_route_metadata(paths):
    """Recover static route context where later rlogs omit repeated carParams.

    The route's own earliest recorded configuration is used only with a matching
    source commit/dirty flag. This is labelled static context, never signal filling.
    """
    seeds = {}
    for _, meta in sorted(paths, key=lambda row: row[1].get("segment_index", 0)):
        if meta.get("car_params"):
            seeds.setdefault(meta["route"], meta)
    keys = ("car_params", "car_params_sp", "fingerprint", "op_long", "torque_factor", "torque_friction")
    for _, meta in paths:
        seed = seeds.get(meta["route"])
        if meta.get("car_params") or seed is None or not meta.get("git_commit"):
            continue
        if any(meta.get(k) != seed.get(k) for k in ("git_commit", "dirty")):
            continue
        inherited = []
        for key in keys:
            if key in seed:
                meta[key] = seed[key]
                inherited.append(key)
        meta["route_provenance_source"] = seed["segment"]
        meta["route_context_inherited_fields"] = inherited


def load_records(paths):
    from controller_extract import hold_calibration

    records = []
    for p, meta in paths:
        with np.load(p, allow_pickle=False) as npz:
            records.append((meta, hold_calibration({k: npz[k] for k in npz.files})))
    return records


def save_replay(meta, d, models, gains, output):
    n = len(d["t"])
    trace = {"t": d["t"], "baseline_torque": d["torque"].copy(),
             "tuned_torque": np.full(n, np.nan), "baseline_accel": d["accel_command"].copy(),
             "tuned_accel": np.full(n, np.nan), "baseline_yaw": np.full(n, np.nan),
             "tuned_yaw": np.full(n, np.nan), "baseline_v": d["v"].copy(),
             "tuned_v": np.full(n, np.nan), "long_simulated": np.zeros(n, dtype=bool),
             "diagnostic_only": np.asarray(any(not m.get("identified", False) for m in models.values())),
             "lateral_model_available": np.asarray("lat" in models),
             "longitudinal_model_available": np.asarray("long" in models)}
    forecast_t, gt, base, tuned = [], [], [], []
    for i in window_indices(d, "lat", stride=10):
        m = 61
        z = slice(i, i+m)
        v0, v1 = d["v"][z].copy(), d["v"][z].copy()
        ac0, ac1 = d["accel_command"][z].copy(), np.full(m, np.nan)
        if models.get("long") and np.all(clean_mask(d, "long")[i-12:i+m]):
            a0, ac0 = rollout(d, i, models["long"], "long")
            a1, ac1 = rollout(d, i, models["long"], "long", gains.get("long", 0.))
            v0 = d["v"][i] + np.r_[0, np.cumsum(a0[:-1]) * DT]
            v1 = d["v"][i] + np.r_[0, np.cumsum(a1[:-1]) * DT]
            trace["long_simulated"][i:i+10] = True
        if models.get("lat"):
            lat0, torque0 = rollout(d, i, models["lat"], "lat", speed_override=v0)
            lat1, torque1 = rollout(d, i, models["lat"], "lat", gains.get("lat", 0.), speed_override=v1)
        else:
            lat0, lat1, torque0, torque1 = [np.full(m, np.nan) for _ in range(4)]
        yaw0, yaw1 = lat0 / np.maximum(v0, 1), lat1 / np.maximum(v1, 1)
        forecast_t.append(d["t"][i])
        gt.append(integrate(d["v"][z], d["yaw"][z], DT))
        base.append(integrate(v0, yaw0, DT))
        tuned.append(integrate(v1, yaw1, DT))
        # First .5 s from each reset gives a chronological interaction trace.
        short = slice(i, min(n, i+10))
        trace["baseline_yaw"][short], trace["tuned_yaw"][short] = yaw0[:10], yaw1[:10]
        trace["tuned_torque"][short] = torque1[:10]
        trace["baseline_accel"][short], trace["tuned_accel"][short] = ac0[:10], ac1[:10]
        trace["baseline_v"][short], trace["tuned_v"][short] = v0[:10], v1[:10]
    trace.update(forecast_t=np.asarray(forecast_t), forecast_dt=np.asarray(DT),
                 gt_xy=np.asarray(gt).reshape(-1, 61, 2),
                 baseline_xy=np.asarray(base).reshape(-1, 61, 2),
                 tuned_xy=np.asarray(tuned).reshape(-1, 61, 2))
    np.savez_compressed(output, **trace)
    output.with_suffix(".json").write_text(json.dumps({"segment": meta["segment"], "gains": gains,
        "forecast_count": len(forecast_t), "forecast_seconds": 3,
        "claim": "experimental counterfactual; tuned controller has not driven this footage",
        "baseline": "empirical plant replay of recorded transmitted commands",
        "candidate": "inverse median-response feedback controller; frozen recorded targets",
        "model_status": {k: v.get("status", "untrusted_plant") for k, v in models.items()},
        "longitudinal": "simulated where an empirical model is available and control active; otherwise recorded speed is held fixed"}, indent=2))
    return len(forecast_t)


def resolve_gitlinks(paths, source_repo):
    """Resolve only locally available history; never infer identity or fetch old builds."""
    result = subprocess.run(["git", "-C", str(source_repo), "rev-list", "--all"],
                            check=True, capture_output=True, text=True)
    known = set(result.stdout.splitlines())
    cache = {}
    for _, meta in paths:
        commit = meta.get("git_commit")
        identity = (commit, meta.get("dirty"))
        if identity not in cache:
            cache[identity] = None
            if commit in known and meta.get("dirty") is False:
                tree = subprocess.run(["git", "-C", str(source_repo), "ls-tree", commit, "opendbc_repo"],
                                      check=True, capture_output=True, text=True).stdout.split()
                if len(tree) >= 3 and tree[0] == "160000":
                    cache[identity] = tree[2]
        meta["opendbc_gitlink"] = cache[identity]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", help="explicit provenance cohort; otherwise largest eligible cohort")
    parser.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cohorts, paths = defaultdict(list), []
    for p in sorted(args.dataset.glob("*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        paths.append((p, meta))
    enrich_route_metadata(paths)
    paths = [(p, m) for p, m in paths if m.get("fingerprint") == "HYUNDAI_ELANTRA_2024"]
    resolve_gitlinks(paths, args.source_repo)
    for p, meta in paths:
        cohorts[cohort_key(meta)].append((p, meta))
    catalog = {k: {"segments": len(v), "routes": len({m["route"] for _, m in v}),
                    "example": v[0][1]} for k, v in cohorts.items()}
    for k, v in catalog.items():
        v["tx_at_409_frames"] = sum(m.get("can_counts", {}).get("tx_at_409_frames", 0) for _, m in cohorts[k])
        v["tx_over_409_frames"] = sum(m.get("can_counts", {}).get("tx_over_409_frames", 0) for _, m in cohorts[k])
        v["lat_active_rows"] = sum(m.get("summary", {}).get("lat_active_rows", 0) for _, m in cohorts[k])
    eligible = [k for k, v in catalog.items() if v["routes"] >= 5 and v["lat_active_rows"] >= 1200
                and v["tx_over_409_frames"] == 0]
    if not eligible:
        raise SystemExit("No provenance cohort has five routes; extract more logs before tuning")
    key = args.cohort or max(eligible, key=lambda k: (catalog[k]["tx_at_409_frames"] > 0,
                                                     catalog[k]["lat_active_rows"]))
    if key not in eligible:
        raise SystemExit("Selected cohort has insufficient independent route support")
    paths = cohorts[key]
    route_times = {}
    for _, meta in paths:
        wall = meta.get("init_data", {}).get("wallTimeNanos")
        if not wall:
            raise SystemExit("Recorded wallTimeNanos missing: cannot establish chronological route split")
        route_times[meta["route"]] = min(route_times.get(meta["route"], wall), wall)
    ordered = sorted(route_times, key=lambda r: (route_times[r], r))
    encoded = {f"{i:08d}": r for i, r in enumerate(ordered)}
    split = {s: [encoded[r] for r in routes] for s, routes in route_split(encoded).items()}
    partitions = {s: load_records([(p, m) for p, m in paths if m["route"] in routes])
                  for s, routes in split.items()}
    report = {"vehicle": "Hyundai Elantra 2024 SEL Convenience Package", "steer_max": 409,
              "cohort": key, "cohort_catalog": catalog, "split": split,
              "route_start_wall_time_nanos": route_times,
              "calibration_state_hold_s": .5, "dynamic_service_max_age_s": .2,
              "source_gitlink_known": all(m.get("opendbc_gitlink") for _, m in paths),
              "axes": {}, "deployment_approved": False,
              "limitations": ["Recorded trajectory is measured ego odometry, not lane/safe-path ground truth.",
                  "Candidate receives frozen recorded targets; vision model is not rerun on changed state.",
                  "Baseline replays logged commands; it does not rerun the historical baseline controller.",
                  "Steady median cells require at least 100 samples and two training routes; no extrapolation.",
                  "Driver overrides, stale telemetry, stops, missing calibration and >409 commands are excluded.",
                  "A simulation advantage does not prove improved road driving; no device deployment performed."]}
    models, gains = {}, {}
    for axis in ("lat", "long"):
        print(f"Identifying {axis} from {len(partitions['train'])} training segments", flush=True)
        model, support = identify(partitions["train"], axis)
        if axis == "lat" and model is None:
            model, transient_support = identify_transient(partitions["train"])
            support["transient"] = transient_support
        axis_report = {"support": support, "model": model, "status": "insufficient_identification"}
        report["axes"][axis] = axis_report
        if model is None:
            continue
        gain_grid = (0., .05, .1, .2)
        val_raw = [score_model(partitions["validation"], model, axis)]
        val_raw += [score_model(partitions["validation"], model, axis, g) for g in gain_grid]
        paired = paired_scores(val_raw)
        val_base = paired[0]
        candidates = dict(zip(map(str, gain_grid), paired[1:], strict=True))
        supported = [(float(g), s) for g, s in candidates.items()
                     if s["used_windows"] >= 10 and s["target_mae"] is not None]
        gain = min(supported, key=lambda item: item[1]["target_mae"])[0] if supported else 0.
        test_base, test_candidate = paired_scores([score_model(partitions["test"], model, axis),
                                                  score_model(partitions["test"], model, axis, gain)])
        identified = (bool(supported) and not model["boundary_optimum"] and val_base["used_windows"] >= 20
                      and val_base["coverage"] >= .5 and val_base["mae"] is not None
                      and val_base["mae"] <= (.25 if axis == "lat" else .3))
        # Candidate selection uses validation only. Test is reported once, after selection.
        axis_report.update(validation_baseline=val_base, validation_candidates=candidates,
                           feedback_gain=gain, test_baseline=test_base, test_candidate=test_candidate,
                           identified=identified, status="offline_candidate" if identified else "untrusted_plant")
        model["status"] = axis_report["status"]
        model["identified"] = identified
        model["recorded_build_gitlink_known"] = report["source_gitlink_known"]
        models[axis], gains[axis] = model, gain
    replays = []
    for meta, d in partitions["test"]:
        out = args.out / f"{meta['segment']}.replay.npz"
        count = save_replay(meta, d, models, gains, out)
        replays.append({"segment": meta["segment"], "path": str(out), "forecasts": count})
    report["replays"] = replays
    report["status"] = "experimental_offline_only"
    dest = args.out / "tuning-report.json"
    dest.write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.out / "candidate.json").write_text(json.dumps({"steer_max": 409, "models": models,
        "feedback_gains": gains, "production_enabled": False,
        "warning": "Offline research artifact. See tuning-report.json identification gates and limitations."}, indent=2))
    print(json.dumps({"report": str(dest), "cohort": key, "routes": split,
                      "axes": {k: v["status"] for k, v in report["axes"].items()},
                      "replays": len(replays)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
