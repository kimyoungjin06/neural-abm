"""Pivot crossover prediction v1 (recommended experiment 3, stage 2): OpenAlex discovery episodes.

Design: docs/research/pivot-crossover-prediction-v1.md
Contract: experiments/configs/pivot_crossover_prediction_v1.yaml
Authorization: experiments/configs/pivot_crossover_prediction_authorization_v1.json

Stage 1 registered a rule that places where a table written before the run
loses to the in-run learner, from between-run heterogeneity and within-run
evidence measured outside the comparison. This stage applies it to the 24
OpenAlex discovery episodes. The policy is an entrant's chance of a
follow-up topic work within four quarters:

- the table T_k forecasts the mean follow-up share of k other episodes;
- the learner L forecasts the episode's own share among the precise
  entrants whose outcome is visible, five quarters after entry.

Each precise entrant of quarters 5 to 11 is forecast and scored by the
Brier score. The rule predicts the table-minus-learner gap from the other
23 episodes only, as H_k - V/n. Groups by table size and evidence are
labelled with an episode-cluster bootstrap and scored against the rule.

A run proceeds in this order: checks that read no label; the generator
check on synthetic labels at the empirical shape; and only if it passes,
the labels and the comparison. Outputs carry no author identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "experiments" / "configs" / "pivot_crossover_prediction_v1.yaml"
DEFAULT_AUTHORIZATION = ROOT / "experiments" / "configs" / "pivot_crossover_prediction_authorization_v1.json"
SCHEMA = "pivot_crossover_prediction.v1"
AUTHORIZATION_SCHEMA = "pivot_crossover_prediction_authorization.v1"
N11B_BUILDER_ROLE = "N11B_BUILDER_PREFLIGHT_AND_CLOCK_REUSED_BY_HASH_BOUND_IMPORT"
N11B_CONTRACT_ROLE = "N11B_CONTRACT_S1_INPUTS_AND_CLOCK"
E4_RECORD_ROLE = "E4_RECORD_DISCOVERY_COUNTS_FOR_PARITY_AND_THE_MINIMUM_EFFECT"
STAGE_1_RECORD_ROLE = "STAGE_1_RECORD"
QUARTERS = 12
LABEL_LAG = 5
FIRST_FORECAST_QUARTER = 5
TABLE_SIZES = (1, 2, 4, 8, 16)
EVIDENCE_EDGES = (1, 15, 50, 150)
SEED = 20261102
IDENTITY = ["summary.json", "receipt.json", "private/episode_quarters.parquet"]
LABELS = ("NEGATIVE", "INDISTINGUISHABLE", "POSITIVE")
QUARTER_SCHEMA = pa.schema([
    ("episode_id", pa.int64()), ("quarter", pa.int8()), ("precise_entrants", pa.int64()), ("follow_ups", pa.int64()),
])


class CrossoverError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossoverError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binding(contract: Mapping[str, Any], role: str) -> Path:
    matches = [b for b in contract["source_bindings"] if b["role"] == role]
    require(len(matches) == 1, f"exactly one binding must have role {role}")
    return ROOT / matches[0]["path"]


# -- contract, authorization, precommit --------------------------------------------------


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require(isinstance(contract, dict), "the contract must be a mapping")
    validate_contract(contract)
    return contract


def minimum_effect(counts: Mapping[str, Any]) -> float:
    """One percent of the base-rate Brier score, p0 (1 - p0), at E4's recorded follow-up share."""

    p0 = int(counts["labelled_precise_entrants"]) / int(counts["discovery_precise_entrants"])
    return 0.01 * p0 * (1.0 - p0)


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == SCHEMA, "unexpected contract schema")
    require(contract.get("status") == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION", "the contract is not a precommit design")
    for binding in contract["source_bindings"]:
        require(_sha256(ROOT / binding["path"]) == binding["sha256"], f"source binding drifted: {binding['path']}")
    e4 = json.loads(_binding(contract, E4_RECORD_ROLE).read_text(encoding="utf-8"))
    counts = e4["data"]["counts"]
    for key, value in contract["precondition"]["e4_parity_counts"].items():
        require(int(counts[key]) == int(value), f"the E4 record differs on {key}")
    require(e4["data_access"]["confirmatory_outcome_rows_read"] == 0, "E4 must have left confirmatory outcomes sealed")
    stage_1 = json.loads(_binding(contract, STAGE_1_RECORD_ROLE).read_text(encoding="utf-8"))
    require(stage_1["status"] == contract["precondition"]["stage_1_status"], "the stage 1 status differs")
    require(contract["data_access"]["partition"] == "DISCOVERY_ONLY", "the study may read the discovery partition only")
    authorization = contract["authorization"]
    require(authorization["outcomes"] == "DISCOVERY_PARTITION_ONLY", "outcomes are authorized for discovery only")
    for key in ("confirmatory_outcomes", "g4_outcomes"):
        require(authorization[key] == "NOT_AUTHORIZED", f"{key} must stay unauthorized")
    require(contract["policies"]["table_sizes"] == list(TABLE_SIZES), "the table sizes are 1, 2, 4, 8, and 16")
    require(contract["groups"]["table_sizes"] == list(TABLE_SIZES), "the groups use every table size")
    bins = contract["groups"]["evidence_bins"]
    require(bins["edges"] == list(EVIDENCE_EDGES) and bins["last_bin_open"] is True, "the evidence bins differ")
    require(contract["groups"]["count"] == len(TABLE_SIZES) * len(EVIDENCE_EDGES), "the group count differs")
    require(abs(float(contract["groups"]["minimum_effect"]["value"]) - minimum_effect(counts)) < 5e-8,
            "the minimum effect must be one percent of E4's base-rate Brier score")
    require(int(contract["groups"]["bootstrap"]["draws"]) == 2000, "the bootstrap has 2,000 draws")
    scoring = contract["scoring"]
    require(float(scoring["agreement_floor"]) == 0.8 and int(scoring["minimum_determined_groups"]) == 2, "the scoring floors differ")
    check = contract["generator_check"]
    require(check["taus"] == [0.0, 0.25, 0.5, 1.0] and int(check["replicates"]) == 40, "the generator check design differs")
    passes = check["passes_at_each_tau"]
    require(float(passes["agreement_share_of_determined_at_least"]) == 0.8, "the generator agreement floor is 0.8")
    require(float(passes["reversal_share_of_determined_at_most"]) == 0.05, "the generator reversal ceiling is 0.05")
    require(check["failure"] == "STOP_BEFORE_ANY_LABEL_IS_READ", "a failed generator check must stop before the labels")
    require(contract["execution"]["identity_artifacts"] == IDENTITY, "identity artifacts differ")
    require(contract["boundary"]["confirmatory_outcome_rows_read"] == 0, "no confirmatory outcome may be read")
    require(contract["boundary"]["identifies_learning"] is False, "the study does not identify learning")


def load_authorization(path: Path, contract_path: Path, script_path: Path) -> dict[str, Any]:
    authorization = json.loads(Path(path).read_text(encoding="utf-8"))
    require(authorization.get("schema_version") == AUTHORIZATION_SCHEMA, "unexpected authorization schema")
    require(authorization.get("status") == "AUTHORIZED_DISCOVERY_CROSSOVER_PREDICTION", "the run is not authorized")
    require(authorization.get("contract_sha256") == _sha256(contract_path), "the authorization names another contract")
    require(authorization.get("script_sha256") == _sha256(script_path), "the authorization names another script")
    require(bool(authorization.get("human_authorization", {}).get("user_instruction")), "the authorization must quote the user")
    require(authorization.get("outcomes") == "DISCOVERY_PARTITION_ONLY", "outcomes: discovery only")
    for key in ("confirmatory_outcomes", "g4_outcomes"):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"{key} must stay unauthorized")
    return authorization


def verify_precommit(commit_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    require(len(commit_sha) == 40 and all(c in "0123456789abcdef" for c in commit_sha), "a full lowercase SHA is needed")
    require(git("rev-parse", "HEAD") == commit_sha, "the precommit SHA is not HEAD")
    paths = list(contract["precommit"]["precommit_paths"])
    require(not git("status", "--porcelain", "--untracked-files=no", "--", *paths), "precommit paths are dirty")
    tracked = set(git("ls-tree", "-r", "--name-only", commit_sha, "--", *paths).split())
    require(tracked == set(paths), f"precommit paths missing from the commit: {sorted(set(paths) - tracked)}")
    return {"commit_sha": commit_sha, "precommit_paths": paths,
            "source_bindings": [{"path": b["path"], "sha256": b["sha256"]} for b in contract["source_bindings"]]}


def load_n11b(contract: Mapping[str, Any]) -> ModuleType:
    """The N11b builder by hash-bound import: its preflight, S1 inputs, and clock."""

    spec = importlib.util.spec_from_file_location("n11b_builder_for_crossover_prediction", _binding(contract, N11B_BUILDER_ROLE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def n11b_inputs(contract: Mapping[str, Any], n11b: ModuleType) -> dict[str, Any]:
    n11b_contract = n11b.load_contract(_binding(contract, N11B_CONTRACT_ROLE))
    s1 = n11b.load_s1(n11b_contract)
    preflight = n11b.preflight(n11b_contract, s1)
    root = Path(n11b_contract["inputs"]["s1_root"])
    return {
        "preflight": preflight,
        "inputs": {name: root / relative for name, relative in n11b.INPUTS.items()},
        "passes": preflight["status"].startswith("PASS"),
    }


# -- the build ---------------------------------------------------------------------------


def build(inputs: Mapping[str, Path], n11b: ModuleType, *, threads: int = 8, structure_only: bool = False) -> dict[str, Any]:
    """Discovery precise entrants by episode and entry quarter, and their follow-ups.

    ``structure_only`` stops before any work of an entrant is read. Otherwise
    follow-up topic works are read for discovery precise entrants only, and
    a follow-up counts only inside its entrant's label window, by E4's rule.
    """

    import duckdb

    off, precise = n11b._offset, n11b._precise
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute("CREATE TABLE ep AS SELECT episode_id, topic, t FROM read_parquet(?) WHERE partition = 'discovery'", [str(inputs["episodes"])])
    con.execute(
        'CREATE TABLE s1e AS SELECT s.episode_id, s.author_id, s.entry_date, s."offset" AS entry_offset '
        "FROM read_parquet(?) s JOIN ep USING (episode_id) WHERE s.precise", [str(inputs["entrants"])])
    episodes = [int(e) for (e,) in con.execute("SELECT episode_id FROM ep ORDER BY episode_id").fetchall()]
    index = {e: k for k, e in enumerate(episodes)}
    entrants = np.zeros((len(episodes), QUARTERS), dtype=np.int64)
    for e, q, count in con.execute("SELECT episode_id, entry_offset, COUNT(*) FROM s1e GROUP BY ALL ORDER BY ALL").fetchall():
        require(0 <= int(q) < QUARTERS, "an entry quarter lies outside the clock")
        entrants[index[int(e)], int(q)] = int(count)
    structure = {"episode_ids": episodes, "entrants": entrants}
    if structure_only:
        con.close()
        return {**structure, "follow_ups": None}

    con.execute("CREATE TABLE a AS SELECT work_id, author_id FROM read_parquet(?) WHERE author_id IN (SELECT author_id FROM s1e)",
                [str(inputs["authorships"])])
    con.execute(
        "CREATE TABLE fw AS SELECT a.author_id, w.year, w.date, w.topic FROM a "
        "JOIN read_parquet(?) w USING (work_id) WHERE w.topic IN (SELECT topic FROM ep)", [str(inputs["works"])])
    # E4's label rule for precise entrants: a follow-up topic work in the four quarters after the entry quarter.
    con.execute(
        f"""
        CREATE TABLE follow AS
        SELECT DISTINCT s.episode_id, s.author_id
        FROM s1e s JOIN ep USING (episode_id)
        JOIN fw f ON f.author_id = s.author_id AND f.topic = ep.topic
        WHERE f.year BETWEEN year(s.entry_date) AND year(s.entry_date) + 2 AND (
                   ({precise("f.date")} AND {off("f.date", "ep.t")} BETWEEN s.entry_offset + 1 AND s.entry_offset + 4)
                OR (NOT {precise("f.date")} AND 4 * (f.year - ep.t) >= s.entry_offset + 1 AND 4 * (f.year - ep.t) + 3 <= s.entry_offset + 4))
        """
    )
    follow_ups = np.zeros_like(entrants)
    for e, q, count in con.execute(
        "SELECT s.episode_id, s.entry_offset, COUNT(*) FROM s1e s JOIN follow f USING (episode_id, author_id) GROUP BY ALL ORDER BY ALL"
    ).fetchall():
        follow_ups[index[int(e)], int(q)] = int(count)
    con.close()
    return {**structure, "follow_ups": follow_ups}


def structure_checks(built: Mapping[str, Any], parity: Mapping[str, int]) -> dict[str, Any]:
    entrants = built["entrants"]
    return {
        "discovery_episodes": len(built["episode_ids"]) == int(parity["discovery_episodes"]),
        "discovery_precise_entrants": int(entrants.sum()) == int(parity["discovery_precise_entrants"]),
        "every_episode_has_two_precise_entrants": bool((entrants.sum(axis=1) >= 2).all()),
    }


# -- the comparison and the rule ---------------------------------------------------------


def evidence_bin(n: int) -> int:
    return int(np.searchsorted(EVIDENCE_EDGES, n, side="right")) - 1


def _donor_moments(shares: np.ndarray, donors: np.ndarray) -> tuple[float, float, int]:
    """Mean and population variance of the donors' shares, and their number."""

    values = shares[donors]
    mean = float(values.mean())
    return mean, float(((values - mean) ** 2).mean()), int(values.size)


def _table_variance(variance: float, m: int, k: int) -> float:
    """Variance of the mean of k of m donors drawn without replacement."""

    return variance * (m - k) / (k * (m - 1))


def rule_components(totals: np.ndarray, shares: np.ndarray, target: int | None) -> dict[str, Any]:
    """H_k and V from the episodes other than ``target`` (all episodes if None).

    For each world d, the table is built from the episodes other than d and
    the target; d's own sampling noise is removed from the table's squared
    error, and the per-label variance of d's share is unbiased.
    """

    episodes = np.arange(shares.size)
    worlds = episodes if target is None else episodes[episodes != target]
    heterogeneity = np.zeros(len(TABLE_SIZES))
    variance = 0.0
    for d in worlds:
        donors = worlds[worlds != d]
        mean, spread, m = _donor_moments(shares, donors)
        noise = shares[d] * (1.0 - shares[d]) / (totals[d] - 1)
        for j, k in enumerate(TABLE_SIZES):
            heterogeneity[j] += (mean - shares[d]) ** 2 + _table_variance(spread, m, k) - noise
        variance += shares[d] * (1.0 - shares[d]) * totals[d] / (totals[d] - 1)
    return {"H": heterogeneity / worlds.size, "V": variance / worlds.size}


def learner_forecast(entrants_row: np.ndarray, follow_row: np.ndarray, quarter: int) -> tuple[int, float | None]:
    """The episode's visible labels at ``quarter`` and their follow-up share.

    A precise entrant's label is visible five quarters after its entry
    quarter, so only entry quarters up to ``quarter - 5`` count.
    """

    last = quarter - LABEL_LAG
    if last < 0:
        return 0, None
    n = int(entrants_row[: last + 1].sum())
    return n, (float(follow_row[: last + 1].sum()) / n if n else None)


def comparison(entrants: np.ndarray, follow_ups: np.ndarray) -> dict[str, np.ndarray]:
    """Per target episode and group: the summed observed gap, the summed predicted gap, and the forecasts.

    The gap is the table's Brier score minus the learner's, summed over the
    precise entrants of quarters 5 to 11 whose learner has a visible label.
    """

    episodes = entrants.shape[0]
    require(episodes >= max(TABLE_SIZES) + 2, "every world needs as many donors as the largest table")
    totals = entrants.sum(axis=1)
    shares = follow_ups.sum(axis=1) / totals
    observed = np.zeros((episodes, len(TABLE_SIZES), len(EVIDENCE_EDGES)))
    predicted = np.zeros_like(observed)
    forecasts = np.zeros((episodes, len(EVIDENCE_EDGES)))
    every = np.arange(episodes)
    for e in range(episodes):
        mean, spread, m = _donor_moments(shares, every[every != e])
        rule = rule_components(totals, shares, e)
        for q in range(FIRST_FORECAST_QUARTER, QUARTERS):
            c, a = int(entrants[e, q]), int(follow_ups[e, q])
            n, learner = learner_forecast(entrants[e], follow_ups[e], q)
            if c == 0 or learner is None:
                continue
            b = evidence_bin(n)
            learner_brier = a * (learner - 1.0) ** 2 + (c - a) * learner ** 2
            forecasts[e, b] += c
            for j, k in enumerate(TABLE_SIZES):
                table_brier = a * (mean - 1.0) ** 2 + (c - a) * mean ** 2 + c * _table_variance(spread, m, k)
                observed[e, j, b] += table_brier - learner_brier
                predicted[e, j, b] += c * (rule["H"][j] - rule["V"] / n)
    return {"observed": observed, "predicted": predicted, "forecasts": forecasts}


def _label(mean: float, low: float, high: float, minimum: float) -> str:
    if low > 0 and mean >= minimum:
        return "POSITIVE"
    if high < 0 and mean <= -minimum:
        return "NEGATIVE"
    return "INDISTINGUISHABLE"


def _predicted_label(mean: float, minimum: float) -> str:
    if mean >= minimum:
        return "POSITIVE"
    if mean <= -minimum:
        return "NEGATIVE"
    return "NEUTRAL"


def groups(sums: Mapping[str, np.ndarray], minimum: float, seed: Sequence[int], draws: int) -> dict[str, dict[str, Any]]:
    """Every group's observed mean, episode-cluster interval, labels, and predicted mean."""

    observed, predicted, forecasts = sums["observed"], sums["predicted"], sums["forecasts"]
    episodes = forecasts.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    picks = rng.integers(0, episodes, size=(draws, episodes))
    counts = forecasts[picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = observed[picks].sum(axis=1) / counts[:, None, :]
    out: dict[str, dict[str, Any]] = {}
    for j, k in enumerate(TABLE_SIZES):
        for b in range(len(EVIDENCE_EDGES)):
            total = float(forecasts[:, b].sum())
            key = f"k={k}/bin={b}"
            if total == 0:
                out[key] = {"forecasts": 0, "label": "INDISTINGUISHABLE", "predicted_label": "NEUTRAL", "mean": None,
                            "low": None, "high": None, "predicted": None, "finite_draws": 0}
                continue
            mean = float(observed[:, j, b].sum()) / total
            pred = float(predicted[:, j, b].sum()) / total
            finite = boot[:, j, b][np.isfinite(boot[:, j, b])]
            require(finite.size > 0, f"no bootstrap draw holds a forecast of group {key}")
            low, high = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
            out[key] = {"forecasts": int(total), "episodes": int((forecasts[:, b] > 0).sum()), "mean": mean, "low": low, "high": high,
                        "label": _label(mean, low, high, minimum), "predicted": pred, "predicted_label": _predicted_label(pred, minimum),
                        "covered": bool(low <= pred <= high), "finite_draws": int(finite.size)}
    return out


def _agrees(group: Mapping[str, Any]) -> bool:
    """The predicted mean has the sign of the observed label."""

    return (group["predicted"] > 0) if group["label"] == "POSITIVE" else (group["predicted"] < 0)


def score(table: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    determined = [g for g in table.values() if g["label"] != "INDISTINGUISHABLE"]
    agreements = sum(_agrees(g) for g in determined)
    matches = sum(g["predicted_label"] == g["label"] for g in determined)
    reversals = sum({g["predicted_label"], g["label"]} == {"POSITIVE", "NEGATIVE"} for g in determined)
    signs = sorted({g["label"] for g in determined})
    return {
        "determined": len(determined),
        "agreements": agreements,
        "agreement_share": agreements / len(determined) if determined else None,
        "matches": matches,
        "reversals": reversals,
        "match_share": matches / len(determined) if determined else None,
        "signs": signs,
        "positive": sum(g["label"] == "POSITIVE" for g in determined),
        "negative": sum(g["label"] == "NEGATIVE" for g in determined),
        "baseline_table_always_wins_agreements": sum(g["label"] == "NEGATIVE" for g in determined),
        "baseline_learner_always_wins_agreements": sum(g["label"] == "POSITIVE" for g in determined),
        "coverage": sum(bool(g.get("covered")) for g in table.values()) / len(table),
    }


def analyse(entrants: np.ndarray, follow_ups: np.ndarray, minimum: float, seed: Sequence[int], draws: int) -> dict[str, Any]:
    sums = comparison(entrants, follow_ups)
    table = groups(sums, minimum, seed, draws)
    return {"groups": table, "score": score(table), "sums": sums}


# -- generator check ---------------------------------------------------------------------


def synthetic_follow_ups(entrants: np.ndarray, p0: float, tau: float, seed: Sequence[int]) -> np.ndarray:
    """Exchangeable entrants: episode e follows up with probability expit(logit(p0) + tau z_e)."""

    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    z = rng.standard_normal(entrants.shape[0])
    pi = 1.0 / (1.0 + np.exp(-(np.log(p0 / (1.0 - p0)) + tau * z)))
    out = np.zeros_like(entrants)
    for e in range(entrants.shape[0]):
        for q in range(QUARTERS):
            out[e, q] = rng.binomial(int(entrants[e, q]), pi[e])
    return out


def generator_check(entrants: np.ndarray, contract: Mapping[str, Any], p0: float, minimum: float) -> dict[str, Any]:
    check = contract["generator_check"]
    draws = int(contract["groups"]["bootstrap"]["draws"])
    floors = check["passes_at_each_tau"]
    report: dict[str, Any] = {}
    for t, tau in enumerate(check["taus"]):
        determined = agreements = matches = reversals = with_positive = with_reversal = 0
        for r in range(int(check["replicates"])):
            labels = synthetic_follow_ups(entrants, p0, float(tau), (SEED, 1, t, r))
            result = analyse(entrants, labels, minimum, (SEED, 2, t, r), draws)["score"]
            determined += result["determined"]
            agreements += result["agreements"]
            matches += result["matches"]
            reversals += result["reversals"]
            with_positive += int(result["positive"] > 0)
            with_reversal += int(result["reversals"] > 0)
        agreement_share = agreements / determined if determined else None
        reversal_share = reversals / determined if determined else None
        passes = bool(determined) and agreement_share >= float(floors["agreement_share_of_determined_at_least"]) \
            and reversal_share <= float(floors["reversal_share_of_determined_at_most"])
        report[f"tau={tau:g}"] = {"determined": determined, "agreements": agreements, "matches": matches, "reversals": reversals,
                                  "agreement_share": agreement_share, "match_share": matches / determined if determined else None,
                                  "reversal_share": reversal_share,
                                  "replicates_with_a_positive_group": with_positive,
                                  "replicates_with_a_reversal": with_reversal, "passes": passes}
    return {"taus": report, "passes": all(row["passes"] for row in report.values()), "replicates": int(check["replicates"])}


# -- description, decision ---------------------------------------------------------------


def describe(entrants: np.ndarray, follow_ups: np.ndarray, sums: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The rule's quantities from all 24 episodes, decided by nothing."""

    totals = entrants.sum(axis=1)
    shares = follow_ups.sum(axis=1) / totals
    components = rule_components(totals, shares, None)
    mean = float(shares.mean())
    between = float(((shares - mean) ** 2).mean() - (shares * (1.0 - shares) / (totals - 1)).mean())
    weights, evidence = [], []
    for e in range(entrants.shape[0]):
        for q in range(FIRST_FORECAST_QUARTER, QUARTERS):
            n, _ = learner_forecast(entrants[e], follow_ups[e], q)
            if entrants[e, q] and n:
                weights.append(int(entrants[e, q]))
                evidence.append(n)
    weights_a, evidence_a = np.array(weights), np.array(evidence)
    rows = {}
    for j, k in enumerate(TABLE_SIZES):
        h = float(components["H"][j])
        crossing = components["V"] / h if h > 0 else None
        beyond = float(weights_a[evidence_a >= crossing].sum() / weights_a.sum()) if crossing is not None else 0.0
        rows[f"k={k}"] = {"H": h, "crossing_labels": crossing, "share_of_forecasts_beyond_the_crossing": beyond}
    return {
        "V": float(components["V"]),
        "tables": rows,
        "between_episode_variance_of_shares": between,
        "between_episode_sd_of_shares": float(np.sqrt(between)) if between > 0 else 0.0,
        "mean_episode_share": mean,
        "forecasts": int(sums["forecasts"].sum()),
    }


def first_sustained_positive(table: Mapping[str, Mapping[str, Any]], k: int) -> int | None:
    labels = [table[f"k={k}/bin={b}"]["label"] for b in range(len(EVIDENCE_EDGES))]
    for b in range(len(labels)):
        if all(label == "POSITIVE" for label in labels[b:]):
            return b
    return None


def expectations(generator: Mapping[str, Any], table: Mapping[str, Mapping[str, Any]] | None, result: Mapping[str, Any] | None) -> dict[str, Any]:
    half = generator["taus"]["tau=0.5"]["replicates_with_a_positive_group"] >= generator["replicates"] / 2
    rows: dict[str, Any] = {
        "X1": {"held": bool(generator["passes"])},
        "X2": {"held": bool(half), "replicates_with_a_positive_group": generator["taus"]["tau=0.5"]["replicates_with_a_positive_group"]},
    }
    if table is None or result is None:
        for key in ("X3", "X4", "X5"):
            rows[key] = {"held": None, "reason": "NO_LABEL_WAS_READ"}
        return rows
    rows["X3"] = {"held": all(table[f"k={k}/bin=0"]["label"] == "NEGATIVE" for k in TABLE_SIZES)}
    last = len(EVIDENCE_EDGES) - 1
    rows["X4"] = {"held": table[f"k=16/bin={last}"]["label"] == "POSITIVE"}
    rows["X5"] = {"held": result["reversals"] == 0 and result["agreement_share"] is not None and result["agreement_share"] >= 0.8}
    return rows


def decide(contract: Mapping[str, Any], generator: Mapping[str, Any], result: Mapping[str, Any] | None) -> str:
    prefix = contract["decision"]["status_prefix"]
    scoring = contract["scoring"]
    if not generator["passes"]:
        return prefix + "STOPPED_THE_RULE_FAILS_ON_KNOWN_TRUTH"
    require(result is not None, "a passing generator check needs the data result")
    if result["determined"] < int(scoring["minimum_determined_groups"]):
        return prefix + "UNDETERMINED_TOO_FEW_DETERMINED_GROUPS"
    if result["reversals"] > 0:
        return prefix + "THE_RULE_REVERSES_A_DETERMINED_COMPARISON"
    if result["agreement_share"] < float(scoring["agreement_floor"]):
        return prefix + "THE_RULE_MISSES_DETERMINED_COMPARISONS"
    if result["signs"] == ["NEGATIVE", "POSITIVE"]:
        return prefix + "THE_RULE_ORDERS_THE_TABLE_AND_THE_LEARNER"
    if result["signs"] == ["NEGATIVE"]:
        return prefix + "THE_TABLE_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES"
    return prefix + "THE_LEARNER_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES"


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


# -- run ---------------------------------------------------------------------------------


def run(contract: Mapping[str, Any], *, threads: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    timing: dict[str, float] = {}
    started = time.monotonic()
    e4 = json.loads(_binding(contract, E4_RECORD_ROLE).read_text(encoding="utf-8"))
    parity = contract["precondition"]["e4_parity_counts"]
    p0 = int(parity["labelled_precise_entrants"]) / int(parity["discovery_precise_entrants"])
    minimum = minimum_effect(e4["data"]["counts"])
    n11b = load_n11b(contract)
    checks = n11b_inputs(contract, n11b)
    require(checks["passes"], f"the N11b preflight failed: {checks['preflight']['status']}")

    structure = build(checks["inputs"], n11b, threads=threads, structure_only=True)
    before = structure_checks(structure, parity)
    require(all(before.values()), f"a check before any label failed: {before}")
    timing["structure"] = round(time.monotonic() - started, 1)

    mark = time.monotonic()
    generator = generator_check(structure["entrants"], contract, p0, minimum)
    timing["generator_check"] = round(time.monotonic() - mark, 1)
    summary: dict[str, Any] = {
        "schema_version": "pivot_crossover_prediction_summary.v1",
        "checks": {"n11b_preflight": checks["preflight"]["status"], "before_any_label": before},
        "generator_check": generator,
        "minimum_effect": minimum,
        "structure": {"episodes": len(structure["episode_ids"]), "precise_entrants": int(structure["entrants"].sum())},
        "boundary": dict(contract["boundary"]),
    }
    built: dict[str, Any] = {"episode_ids": structure["episode_ids"], "entrants": structure["entrants"], "follow_ups": None}
    if not generator["passes"]:
        summary.update({"status": decide(contract, generator, None), "expectations": expectations(generator, None, None),
                        "data_access": {"partition": "DISCOVERY_ONLY", "labels_read": 0, "confirmatory_outcome_rows_read": 0,
                                        "g4_outcome_rows_read": 0}})
        timing["total"] = round(time.monotonic() - started, 1)
        return _rounded(summary), built, timing

    mark = time.monotonic()
    built = build(checks["inputs"], n11b, threads=threads)
    require(np.array_equal(built["entrants"], structure["entrants"]), "the labelled build changed the structure")
    after = {"labelled_precise_entrants": int(built["follow_ups"].sum()) == int(parity["labelled_precise_entrants"])}
    require(all(after.values()), f"the label parity with E4 failed: {int(built['follow_ups'].sum())}")
    timing["labels"] = round(time.monotonic() - mark, 1)

    mark = time.monotonic()
    data = analyse(built["entrants"], built["follow_ups"], minimum, (SEED, 3), int(contract["groups"]["bootstrap"]["draws"]))
    result = data["score"]
    status = decide(contract, generator, result)
    summary.update({
        "status": status,
        "checks": {**summary["checks"], "after_the_labels": after},
        "groups": data["groups"],
        "score": result,
        "crossings": {f"k={k}": first_sustained_positive(data["groups"], k) for k in TABLE_SIZES},
        "described": describe(built["entrants"], built["follow_ups"], data["sums"]),
        "expectations": expectations(generator, data["groups"], result),
        "data_access": {"partition": "DISCOVERY_ONLY", "labels_read": int(built["entrants"].sum()),
                        "confirmatory_outcome_rows_read": 0, "g4_outcome_rows_read": 0},
    })
    timing["comparison"] = round(time.monotonic() - mark, 1)
    timing["total"] = round(time.monotonic() - started, 1)
    return _rounded(summary), built, timing


def quarter_table(built: Mapping[str, Any]) -> pa.Table:
    rows = []
    follow = built["follow_ups"]
    for e, episode in enumerate(built["episode_ids"]):
        for q in range(QUARTERS):
            rows.append({"episode_id": int(episode), "quarter": q, "precise_entrants": int(built["entrants"][e, q]),
                         "follow_ups": None if follow is None else int(follow[e, q])})
    return pa.Table.from_pylist(rows, schema=QUARTER_SCHEMA)


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(0o600)


def write_outputs(output_dir: Path, summary: Mapping[str, Any], built: Mapping[str, Any], *, contract_path: Path, authorization_path: Path,
                  precommit: Mapping[str, Any], timing: Mapping[str, float]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    output_dir.chmod(0o700)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    pq.write_table(quarter_table(built), private / "episode_quarters.parquet")
    (private / "episode_quarters.parquet").chmod(0o600)
    _write_exclusive(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "pivot_crossover_prediction_receipt.v1",
        "status": "COMPLETE",
        "contract_sha256": _sha256(contract_path),
        "script_sha256": _sha256(Path(__file__)),
        "authorization_sha256": _sha256(authorization_path),
        "precommit": dict(precommit),
        "outputs": {name: _sha256(output_dir / name) for name in ("summary.json", "private/episode_quarters.parquet")},
    }
    _write_exclusive(output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write_exclusive(output_dir / "timing.json", json.dumps(dict(timing), indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    load_authorization(args.authorization, config_path, Path(__file__).resolve())
    if args.command == "plan":
        n11b = load_n11b(contract)
        checks = n11b_inputs(contract, n11b)
        print(json.dumps({"status": "PRECOMMIT_PLAN_VALID", "contract_sha256": _sha256(config_path),
                          "n11b_preflight": checks["preflight"]["status"]}, indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or args.output_dir is None:
        parser.error("run requires --precommit-sha and --output-dir")
    precommit = verify_precommit(args.precommit_sha, contract)
    summary, built, timing = run(contract, threads=args.threads)
    summary = {**summary, "precommit_sha": args.precommit_sha}
    receipt = write_outputs(args.output_dir.resolve(), summary, built, contract_path=config_path,
                            authorization_path=args.authorization.resolve(), precommit=precommit, timing=timing)
    print(json.dumps({"status": summary["status"], "receipt": receipt["status"], "wall_seconds": timing["total"]}))


if __name__ == "__main__":
    main()
