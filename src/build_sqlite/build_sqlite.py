from pathlib import Path
import datetime as dt
import json
import sqlite3
import struct


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data/raw_data"
DATA_DIR = PROJECT_ROOT / "data"

BSON_PATH = RAW_DATA_DIR / "runs.bson"
ABLATION_BSON_PATH = RAW_DATA_DIR / "runs_ablation_minimality.bson"
BSON_PATHS = (BSON_PATH, ABLATION_BSON_PATH)
DB_PATH = DATA_DIR / "causal_runs.sqlite"


APPLICABLE_ACTIONS_BY_DOMAIN = {
    "GSM8K": ["LOCAL_REPAIR", "RETRY", "REPLAN", "ESCALATE"],
    "MBPP": ["LOCAL_REPAIR", "RETRY", "REPLAN", "TOOL_FIX", "ESCALATE"],
    "SealQA": ["LOCAL_REPAIR", "RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "ESCALATE"],
    "MedBrowseComp": ["LOCAL_REPAIR", "RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "ESCALATE"],
    "BrowseComp": ["LOCAL_REPAIR", "RETRY", "REPLAN", "RETRIEVE_MORE", "TOOL_FIX", "ESCALATE"],
    "Humaneval": ["LOCAL_REPAIR", "RETRY", "REPLAN", "TOOL_FIX", "ESCALATE"],
}


class ObjectId:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return self.data.hex()

    def __repr__(self):
        return str(self)


def _read_cstring(buf, idx):
    end = buf.index(0, idx)
    return buf[idx:end].decode("utf-8", "replace"), end + 1


def _read_bson_value(value_type, buf, idx):
    if value_type == 0x01:
        return struct.unpack_from("<d", buf, idx)[0], idx + 8
    if value_type == 0x02:
        n = struct.unpack_from("<i", buf, idx)[0]
        idx += 4
        return buf[idx : idx + n - 1].decode("utf-8", "replace"), idx + n
    if value_type == 0x03:
        return _read_bson_doc(buf, idx)
    if value_type == 0x04:
        raw, next_idx = _read_bson_doc(buf, idx)
        return [raw[key] for key in sorted(raw, key=lambda x: int(x) if x.isdigit() else 10**12)], next_idx
    if value_type == 0x05:
        n = struct.unpack_from("<i", buf, idx)[0]
        subtype = buf[idx + 4]
        idx += 5
        return {"$binary_subtype": subtype, "$binary_size": n}, idx + n
    if value_type == 0x07:
        return ObjectId(buf[idx : idx + 12]), idx + 12
    if value_type == 0x08:
        return bool(buf[idx]), idx + 1
    if value_type == 0x09:
        ms = struct.unpack_from("<q", buf, idx)[0]
        return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(), idx + 8
    if value_type == 0x0A:
        return None, idx
    if value_type == 0x0B:
        pattern, idx = _read_cstring(buf, idx)
        options, idx = _read_cstring(buf, idx)
        return {"$regex": pattern, "$options": options}, idx
    if value_type == 0x10:
        return struct.unpack_from("<i", buf, idx)[0], idx + 4
    if value_type == 0x11:
        inc, ts = struct.unpack_from("<II", buf, idx)
        return {"$timestamp": ts, "i": inc}, idx + 8
    if value_type == 0x12:
        return struct.unpack_from("<q", buf, idx)[0], idx + 8
    if value_type == 0x13:
        return {"$decimal128_raw_bytes": buf[idx : idx + 16].hex()}, idx + 16
    raise ValueError(f"Unsupported BSON value type {value_type:#x} at byte {idx}")


def _read_bson_doc(buf, idx=0):
    start = idx
    doc_size = struct.unpack_from("<i", buf, idx)[0]
    end = start + doc_size
    idx += 4
    out = {}
    while idx < end - 1:
        value_type = buf[idx]
        idx += 1
        key, idx = _read_cstring(buf, idx)
        out[key], idx = _read_bson_value(value_type, buf, idx)
    return out, end


def iter_bson_documents(path):
    buf = Path(path).read_bytes()
    idx = 0
    while idx < len(buf):
        doc_size = struct.unpack_from("<i", buf, idx)[0]
        doc, _ = _read_bson_doc(buf, idx)
        yield doc
        idx += doc_size


def as_json(value):
    if value is None:
        return None
    return json.dumps(value, default=str, ensure_ascii=False)


def as_int_bool(value):
    if value is None:
        return None
    return int(bool(value))


def benchmark_from_experiment(experiment_name):
    if not experiment_name:
        return None
    name = str(experiment_name)
    if name.startswith("ablation_"):
        return name.replace("ablation_minimality_", "").replace("ablation_nogold_", "")
    return name


def normalize_domain(experiment_name):
    benchmark = benchmark_from_experiment(experiment_name)
    if not benchmark:
        return None
    lookup = {
        "gsm8k": "GSM8K",
        "mbpp": "MBPP",
        "sealqa": "SealQA",
        "sealqa hard": "SealQA",
        "medbrowse": "MedBrowseComp",
        "medbrowsecomp": "MedBrowseComp",
        "browsecomp": "BrowseComp",
        "humaneval": "Humaneval",
    }
    return lookup.get(str(benchmark).lower(), benchmark)


def ablation_type_from_experiment(experiment_name):
    name = str(experiment_name or "")
    if name.startswith("ablation_minimality_"):
        return "minimality"
    if name.startswith("ablation_nogold_"):
        return "nogold"
    if name.startswith("ablation_"):
        return name.split("_", 2)[1] if "_" in name else "unknown"
    return None


def applicable_actions_for_domain(domain):
    return APPLICABLE_ACTIONS_BY_DOMAIN.get(domain, ["LOCAL_REPAIR", "RETRY", "REPLAN", "ESCALATE"])


def trace_id_for(run_id, problem_id):
    return f"{run_id}::{problem_id}"


def step_uid_for(trace_id, step_id):
    return f"{trace_id}::step::{step_id}"


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return PROJECT_ROOT / path


def create_schema(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS runs;
        DROP TABLE IF EXISTS traces;
        DROP TABLE IF EXISTS steps;
        DROP TABLE IF EXISTS repair_attempts;
        DROP TABLE IF EXISTS judge_votes;
        DROP TABLE IF EXISTS consensus_steps;
        DROP TABLE IF EXISTS trace_metrics;
        DROP TABLE IF EXISTS triage_labels;

        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            experiment_name TEXT,
            benchmark TEXT,
            domain TEXT,
            is_ablation INTEGER NOT NULL,
            ablation_type TEXT,
            ablation_target TEXT,
            timestamp TEXT,
            model_used TEXT,
            num_problems_declared INTEGER,
            num_traces_stored INTEGER,
            num_passing_traces INTEGER,
            num_failing_traces INTEGER,
            stats_total INTEGER,
            stats_passing INTEGER,
            stats_failing INTEGER,
            stats_fixed INTEGER,
            stats_accuracy REAL,
            total_experiment_time_minutes REAL,
            source_file TEXT
        );

        CREATE TABLE traces (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            benchmark TEXT,
            domain TEXT,
            model_used TEXT,
            is_ablation INTEGER NOT NULL,
            ablation_type TEXT,
            timestamp TEXT,
            success INTEGER,
            problem_statement TEXT,
            gold_answer TEXT,
            final_answer TEXT,
            num_steps INTEGER,
            causal_flow_analysis_minutes REAL,
            is_passing_trace INTEGER NOT NULL,
            is_failing_trace INTEGER NOT NULL,
            answer_exact_match INTEGER,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE triage_labels (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            benchmark TEXT,
            domain TEXT,
            is_ablation INTEGER NOT NULL,
            ablation_type TEXT,
            action_label TEXT,
            label_source TEXT,
            is_auto_labeled INTEGER NOT NULL,
            needs_labeling INTEGER NOT NULL,
            is_local_repairable INTEGER NOT NULL,
            num_successful_repair_steps INTEGER,
            applicable_actions_json TEXT,
            llm_1_action TEXT,
            llm_1_rationale TEXT,
            llm_2_action TEXT,
            llm_2_rationale TEXT,
            human_action TEXT,
            human_rationale TEXT,
            split TEXT,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE steps (
            step_uid TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            step_id INTEGER,
            step_index INTEGER,
            step_type TEXT,
            dependencies_json TEXT,
            text TEXT,
            tool_name TEXT,
            tool_args_json TEXT,
            tool_output_json TEXT,
            tool_call_result INTEGER,
            state_snapshot_json TEXT,
            trace_success INTEGER,
            has_tool INTEGER,
            is_reasoning_step INTEGER,
            is_tool_call INTEGER,
            is_tool_response INTEGER,
            is_final_answer INTEGER,
            text_length INTEGER,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE repair_attempts (
            repair_id TEXT PRIMARY KEY,
            step_uid TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            step_id INTEGER,
            repair_idx TEXT,
            success_predicted INTEGER,
            repair_succeeded INTEGER,
            minimality_score REAL,
            minimality_lex REAL,
            minimality_edit REAL,
            minimality_sem REAL,
            original_step_type TEXT,
            original_text TEXT,
            repaired_text TEXT,
            original_tool_name TEXT,
            original_tool_args_json TEXT,
            repaired_tool_name TEXT,
            repaired_tool_args_json TEXT,
            raw_repair_json TEXT,
            FOREIGN KEY (step_uid) REFERENCES steps(step_uid),
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        );

        CREATE TABLE judge_votes (
            judge_vote_id TEXT PRIMARY KEY,
            step_uid TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            step_id INTEGER,
            candidate_idx TEXT,
            proposed_by TEXT,
            judge_agent TEXT,
            judge_role TEXT,
            agrees INTEGER,
            confidence REAL,
            reasoning TEXT,
            agreement_text TEXT,
            evidence_strength TEXT,
            judge_says_causal INTEGER,
            is_repairable_step INTEGER,
            vote_matches_repairability INTEGER,
            FOREIGN KEY (step_uid) REFERENCES steps(step_uid),
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        );

        CREATE TABLE consensus_steps (
            consensus_id TEXT PRIMARY KEY,
            step_uid TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            step_id INTEGER,
            step_type TEXT,
            consensus_score REAL,
            final_verdict TEXT,
            proposed_by TEXT,
            num_critiques INTEGER,
            final_critic_summary TEXT,
            text TEXT,
            tool_name TEXT,
            tool_args_json TEXT,
            dependencies_json TEXT,
            is_repairable_step INTEGER,
            has_successful_repair INTEGER,
            raw_consensus_json TEXT,
            FOREIGN KEY (step_uid) REFERENCES steps(step_uid),
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        );

        CREATE TABLE trace_metrics (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            problem_id TEXT,
            minimality_average REAL,
            minimality_min REAL,
            minimality_max REAL,
            num_identified_causal_steps INTEGER,
            attribution_precision REAL,
            attribution_recall REAL,
            attribution_f1 REAL,
            repairs_attempted INTEGER,
            repairs_successful INTEGER,
            repairs_failed INTEGER,
            repair_success_rate REAL,
            num_successful_repair_steps INTEGER,
            num_consensus_steps INTEGER,
            multi_agent_skipped INTEGER,
            multi_agent_skip_reason TEXT,
            causal_steps_json TEXT,
            identified_steps_json TEXT,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE INDEX idx_traces_run ON traces(run_id);
        CREATE INDEX idx_traces_benchmark_model ON traces(benchmark, model_used);
        CREATE INDEX idx_traces_domain ON traces(domain, success);
        CREATE INDEX idx_triage_domain_label ON triage_labels(domain, action_label);
        CREATE INDEX idx_triage_needs_labeling ON triage_labels(needs_labeling, domain);
        CREATE INDEX idx_steps_trace ON steps(trace_id);
        CREATE INDEX idx_steps_type_tool ON steps(step_type, tool_name);
        CREATE INDEX idx_repairs_step ON repair_attempts(step_uid);
        CREATE INDEX idx_judge_votes_step ON judge_votes(step_uid);
        CREATE INDEX idx_consensus_step ON consensus_steps(step_uid);
        """
    )


def insert_run(conn, run, source_file):
    experiment_name = run.get("experiment_name")
    is_ablation = str(experiment_name).startswith("ablation_")
    domain = normalize_domain(experiment_name)
    passing = run.get("passing_traces", []) or []
    failing = run.get("failing_traces", []) or []
    stats = run.get("stats", {}) or {}
    conn.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run.get("run_id"),
            experiment_name,
            benchmark_from_experiment(experiment_name),
            domain,
            int(is_ablation),
            ablation_type_from_experiment(experiment_name),
            domain if is_ablation else None,
            run.get("timestamp"),
            run.get("model_used"),
            run.get("num_problems"),
            len(passing) + len(failing),
            len(passing),
            len(failing),
            stats.get("total"),
            stats.get("passing"),
            stats.get("failing"),
            stats.get("fixed"),
            stats.get("accuracy"),
            stats.get("total_experiment_time_minutes"),
            source_file,
        ),
    )


def successful_repair_items(counterfactual_repair):
    counterfactual_repair = counterfactual_repair or {}
    repairs = counterfactual_repair.get("successful_repairs") or counterfactual_repair.get("best_repairs") or {}
    if isinstance(repairs, dict):
        return list(repairs.items())
    if isinstance(repairs, list):
        return [(str(idx), value) for idx, value in enumerate(repairs)]
    return []


def insert_trace_and_steps(conn, run, trace, is_passing_trace):
    run_id = run.get("run_id")
    experiment_name = run.get("experiment_name")
    is_ablation = str(experiment_name).startswith("ablation_")
    domain = normalize_domain(experiment_name)
    problem_id = str(trace.get("problem_id"))
    trace_id = trace_id_for(run_id, problem_id)
    trace_obj = trace.get("trace", {}) or {}
    steps = trace_obj.get("steps", []) or []
    success = trace.get("success", trace_obj.get("success"))
    gold = trace.get("gold_answer", trace_obj.get("gold_answer"))
    final = trace.get("final_answer", trace_obj.get("final_answer"))

    conn.execute(
        "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trace_id,
            run_id,
            problem_id,
            benchmark_from_experiment(experiment_name),
            domain,
            run.get("model_used"),
            int(is_ablation),
            ablation_type_from_experiment(experiment_name),
            trace.get("timestamp"),
            as_int_bool(success),
            trace.get("problem_statement", trace_obj.get("problem_statement")),
            None if gold is None else str(gold),
            None if final is None else str(final),
            trace_obj.get("num_steps", len(steps)),
            trace.get("causal_flow_analysis_time_minutes"),
            int(is_passing_trace),
            int(not is_passing_trace),
            None if gold is None or final is None else int(str(gold).strip() == str(final).strip()),
        ),
    )

    for step_index, step in enumerate(steps):
        step_id = step.get("step_id", step_index)
        step_type = step.get("step_type")
        text = step.get("text")
        tool_name = step.get("tool_name")
        conn.execute(
            "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                step_uid_for(trace_id, step_id),
                trace_id,
                run_id,
                problem_id,
                step_id,
                step_index,
                step_type,
                as_json(step.get("dependencies")),
                text,
                tool_name,
                as_json(step.get("tool_args")),
                as_json(step.get("tool_output")),
                as_int_bool(step.get("tool_call_result")),
                as_json(step.get("state_snapshot")),
                as_int_bool(success),
                int(bool(tool_name)),
                int(step_type == "reasoning"),
                int(step_type == "tool_call"),
                int(step_type == "tool_response"),
                int(step_type == "final_answer"),
                len(text) if isinstance(text, str) else None,
            ),
        )

    if not is_passing_trace:
        insert_failed_trace_analysis(conn, run, trace, trace_id, problem_id)


def insert_failed_trace_analysis(conn, run, trace, trace_id, problem_id):
    run_id = run.get("run_id")
    experiment_name = run.get("experiment_name")
    is_ablation = str(experiment_name).startswith("ablation_")
    domain = normalize_domain(experiment_name)
    analysis = trace.get("analysis", {}) or {}
    metrics = trace.get("metrics", {}) or {}
    repair_metrics = metrics.get("repairs", {}) or {}
    counterfactual_repair = analysis.get("counterfactual_repair") or {}
    # MBPP stores repairs directly in final_repairs (not nested under successful_repairs)
    if not counterfactual_repair:
        raw_final = analysis.get("final_repairs") or {}
        if raw_final:
            counterfactual_repair = {"successful_repairs": raw_final}
    multi_agent = analysis.get("multi_agent_critique", {}) or {}
    successful_repairs = successful_repair_items(counterfactual_repair)
    repairable_step_ids = {int(step_id) for step_id, _ in successful_repairs if str(step_id).isdigit()}
    is_local_repairable = bool(successful_repairs) or (counterfactual_repair.get("num_successful_repairs") or 0) > 0 or (repair_metrics.get("successful_repairs") or 0) > 0
    num_successful_repair_steps = len(successful_repairs) or counterfactual_repair.get("num_successful_repairs") or repair_metrics.get("successful_repairs") or 0

    conn.execute(
        "INSERT OR REPLACE INTO triage_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trace_id,
            run_id,
            problem_id,
            benchmark_from_experiment(experiment_name),
            domain,
            int(is_ablation),
            ablation_type_from_experiment(experiment_name),
            "LOCAL_REPAIR" if is_local_repairable else None,
            "auto_causalflow" if is_local_repairable else None,
            int(is_local_repairable),
            int(not is_local_repairable),
            int(is_local_repairable),
            num_successful_repair_steps,
            as_json(applicable_actions_for_domain(domain)),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )

    for repair_idx, repair in successful_repairs:
        if not isinstance(repair, dict):
            continue
        step_id = int(repair_idx) if str(repair_idx).isdigit() else None
        original_step = repair.get("original_step", {}) or {}
        repaired_step = repair.get("repaired_step", {}) or {}
        conn.execute(
            "INSERT OR REPLACE INTO repair_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{trace_id}::step::{repair_idx}::repair::successful",
                step_uid_for(trace_id, repair_idx),
                trace_id,
                run_id,
                problem_id,
                step_id,
                str(repair_idx),
                as_int_bool(repair.get("success_predicted", True)),
                1,
                repair.get("minimality_score"),
                repair.get("minimality_lex"),
                repair.get("minimality_edit"),
                repair.get("minimality_sem"),
                original_step.get("step_type"),
                repair.get("original_text", original_step.get("text")),
                repair.get("repaired_text", repaired_step.get("text")),
                original_step.get("tool_name"),
                as_json(original_step.get("tool_args")),
                repaired_step.get("tool_name"),
                as_json(repaired_step.get("tool_args")),
                as_json(repair),
            ),
        )

    critique_details = multi_agent.get("critique_details", {}) if isinstance(multi_agent, dict) else {}
    if isinstance(critique_details, dict):
        for candidate_idx, candidate in critique_details.items():
            if not isinstance(candidate, dict):
                continue
            step_id = candidate.get("step_id")
            if step_id is None and str(candidate_idx).isdigit():
                step_id = int(candidate_idx)
            repairable = step_id in repairable_step_ids if step_id is not None else None
            final_verdict = candidate.get("final_verdict")
            for judge_number, vote in enumerate(candidate.get("judge_ensemble", []) or []):
                if not isinstance(vote, dict):
                    continue
                agrees = vote.get("agrees")
                judge_says_causal = bool(final_verdict) if final_verdict is not None else bool(agrees)
                conn.execute(
                    "INSERT OR REPLACE INTO judge_votes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"{trace_id}::candidate::{candidate_idx}::judge::{vote.get('agent', judge_number)}",
                        step_uid_for(trace_id, step_id),
                        trace_id,
                        run_id,
                        problem_id,
                        step_id,
                        str(candidate_idx),
                        candidate.get("proposed_by"),
                        vote.get("agent"),
                        vote.get("role"),
                        as_int_bool(agrees),
                        vote.get("confidence"),
                        vote.get("reasoning"),
                        vote.get("agreement"),
                        vote.get("evidence_strength"),
                        as_int_bool(judge_says_causal),
                        as_int_bool(repairable),
                        None if repairable is None else int(judge_says_causal == repairable),
                    ),
                )

    consensus_list = multi_agent.get("consensus_steps", []) if isinstance(multi_agent, dict) else []
    for idx, consensus in enumerate(consensus_list):
        if not isinstance(consensus, dict):
            continue
        step_id = consensus.get("step_id", idx)
        repairable = step_id in repairable_step_ids if step_id is not None else None
        conn.execute(
            "INSERT OR REPLACE INTO consensus_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{trace_id}::consensus::{idx}",
                step_uid_for(trace_id, step_id),
                trace_id,
                run_id,
                problem_id,
                step_id,
                consensus.get("step_type"),
                consensus.get("consensus_score"),
                None if consensus.get("final_verdict") is None else str(consensus.get("final_verdict")),
                consensus.get("proposed_by"),
                consensus.get("num_critiques"),
                consensus.get("final_critic_summary"),
                consensus.get("text"),
                consensus.get("tool_name"),
                as_json(consensus.get("tool_args")),
                as_json(consensus.get("dependencies")),
                as_int_bool(repairable),
                as_int_bool(repairable),
                as_json(consensus),
            ),
        )

    minimality = metrics.get("minimality", {}) or {}
    attribution = metrics.get("attribution", {}) or {}
    causal_attr = analysis.get("causal_attribution", {}) or {}
    conn.execute(
        "INSERT OR REPLACE INTO trace_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trace_id,
            run_id,
            problem_id,
            minimality.get("average"),
            minimality.get("min"),
            minimality.get("max"),
            attribution.get("num_identified_causal_steps"),
            attribution.get("precision"),
            attribution.get("recall"),
            attribution.get("f1_score"),
            repair_metrics.get("total_repairs_attempted"),
            repair_metrics.get("successful_repairs"),
            repair_metrics.get("failed_repairs"),
            repair_metrics.get("success_rate"),
            num_successful_repair_steps,
            len(consensus_list),
            as_int_bool(multi_agent.get("skipped")) if isinstance(multi_agent, dict) else None,
            multi_agent.get("reason") if isinstance(multi_agent, dict) else None,
            as_json(causal_attr.get("causal_steps")),
            as_json(attribution.get("identified_steps")),
        ),
    )


def load_database(bson_paths=BSON_PATHS, db_path=DB_PATH, include_ablations=True):
    db_path = resolve_project_path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    create_schema(conn)

    loaded_runs = 0
    skipped_ablation_runs = 0
    source_files = [bson_paths] if isinstance(bson_paths, (str, Path)) else list(bson_paths)
    source_files = [resolve_project_path(path) for path in source_files]
    missing_files = []
    for bson_path in source_files:
        if not bson_path.exists():
            missing_files.append(str(bson_path))
            continue
        for run in iter_bson_documents(bson_path):
            is_ablation = str(run.get("experiment_name", "")).startswith("ablation_")
            if is_ablation and not include_ablations:
                skipped_ablation_runs += 1
                continue

            insert_run(conn, run, Path(bson_path).name)
            for trace in run.get("passing_traces", []) or []:
                insert_trace_and_steps(conn, run, trace, is_passing_trace=True)
            for trace in run.get("failing_traces", []) or []:
                insert_trace_and_steps(conn, run, trace, is_passing_trace=False)
            loaded_runs += 1

    if loaded_runs == 0:
        raise FileNotFoundError(
            "No BSON run files were loaded. Looked for: " + ", ".join(str(path) for path in source_files)
        )

    conn.commit()
    return conn, {
        "loaded_runs": loaded_runs,
        "skipped_ablation_runs": skipped_ablation_runs,
        "db_path": str(db_path),
        "source_files": [str(path) for path in source_files],
        "missing_source_files": missing_files,
    }


def table_counts(conn):
    tables = [
        "runs",
        "traces",
        "steps",
        "repair_attempts",
        "judge_votes",
        "consensus_steps",
        "trace_metrics",
        "triage_labels",
    ]
    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing_tables = [table for table in tables if table not in existing_tables]
    if missing_tables:
        raise RuntimeError(
            "This SQLite connection does not have the expected Causal Runs schema. "
            f"Missing tables: {missing_tables}. "
            "Rebuild with `conn, load_summary = load_database(include_ablations=True)` "
            "or reconnect to `DB_PATH`."
        )
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}


if __name__ == "__main__":
    connection, summary = load_database()
    print(summary)
    print(table_counts(connection))
