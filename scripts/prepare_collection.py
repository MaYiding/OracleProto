"""Freeze an anchored question set and inventory raw observations without scoring."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
FIELDS = "id,choice_type,question_type,event,options,answer,end_time"


def local(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("collection paths must remain inside the local project")
    return path


def read_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(local(path).as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with local(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare() -> dict:
    panel = local(ROOT / "runs/panel_emnlp_2026")
    output = local(ROOT / "runs/collection_300")
    output.mkdir(exist_ok=True)
    manifest = json.loads(local(panel / "manifest.json").read_text())
    anchors = None
    templates = None
    inventory = []
    archive = local(output / "reference_db")
    archive.mkdir(exist_ok=True)
    for model in manifest["models"]:
        path = local(panel / "db" / manifest["model_files"][model])
        conn = read_db(path)
        questions = {row["id"]: dict(row) for row in conn.execute(f"SELECT {FIELDS} FROM questions")}
        prompt = dict(conn.execute("SELECT key,value FROM prompt_templates"))
        if anchors is None:
            anchors, templates = questions, prompt
        if questions != anchors or prompt != templates:
            raise ValueError(f"reference questions or prompts disagree: {model}")
        counts = Counter()
        errors = []
        for row in conn.execute("SELECT * FROM run_results"):
            for sample_idx in range(3):
                prefix = f"s{sample_idx}_"
                counts["sample_slots"] += 1
                if row[prefix + "created_at"]:
                    counts["recorded_samples"] += 1
                if row[prefix + "error"]:
                    errors.append({"question_id": row["question_id"], "sample_idx": sample_idx,
                                   "error": row[prefix + "error"]})
                if row[prefix + "messages_trace"]:
                    counts["message_traces"] += 1
                calls = json.loads(row[prefix + "search_calls"] or "[]")
                counts["search_calls"] += len(calls)
                for call in calls:
                    counts["filter_audits"] += "detector_verdicts" in call
                    counts["dropped_items"] += call.get("n_results_raw", 0) - call.get("n_results_kept", 0)
                    counts["unfiltered_payloads"] += "raw_response" in call
                    counts["filter_reasons"] += "detector_reasons" in call
        target = local(archive / path.name)
        if not target.exists():
            destination = sqlite3.connect(target)
            conn.backup(destination)
            destination.close()
        metadata = dict(conn.execute("SELECT * FROM run_meta LIMIT 1").fetchone())
        inventory.append({"model": model, "source_path": str(path.relative_to(ROOT)),
                          "source_sha256": digest(path), "copy_path": str(target.relative_to(ROOT)),
                          "copy_sha256": digest(target), "counts": dict(counts), "errors": errors,
                          "run_id": metadata["run_id"],
                          "cutoff": manifest["model_training_cutoffs"][model],
                          "stored_cutoff": metadata["training_cutoff"]})
        conn.close()
    assert anchors is not None and templates is not None and len(anchors) == 80
    candidate_path = local(ROOT / "forecast_eval_set_example.db")
    conn = read_db(candidate_path)
    candidates = {row["id"]: dict(row) for row in conn.execute(f"SELECT {FIELDS} FROM test_cases")}
    conn.close()
    overlap = set(anchors) & set(candidates)
    changes = {qid: [key for key in anchors[qid] if anchors[qid][key] != candidates[qid][key]]
               for qid in sorted(overlap) if anchors[qid] != candidates[qid]}
    buckets = defaultdict(list)
    for qid, row in candidates.items():
        if qid not in anchors:
            buckets[(row["question_type"], row["choice_type"])].append(row)
    total_candidates = sum(map(len, buckets.values()))
    if total_candidates < 220:
        raise ValueError("fewer than 220 distinct additional questions")
    # Largest-remainder allocation preserves the candidate pool's type proportions.
    quota = {bucket: 220 * len(rows) // total_candidates for bucket, rows in buckets.items()}
    order = sorted(buckets, key=lambda bucket: (-(220 * len(buckets[bucket]) % total_candidates), bucket))
    for bucket in order[:220 - sum(quota.values())]:
        quota[bucket] += 1
    additions = []
    for bucket in sorted(buckets):
        rows = sorted(buckets[bucket], key=lambda row: hashlib.sha256(("OracleProto-300:" + row["id"]).encode()).hexdigest())
        additions.extend(rows[:quota[bucket]])
    rows = list(anchors.values()) + additions
    destination = local(output / "source.db")
    if not destination.exists():
        conn = sqlite3.connect(destination)
        conn.execute("CREATE TABLE test_cases (id TEXT PRIMARY KEY, choice_type TEXT NOT NULL, "
                     "question_type TEXT NOT NULL, event TEXT NOT NULL, options TEXT NOT NULL, "
                     "answer TEXT NOT NULL, end_time TEXT NOT NULL)")
        conn.executemany("INSERT INTO test_cases VALUES (?,?,?,?,?,?,?)",
                         [tuple(row[key] for key in FIELDS.split(",")) for row in rows])
        conn.execute("CREATE TABLE dataset_metadata (features_json TEXT NOT NULL)")
        conn.execute("INSERT INTO dataset_metadata VALUES (?)", (json.dumps({"prompt_reconstruction": templates}, ensure_ascii=False, sort_keys=True),))
        conn.commit()
        conn.close()
    conn = read_db(destination)
    actual = {row["id"]: dict(row) for row in conn.execute(f"SELECT {FIELDS} FROM test_cases")}
    conn.close()
    if actual != {row["id"]: row for row in rows}:
        raise ValueError("frozen source differs from the deterministic collection plan")
    result = {"source_db": str(destination.relative_to(ROOT)), "source_sha256": digest(destination),
              "candidate_source_sha256": digest(candidate_path), "anchor_ids": sorted(anchors),
              "additional_ids": sorted(row["id"] for row in additions), "reference_inventory": inventory,
              "candidate_overlap": len(overlap), "candidate_changed_fields": changes,
              "selection": "sha256(OracleProto-300:id), largest-remainder allocation over candidate type buckets",
              "question_counts": {"anchors": 80, "additional": 220, "total": 300}}
    local(output / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ["# 300 题采集数据审计", "", "固定题库由原实验 80 题原文与 220 个新增题目组成；原始结果保存在 reference_db。", "",
             "原库的过滤判定不包含被丢弃原文和理由，此缺口不能从过滤后的对话恢复。", "",
             "| 模型配置 | 样本槽 | 对话 | 搜索 | 过滤判定 | 失败 |", "|---|---:|---:|---:|---:|---:|"]
    for item in inventory:
        c = item["counts"]
        lines.append(f"| `{item['model']}` | {c['sample_slots']} | {c['message_traces']} | {c['search_calls']} | {c['filter_audits']} | {len(item['errors'])} |")
    lines.extend(["", "逐题来源、内容差异、数据库 SHA-256 与失败样本见 inventory.json。",
                  "原记录的模型默认思考参数保持未指定；不能根据 token 统计追认成某个明确档位。",
                  "部分原库未存 training_cutoff；panel 清单中的 cutoff 仅作为后续配置来源，原库元数据保持原值。"])
    local(output / "AUDIT.md").write_text("\n".join(lines) + "\n")
    logger.info("Frozen 300 questions: 80 exact anchors + 220 additions; archived {} model databases", len(inventory))
    return result


if __name__ == "__main__":
    prepare()
