#!/usr/bin/env python3
"""Export and rebuild OPD training schedules as prompt-ID tables.

A verl training parquet is consumed in file order (``data.shuffle=False``), so
row ``i`` is the ``i mod B``-th prompt of optimizer step ``i // B``. This tool
stores that order compactly and rebuilds the exact parquet from it.

Identity
--------
``item_sha256``    sha256 of the canonical JSON (sorted keys, UTF-8, no spaces)
                   of the row without ``extra_info``: prompt messages, answer /
                   tests (``reward_model``), ``data_source`` and any other column.
``prompt_sha256``  sha256 of the canonical JSON of the ``prompt`` column only.

``extra_info`` holds per-presentation bookkeeping (selection method, selection
and training seed, presentation index, ...). The first appearance of an item is
stored in ``support.parquet``; every schedule row stores only the top-level
``extra_info`` keys that differ from that first appearance.

Files written by ``export``
---------------------------
``order.csv.gz``    position, step, item_sha256, prompt_sha256, data_source,
                    source_index, source_split, extra_info_delta (JSON)
``support.csv``     distinct items in first-appearance order with counts.
``support.parquet`` the distinct items (full rows at first appearance). Written
                    unless larger than ``--max-support-mb``; in that case the
                    items must be taken from the source pool named in meta.json.
``support_extra_info.jsonl.gz``
                    only when support.parquet is not written: the
                    ``extra_info`` of each item's first appearance, which the
                    source pool row does not carry.
``meta.json``       source path + sha256, row counts, batch size.

``rebuild`` reverses the process and checks that every rebuilt row hashes to
the recorded ``item_sha256``. With ``--verify-against`` it also compares the
result with the original parquet row by row.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXTRA = "extra_info"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def canonical(value) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def item_key(row: dict) -> str:
    return sha256_bytes(canonical({k: v for k, v in row.items() if k != EXTRA}))


def _source_index(row: dict):
    extra = row.get(EXTRA)
    if isinstance(extra, dict):
        for key in ("index", "source_index", "problem_id", "question_id", "id"):
            if extra.get(key) is not None:
                return extra[key], extra.get("split", "")
    if row.get("sample_id") is not None:
        return row["sample_id"], ""
    return "", ""


def _delta(first, current):
    if isinstance(first, dict) and isinstance(current, dict):
        delta = {k: v for k, v in current.items() if k not in first or canonical(first[k]) != canonical(v)}
        removed = [k for k in first if k not in current]
        if removed:
            delta["__removed__"] = removed
        return delta
    return None if canonical(first) == canonical(current) else {"__replace__": current}


def _apply(first, delta):
    if not delta:
        return first
    if "__replace__" in delta:
        return delta["__replace__"]
    merged = dict(first)
    for key in delta.get("__removed__", []):
        merged.pop(key, None)
    merged.update({k: v for k, v in delta.items() if k != "__removed__"})
    return merged


def iter_rows(path: Path, batch_rows: int = 1024):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        yield from batch.to_pylist()


def export(args) -> None:
    src = Path(args.parquet)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limit = args.max_rows if args.max_rows and args.max_rows > 0 else None
    first: dict[str, dict] = {}
    info: dict[str, dict] = {}
    support_bytes = 0
    consumed = 0
    with gzip.open(out / "order.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["position", "step", "item_sha256", "prompt_sha256", "data_source", "source_index", "source_split", "extra_info_delta"]
        )
        for position, row in enumerate(iter_rows(src)):
            if limit is not None and position >= limit:
                break
            key = item_key(row)
            index, split = _source_index(row)
            if key not in first:
                first[key] = row
                info[key] = {
                    "prompt_sha256": sha256_bytes(canonical(row.get("prompt"))),
                    "data_source": row.get("data_source", ""),
                    "source_index": index,
                    "source_split": split,
                    "first_position": position,
                    "count": 0,
                }
                support_bytes += len(canonical(row))
            info[key]["count"] += 1
            delta = _delta(first[key].get(EXTRA), row.get(EXTRA)) if position != info[key]["first_position"] else None
            writer.writerow(
                [
                    position,
                    position // args.batch_size,
                    key,
                    info[key]["prompt_sha256"],
                    info[key]["data_source"],
                    index,
                    split,
                    json.dumps(_plain(delta), sort_keys=True, ensure_ascii=False, default=str) if delta else "",
                ]
            )
            consumed += 1
    with open(out / "support.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "item_sha256", "prompt_sha256", "data_source", "source_index", "source_split", "first_position", "count"]
        )
        for rank, (key, item) in enumerate(info.items()):
            writer.writerow(
                [rank, key, item["prompt_sha256"], item["data_source"], item["source_index"], item["source_split"],
                 item["first_position"], item["count"]]
            )
    stored = support_bytes <= args.max_support_mb * 1e6
    if stored:
        pq.write_table(pa.Table.from_pylist(list(first.values())), out / "support.parquet", compression="zstd")
    else:
        with gzip.open(out / "support_extra_info.jsonl.gz", "wt") as handle:
            for key, row in first.items():
                handle.write(json.dumps({"item_sha256": key, EXTRA: _plain(row.get(EXTRA))}, sort_keys=True,
                                        ensure_ascii=False, default=str) + "\n")
    meta = {
        "schema_version": "opd_schedule_ids_v2",
        "source_parquet": str(src),
        "source_sha256": None if args.skip_source_hash else sha256_file(src),
        "source_rows": pq.ParquetFile(src).metadata.num_rows,
        "rows_consumed": consumed,
        "batch_size": args.batch_size,
        "steps_covered": (consumed + args.batch_size - 1) // args.batch_size,
        "distinct_items": len(info),
        "distinct_prompts": len({v["prompt_sha256"] for v in info.values()}),
        "support_parquet_stored": stored,
        "support_content_mb": round(support_bytes / 1e6, 3),
        "pool": args.pool or None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


def rebuild(args) -> None:
    ids = Path(args.ids)
    items: dict[str, dict] = {}
    for source in args.rows:
        for row in iter_rows(Path(source)):
            items.setdefault(item_key(row), row)
    first_extra = {}
    if (ids / "support_extra_info.jsonl.gz").exists():
        with gzip.open(ids / "support_extra_info.jsonl.gz", "rt") as handle:
            for line in handle:
                record = json.loads(line)
                first_extra[record["item_sha256"]] = record[EXTRA]
    with gzip.open(ids / "order.csv.gz", "rt") as handle:
        order = list(csv.DictReader(handle))
    missing = sorted({r["item_sha256"] for r in order} - items.keys())
    if missing:
        raise SystemExit(f"{len(missing)} referenced items not found in --rows, e.g. {missing[:3]}")
    rows = []
    for record in order:
        base = dict(items[record["item_sha256"]])
        if record["item_sha256"] in first_extra:
            base[EXTRA] = first_extra[record["item_sha256"]]
        row = dict(base)
        delta = json.loads(record["extra_info_delta"]) if record["extra_info_delta"] else None
        if EXTRA in row or delta:
            row[EXTRA] = _apply(base.get(EXTRA), delta)
        if item_key(row) != record["item_sha256"]:
            raise SystemExit(f"hash mismatch at position {record['position']}")
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    if args.verify_against:
        original = iter_rows(Path(args.verify_against))
        for position, (mine, theirs) in enumerate(zip(rows, original)):
            if canonical(mine) != canonical(theirs):
                raise SystemExit(f"row {position} differs from {args.verify_against}")
        print(f"verified {len(rows)} rows against {args.verify_against}")
    out = Path(args.out)
    partial = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    pq.write_table(table, partial, compression="zstd")
    os.replace(partial, out)
    print(f"wrote {len(rows)} rows to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("parquet")
    e.add_argument("out")
    e.add_argument("--batch-size", type=int, required=True)
    e.add_argument("--max-rows", type=int, default=0, help="rows consumed by training (data.train_max_samples)")
    e.add_argument("--max-support-mb", type=float, default=50.0)
    e.add_argument("--pool", default="", help="source pool the items were drawn from")
    e.add_argument("--skip-source-hash", action="store_true")
    e.set_defaults(func=export)
    r = sub.add_parser("rebuild")
    r.add_argument("ids", help="directory written by export")
    r.add_argument("--rows", nargs="+", required=True, help="support.parquet and/or source pool parquet(s)")
    r.add_argument("--out", required=True)
    r.add_argument("--verify-against", default="")
    r.set_defaults(func=rebuild)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
