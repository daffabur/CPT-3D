"""Crash-aware, single-writer benchmark for the frozen V3 24K plan.

The CLI deliberately caps generation at 300 and refuses the production root.
Production enablement requires a separate reviewed change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import uuid

from .v3_24k_plan import CONFIG_PATH, validate_plan
from .v3_dataset_production import _files, canonical, digest, file_sha, make_volume, validate_entry
from synthetic3d.v2.geotechnical_v3 import save

PLAN_SHA256 = 'ff2884e0370e9597cb3afa3f8c0b613f19670835fc96ff2dc26cdebf0990d663'
JOURNAL_SCHEMA = 'schemagan-v3-24k-journal-1'
MANIFEST_SCHEMA = 'schemagan-v3-pilot-manifest-1'  # Accepted by the unchanged V3 adapter.
FAULT_POINTS = ('before_rename', 'after_rename', 'before_journal', 'after_journal',
                'partial_journal', 'before_checkpoint', 'during_checkpoint', 'after_checkpoint')


class InjectedCrash(RuntimeError):
    pass


def _hit(point, requested):
    if requested == point:
        raise InjectedCrash(point)


def _sync_dir(path):
    """Return False if the filesystem does not support directory fsync."""
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
            return True
        except OSError as exc:
            if exc.errno in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
                return False
            raise
    finally:
        os.close(fd)


def _write_atomic(path, value, fault=None):
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    _hit('during_checkpoint', fault)
    os.replace(temp, path)
    supported = _sync_dir(path.parent)
    return supported


@contextmanager
def writer_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / '.writer.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('dataset has an active writer lock') from exc
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({'pid': os.getpid(), 'since_utc': datetime.now(timezone.utc).isoformat()}).encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_frozen(plan_path, config_path=CONFIG_PATH):
    config = json.loads(Path(config_path).read_text())
    plan = json.loads(Path(plan_path).read_text())
    status = validate_plan(plan, config, expected_hash=PLAN_SHA256, check_determinism=True)
    return plan, config, status


def benchmark_selection(plan, count=300):
    """Fixed stratified selection: six splits, three grids, varied CPT and families."""
    if count < 1 or count > 300:
        raise ValueError('benchmark count must be in 1..300')
    entries = plan['entries']
    buckets = {}
    for index, entry in enumerate(entries):
        key = (entry['split'], entry['grid_id'], entry['family'], entry['n_cpt'])
        buckets.setdefault(key, []).append(index)
    keys = sorted(buckets)
    chosen = []
    round_number = 0
    while len(chosen) < count:
        progressed = False
        for key in keys:
            if round_number < len(buckets[key]):
                chosen.append(buckets[key][round_number])
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            raise ValueError('not enough plan entries')
        round_number += 1
    return chosen


def _record(entry, plan_index, sequence):
    body = {'schema_version': JOURNAL_SCHEMA, 'journal_sequence': sequence,
            'plan_index': plan_index, 'realization_id': entry['realization_id'],
            'split': entry['split'], 'lineage_group': entry['lineage_group'],
            'fingerprint': entry['fingerprint'], 'file_hashes': entry['file_sha256'],
            'commit_timestamp': datetime.now(timezone.utc).isoformat(), 'entry': entry}
    return {**body, 'record_sha256': digest(body)}


def _validate_record(record, plan):
    body = {k: v for k, v in record.items() if k != 'record_sha256'}
    if record.get('schema_version') != JOURNAL_SCHEMA or record.get('record_sha256') != digest(body):
        raise ValueError('invalid journal record checksum/schema')
    index = record['plan_index']
    if not isinstance(index, int) or not 0 <= index < 24000:
        raise ValueError('invalid journal plan index')
    planned = plan['entries'][index]
    entry = record['entry']
    for key in ('realization_id', 'split', 'lineage_group'):
        if record[key] != planned[key] or entry[key] != planned[key]:
            raise ValueError('journal identity mismatch')
    if record['fingerprint'] != entry['fingerprint'] or record['file_hashes'] != entry['file_sha256']:
        raise ValueError('journal content mismatch')
    return entry


def _read_journal(root, plan, *, recover_torn=True):
    path = root / 'journal.jsonl'
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    complete = raw.endswith(b'\n') or not raw
    lines = raw.split(b'\n')
    if not complete:
        if not recover_torn:
            raise ValueError('torn journal tail; resume under writer lock to recover')
        tail = lines.pop()
        evidence = root / ('journal.torn-' + uuid.uuid4().hex + '.bin')
        with evidence.open('xb') as f:
            f.write(tail)
            f.flush()
            os.fsync(f.fileno())
        with path.open('r+b') as f:
            f.truncate(len(raw) - len(tail))
            f.flush()
            os.fsync(f.fileno())
        _sync_dir(root)
    records = []
    seen = set()
    for line in lines:
        if not line:
            continue
        try:
            record = json.loads(line)
            entry = _validate_record(record, plan)
        except Exception as exc:
            raise ValueError(f'invalid complete journal record at line {len(records)+1}') from exc
        if record['journal_sequence'] != len(records) + 1 or entry['realization_id'] in seen:
            raise ValueError('journal sequence or duplicate commit')
        seen.add(entry['realization_id'])
        records.append(record)
    return records, not complete


def _append_journal(root, record, fault=None):
    payload = (canonical(record) + '\n').encode()
    fd = os.open(root / 'journal.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if fault == 'partial_journal':
            os.write(fd, payload[:max(1, len(payload) // 2)])
            os.fsync(fd)
            raise InjectedCrash('partial_journal')
        if os.write(fd, payload) != len(payload):
            raise OSError('short journal write')
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync_dir(root)


def _manifest(plan, records):
    return {'schema_version': MANIFEST_SCHEMA, 'config_sha256': plan['config_sha256'],
            'plan_sha256': PLAN_SHA256, 'journal_sequence': len(records),
            'journal_last_sha256': records[-1]['record_sha256'] if records else None,
            'entries': [r['entry'] for r in records]}


def checkpoint(root, plan, records, fault=None):
    _hit('before_checkpoint', fault)
    start = time.perf_counter()
    supported = _write_atomic(root / 'manifest.json', _manifest(plan, records), fault)
    _hit('after_checkpoint', fault)
    return {'seconds': time.perf_counter() - start, 'directory_fsync_supported': supported}


def _check_checkpoint(root, plan, records, *, preserve_invalid=True):
    path = root / 'manifest.json'
    if not path.exists():
        return 0, False
    try:
        saved = json.loads(path.read_text())
        seq = saved['journal_sequence']
        if not isinstance(seq, int) or seq < 0 or seq > len(records):
            raise ValueError('checkpoint sequence')
        if saved != _manifest(plan, records[:seq]):
            raise ValueError('checkpoint journal mismatch')
        return seq, True
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        if preserve_invalid:
            evidence = root / ('manifest.invalid-' + uuid.uuid4().hex + '.json')
            shutil.copy2(path, evidence)
        return 0, False


def _sidecar(root, entry):
    path = root / entry['realization_id'] / '_entry.json'
    if not path.is_file():
        raise ValueError(f'missing sidecar: {path}')
    saved = json.loads(path.read_text())
    if saved != entry:
        raise ValueError(f'sidecar mismatch: {path}')


def resume(root, plan, config, *, checkpoint_now=True):
    """Fast resume: all sidecars/presence; hash tail and every 64th older volume."""
    records, torn = _read_journal(root, plan)
    checkpoint_seq, checkpoint_valid = _check_checkpoint(root, plan, records)
    by_id = {r['realization_id']: r for r in records}
    for i, record in enumerate(records):
        entry = record['entry']
        _sidecar(root, entry)
        for rel in _files(entry):
            p = (root / rel).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():
                raise ValueError(f'missing committed file: {rel}')
        if i >= checkpoint_seq or i % 64 == 0 or i == checkpoint_seq - 1:
            validate_entry(root, entry, plan['entries'][record['plan_index']], config, reproduce=False)
    recovered = 0
    indices = {e['realization_id']: i for i, e in enumerate(plan['entries'])}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'):
            continue
        if folder.name not in indices:
            raise ValueError(f'unplanned directory: {folder}')
        if folder.name in by_id:
            continue
        sidecar = folder / '_entry.json'
        if not sidecar.is_file():
            raise ValueError(f'orphan without sidecar: {folder}')
        entry = json.loads(sidecar.read_text())
        validate_entry(root, entry, plan['entries'][indices[folder.name]], config, reproduce=False)
        record = _record(entry, indices[folder.name], len(records) + 1)
        _append_journal(root, record)
        records.append(record)
        by_id[folder.name] = record
        recovered += 1
    if checkpoint_now and (not checkpoint_valid or recovered or torn or len(records) != checkpoint_seq):
        checkpoint(root, plan, records)
    return records, {'committed': len(records), 'checkpoint_valid': checkpoint_valid,
                     'checkpoint_sequence': checkpoint_seq, 'hashed_since_checkpoint': len(records) - checkpoint_seq,
                     'checkpointed_content_policy': 'sidecar equality and file presence for every entry; SHA/physics for every 64th entry and final checkpointed entry; full validation required for all hashes',
                     'checkpointed_sample_hashed': sum(i % 64 == 0 or i == checkpoint_seq - 1 for i in range(checkpoint_seq)),
                     'recovered_orphans': recovered, 'torn_journal_recovered': torn}


def write_volume(root, planned, config, plan_index, sequence, fault=None):
    rid = planned['realization_id']
    final = root / rid
    if final.exists():
        raise FileExistsError(final)
    stage = root / ('.staging-' + rid + '-' + uuid.uuid4().hex)
    stage.mkdir()
    timings = {}
    try:
        start = time.perf_counter()
        volume, observed = make_volume(planned, config)
        timings['generate_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        saved = save(volume, observed, stage, planned['split'], planned['lineage_group'], rid)
        entry = {**planned, **saved}
        entry.update({key: planned[key] for key in planned})
        entry['file_sha256'] = {rel: file_sha(stage / rel) for rel in _files(entry)}
        entry['stored_bytes'] = sum((stage / rel).stat().st_size for rel in _files(entry))
        sidecar = stage / rid / '_entry.json'
        sidecar.write_text(json.dumps(entry, indent=2) + '\n')
        for file in (stage / rid).iterdir():
            with file.open('rb') as stream:
                os.fsync(stream.fileno())
        _sync_dir(stage / rid)
        validate_entry(stage, entry, planned, config, reproduce=False)
        timings['write_validate_seconds'] = time.perf_counter() - start
        _hit('before_rename', fault)
        os.rename(stage / rid, final)
        _sync_dir(root)
        _hit('after_rename', fault)
        _hit('before_journal', fault)
        record = _record(entry, plan_index, sequence)
        start = time.perf_counter()
        _append_journal(root, record, fault)
        timings['journal_seconds'] = time.perf_counter() - start
        _hit('after_journal', fault)
        return record, timings
    finally:
        # Empty stage only. Completed orphan volumes and evidence are never deleted.
        if stage.exists() and not any(stage.iterdir()):
            stage.rmdir()


def validate_full(root, plan, config, *, reproduce=False):
    records, _ = _read_journal(root, plan, recover_torn=False)
    seq, valid = _check_checkpoint(root, plan, records, preserve_invalid=False)
    if not valid or seq != len(records):
        raise ValueError('manifest checkpoint not current/valid')
    ids = set(); fingerprints = set(); lineages = set(); splits = Counter()
    start = time.perf_counter()
    for record in records:
        entry = record['entry']
        _sidecar(root, entry)
        validate_entry(root, entry, plan['entries'][record['plan_index']], config, reproduce=reproduce)
        rid = entry['realization_id']
        if rid in ids or entry['fingerprint'] in fingerprints or entry['lineage_group'] in lineages:
            raise ValueError(f'duplicate identity/fingerprint/lineage: {rid}')
        ids.add(rid); fingerprints.add(entry['fingerprint']); lineages.add(entry['lineage_group'])
        splits[entry['split']] += 1
    extras = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.') and p.name not in ids]
    if extras:
        raise ValueError(f'unindexed volume: {extras[:3]}')
    return {'valid': True, 'volumes': len(records), 'split_counts': dict(splits),
            'fingerprints_unique': len(fingerprints), 'lineages_unique': len(lineages),
            'seed_reproduction': reproduce, 'seconds': time.perf_counter() - start}


def _run_selected(root, plan, config, selected, target, *, interval=250, fault=None, benchmark=False):
    with writer_lock(root):
        provenance = root / 'provenance.json'
        expected = {'plan_sha256': PLAN_SHA256, 'config_sha256': digest(config), 'selected_indices': selected}
        if provenance.exists():
            if json.loads(provenance.read_text()) != expected:
                raise ValueError('benchmark provenance mismatch')
        else:
            _write_atomic(provenance, expected)
        start = time.perf_counter()
        records, resume_status = resume(root, plan, config)
        if (benchmark and len(records) > 300) or any(r['plan_index'] not in selected for r in records):
            raise ValueError('dataset contains out-of-scope commits')
        resume_seconds = time.perf_counter() - start
        existing = {r['realization_id'] for r in records}
        timings = []
        checkpoint_times = []
        for index in selected[:target]:
            planned = plan['entries'][index]
            if planned['realization_id'] in existing:
                continue
            record, timing = write_volume(root, planned, config, index, len(records) + 1, fault)
            records.append(record)
            existing.add(record['realization_id'])
            timings.append(timing)
            if len(records) % interval == 0:
                checkpoint_times.append(checkpoint(root, plan, records, fault))
                print(json.dumps({'event': 'checkpoint', 'committed': len(records), 'target': target}), file=sys.stderr, flush=True)
        if timings or not (root / 'manifest.json').exists():
            checkpoint_times.append(checkpoint(root, plan, records))
        print(json.dumps({'event': 'batch_complete', 'committed': len(records), 'target': target}), file=sys.stderr, flush=True)
        return {'target': target, 'total': len(records), 'new_volumes': len(timings),
                'resume_seconds': resume_seconds, 'resume': resume_status,
                'timings': timings, 'checkpoint_times': checkpoint_times,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run_benchmark(root, plan, config, selected, target, *, interval=250, fault=None):
    root = Path(root).resolve()
    validate_plan(plan, config, expected_hash=PLAN_SHA256, check_determinism=True)
    production_root = (Path(__file__).resolve().parents[1] / config['output_root']).resolve()
    if (root == production_root or target < 1 or target > 300 or
            selected != benchmark_selection(plan) or target > len(selected) or interval < 1):
        raise ValueError('benchmark root/target unsafe')
    return _run_selected(root, plan, config, selected, target, interval=interval, fault=fault, benchmark=True)


def run_approved_production(root, plan, config, *, approval_plan_sha256=None, interval=250, target=24000):
    """Explicit later authorization gate; never invoked by the benchmark CLI."""
    validate_plan(plan, config, expected_hash=PLAN_SHA256, check_determinism=True)
    production_root = (Path(__file__).resolve().parents[1] / config['output_root']).resolve()
    if (Path(root).resolve() != production_root or approval_plan_sha256 != PLAN_SHA256 or
            interval < 1 or target not in (1000, 4000, 8000, 16000, 24000)):
        raise PermissionError('24K production requires a separate explicit approval and exact root/plan digest')
    selected = list(range(24000))
    return _run_selected(production_root, plan, config, selected, target, interval=interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--config', type=Path, default=CONFIG_PATH)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--target', type=int, choices=(50, 150, 300), required=True)
    p.add_argument('--interval', type=int, default=250)
    args = p.parse_args()
    plan, config, status = load_frozen(args.plan, args.config)
    result = run_benchmark(args.root, plan, config, benchmark_selection(plan), args.target, interval=args.interval)
    print(json.dumps({'plan_sha256': status['canonical_plan_sha256'], **result}))


if __name__ == '__main__':
    main()
