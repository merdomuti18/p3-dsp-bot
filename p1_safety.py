"""P1 state persistence and independent forward-ledger checks.

The OS lock serializes local saves; a stale calculation is rejected, never
merged. GitHub runners are ordered separately by workflow dependencies.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import os
import time


class StateConflict(RuntimeError):
    """Reload state and recompute the whole decision before retrying."""


def file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@contextmanager
def state_lock(path: Path, timeout: float = 10):
    """Advisory process lock, automatically released even after process death."""
    lock_path = path.with_suffix(path.suffix + '.lock')
    with lock_path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
        start = time.monotonic()
        while True:
            try:
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() - start >= timeout:
                    raise StateConflict('P1 state lock timeout')
                time.sleep(.02)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def start_ledger(state: dict, timestamp: str) -> None:
    """Anchor new accounting at the actual current balance, preserving history.

    Earlier unexplained differences are not repaired by this opening record.
    Only entries after start_index are covered by the forward invariant.
    """
    if 'muhasebe_acilis' not in state:
        state['muhasebe_acilis'] = {
            'nakit': state['nakit'],
            'lotlar': {s: p['lotlar'] for s, p in state.get('pozisyonlar', {}).items()},
            'start_index': len(state.get('islem_defteri', [])),
            'zaman': timestamp,
            'kapsam': 'ileri_donuk; tarihsel farklar duzeltilmedi',
        }


def reconcile(state: dict) -> dict:
    """Recompute cash and lots independently using Decimal, not trade helpers."""
    opening = state.get('muhasebe_acilis')
    if opening is None:
        return {'status': 'not_started', 'cash_error': None, 'lot_errors': {}}
    cash = Decimal(str(opening['nakit']))
    lots = dict(opening['lotlar'])
    seen = set()
    entries = state.get('islem_defteri', [])[opening['start_index']:]
    for entry in entries:
        eid = entry['event_id']
        if eid in seen:
            raise ValueError(f'Duplicate ledger event: {eid}')
        seen.add(eid)
        cash += Decimal(str(entry['nakit_etkisi']))
        symbol = entry['symbol']
        quantity = int(entry['lot'])
        if quantity <= 0:
            raise ValueError('Ledger lot must be positive')
        direction = 1 if entry['islem_tipi'] == 'ALIS' else -1
        lots[symbol] = lots.get(symbol, 0) + direction * quantity
        if lots[symbol] < 0:
            raise ValueError(f'Negative ledger lots: {symbol}')
    current = {s: int(p['lotlar']) for s, p in state.get('pozisyonlar', {}).items()}
    errors = {s: lots.get(s, 0)-current.get(s, 0) for s in set(lots)|set(current)
              if lots.get(s, 0) != current.get(s, 0)}
    error = cash - Decimal(str(state['nakit']))
    if abs(error) > Decimal('.01') or errors:
        raise ValueError(f'P1 reconciliation failed: cash={error}, lots={errors}')
    return {'status': 'pass', 'cash_error': float(error), 'lot_errors': errors,
            'event_count': len(entries)}


def closed_daily(frame, now):
    """Daily indicators cannot use today's still-forming or future session."""
    from datetime import datetime, timedelta
    from mott_bist_takvim import to_tsi, bist_seans_saatleri, is_bist_islem_gunu
    ref = to_tsi(now)
    keep = []
    for stamp in frame.index:
        stamp = stamp.to_pydatetime() if hasattr(stamp, 'to_pydatetime') else stamp
        day = to_tsi(stamp).date()
        end = to_tsi(datetime.combine(day, bist_seans_saatleri(day)[1]))
        keep.append(is_bist_islem_gunu(day) and end + timedelta(minutes=10) <= ref)
    return frame.loc[keep].copy()
