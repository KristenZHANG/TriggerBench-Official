"""Temporal helpers shared across TriggerBench generation dimensions."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

TIME_FMT = "%Y-%m-%dT%H:%M"

_UNIT_TO_MINUTES: dict[str, float] = {
    "minute": 1,
    "minutes": 1,
    "min": 1,
    "mins": 1,
    "m": 1,
    "hour": 60,
    "hours": 60,
    "hr": 60,
    "hrs": 60,
    "h": 60,
    "day": 1440,
    "days": 1440,
    "d": 1440,
    "week": 10080,
    "weeks": 10080,
}

_WORD_NUM: dict[str, float] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
    "half": 0.5,
    "a": 1,
    "an": 1,
    "couple": 2,
    "few": 3,
    "several": 4,
}


def parse_iso(timestamp: str) -> datetime | None:
    """Parse ``YYYY-MM-DDTHH:MM``; return None on failure."""
    try:
        return datetime.strptime(timestamp.strip(), TIME_FMT)
    except (ValueError, AttributeError):
        return None


def format_iso(dt: datetime) -> str:
    return dt.strftime(TIME_FMT)


def add_minutes(timestamp: str, minutes: int) -> str:
    dt = datetime.strptime(timestamp, TIME_FMT)
    return format_iso(dt + timedelta(minutes=minutes))


def time_diff_minutes(t1: str, t2: str) -> float:
    """Return ``t2 - t1`` in minutes."""
    return (
        datetime.strptime(t2, TIME_FMT) - datetime.strptime(t1, TIME_FMT)
    ).total_seconds() / 60


def compute_assistant_time(curr_user_time: str, next_user_time: str) -> str:
    """Place the assistant reply 1 minute after the user if the gap is >= 2 min."""
    t_curr = datetime.strptime(curr_user_time, TIME_FMT)
    t_next = datetime.strptime(next_user_time, TIME_FMT)
    delta = (t_next - t_curr).total_seconds() / 60
    t_ast = t_curr + timedelta(minutes=1) if delta >= 2 else t_curr
    return t_ast.strftime(TIME_FMT)


def parse_time_gap_minutes(text: str) -> float | None:
    """Convert a free-text duration to minutes, or None if unparseable.

    Handles patterns such as ``30 minutes``, ``2 hours``, ``half an hour``,
    ``overnight``, and chained values like ``1 hour 30 minutes``.
    """
    raw = text.strip().lower()
    raw = re.sub(r"\b(about|around|roughly|approximately|nearly|~)\b", "", raw).strip()
    raw = re.sub(r"\ba\s+couple\s+(?:of\s+)?", "couple ", raw)
    raw = re.sub(r"\ba\s+few\s+", "few ", raw)

    special: dict[str, float] = {
        "overnight": 480,
        "next day": 1440,
        "next morning": 720,
        "later that day": 360,
        "later that evening": 480,
        "same day": 720,
    }
    for phrase, mins in special.items():
        if phrase in raw:
            return mins

    m = re.match(r"half\s+(?:an?\s+)?(\w+)", raw)
    if m:
        unit = m.group(1)
        if unit in _UNIT_TO_MINUTES:
            return 0.5 * _UNIT_TO_MINUTES[unit]

    total = 0.0
    found = False
    word_alt = "|".join(re.escape(w) for w in _WORD_NUM)
    for num_str, unit in re.findall(rf"([\d]+(?:\.[\d]+)?|{word_alt})\s+(\w+)", raw):
        mult = _UNIT_TO_MINUTES.get(unit.lower())
        if mult is None:
            mult = _UNIT_TO_MINUTES.get(unit.lower().rstrip("s"))
        if mult is None:
            continue
        try:
            num = float(num_str)
        except ValueError:
            num = _WORD_NUM.get(num_str.lower())
            if num is None:
                continue
        total += num * mult
        found = True

    return total if found else None
