"""Picks today's posting times from audience activity + past reel performance."""
import random
from collections import defaultdict
from datetime import datetime, time as dtime

import config

# How much each signal counts toward a post's "performance score".
WEIGHTS = {"views": 1, "likes": 5, "comments": 15, "shares": 20, "saved": 20}


def performance(metrics):
    return sum(metrics.get(k, 0) * w for k, w in WEIGHTS.items())


def _normalise(d):
    peak = max(d.values(), default=0) or 1
    return {k: v / peak for k, v in d.items()}


def hour_scores(state, online):
    scores = {h: 0.0 for h in config.ACTIVE_HOURS}

    # 1) When followers are online (60%)
    if online:
        ref = datetime.now(config.ONLINE_FOLLOWERS_TZ).replace(minute=0, second=0, microsecond=0)
        local = {ref.replace(hour=h).astimezone(config.LOCAL_TZ).hour: v for h, v in online.items()}
        for h, v in _normalise(local).items():
            if h in scores:
                scores[h] += 0.6 * v

    # 2) When our own reels actually performed best (40%)
    perf = defaultdict(list)
    for p in state["posts"]:
        if p.get("metrics"):
            perf[p["local_hour"]].append(performance(p["metrics"]))
    avg = {h: sum(v) / len(v) for h, v in perf.items()}
    for h, v in _normalise(avg).items():
        if h in scores:
            scores[h] += 0.4 * v
    return scores


def _default_hours():
    slots = [dtime.fromisoformat(s.strip()) for s in config.DEFAULT_SLOTS.split(",") if s.strip()]
    if len(slots) >= config.POSTS_PER_DAY:
        return slots[: config.POSTS_PER_DAY]
    hours = list(config.ACTIVE_HOURS)
    step = max(1, len(hours) // config.POSTS_PER_DAY)
    return [dtime(hours[i * step], 0) for i in range(config.POSTS_PER_DAY)]


def plan_day(state, online):
    today = datetime.now(config.LOCAL_TZ).date()
    scores = hour_scores(state, online)

    if not any(scores.values()):
        times = _default_hours()
    else:
        ranked = sorted(scores, key=scores.get, reverse=True)
        chosen = []
        for h in ranked:  # best hours, spaced apart
            if all(abs(h - c) >= config.MIN_GAP_HOURS for c in chosen):
                chosen.append(h)
            if len(chosen) == config.POSTS_PER_DAY:
                break
        for h in ranked:  # fill if spacing made it impossible
            if len(chosen) >= config.POSTS_PER_DAY:
                break
            if h not in chosen:
                chosen.append(h)

        # Exploration: occasionally test an hour we have no data for,
        # so the system doesn't lock onto its first guesses.
        tested = {p["local_hour"] for p in state["posts"]}
        untested = [h for h in config.ACTIVE_HOURS if h not in tested and h not in chosen]
        if len(chosen) > 1 and untested and random.random() < config.EXPLORE_RATE:
            chosen[-1] = random.choice(untested)

        times = [dtime(h, random.randint(0, 20)) for h in chosen]

    return sorted(datetime.combine(today, t, config.LOCAL_TZ).isoformat() for t in times)
