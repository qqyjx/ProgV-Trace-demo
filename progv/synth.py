"""Seeded generator of synthetic toy cases (same schema as the JSON files), used to train the toy student."""
from __future__ import annotations

import random

from .claims import LOCATIONS, Case

_EVENTS = {"flood": "flooding", "fire": "fire", "protest": "protest", "storm": "storm",
           "press conference": "press conference"}
_TITLES = ("Breaking: {ev} in {city} on {date}", "Footage shows the {ev} in {city}, filmed {date}",
           "{ev} reported in {city} on {date}")
# (original line, altered line)
_ALTERED = (
    ("We will not open the dam gates tonight.", "We will open the dam gates tonight."),
    ("Two people were treated for smoke inhalation.", "Nobody was hurt."),
    ("The bridge remains closed until Friday.", "The bridge reopens on Monday."),
    ("Evacuation is voluntary at this stage.", "Evacuation is mandatory for all residents."),
    ("The council approved the plan by a narrow vote.", "The council rejected the plan."),
    ("Water levels have started to fall.", "Water levels are still rising."),
)
_NEUTRAL = ("Crews are on site.", "Updates follow later today.", "Residents are asked to stay calm.",
            "The road is open to local traffic.", "Officials will brief the press at noon.",
            "Power was restored in the evening.")


def generate_case(rng: random.Random, idx: int, p_loc: float = 0.3, p_time: float = 0.25,
                  p_speech: float = 0.3, p_missing_quotes: float = 0.1) -> Case:
    ev_key = rng.choice(sorted(_EVENTS))
    city, wrong_city = rng.sample(LOCATIONS, 2)
    year = rng.randint(2020, 2024)
    date = f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    wrong_date = f"{year + 2}{date[4:]}"
    forge_loc, forge_time = rng.random() < p_loc, rng.random() < p_time
    forge_speech = rng.random() < p_speech
    has_quotes = rng.random() >= p_missing_quotes
    n_seg = rng.randint(2, 3)
    altered_at = rng.randrange(n_seg) if forge_speech else -1
    lines = list(_NEUTRAL)
    rng.shuffle(lines)
    orig, shown = [], []
    for i in range(n_seg):
        t0 = 1.0 + 7.0 * i
        if i == altered_at:
            o, a = rng.choice(_ALTERED)
        else:
            o = a = lines[i]
        orig.append({"t0": t0, "t1": t0 + 5.0, "text": o})
        shown.append({"id": f"s{i + 1}", "t0": t0, "t1": t0 + 5.0, "text": a})
    end = orig[-1]["t1"] + 1.0
    title = rng.choice(_TITLES).format(ev=_EVENTS[ev_key], city=wrong_city if forge_loc else city,
                                       date=wrong_date if forge_time else date)
    ocr = [{"t0": 0.0, "t1": end, "text": f"{city} Cam 2"}] if rng.random() < 0.5 else []
    evidence = [
        {"id": "ev1", "type": "reverse_image", "source": "Toy Archive (synthetic)", "queries": ["frame:f1"],
         "text": f"Archive frame, {city}, {date}", "facts": {"location": city, "time": date, "event": ev_key}},
        {"id": "ev2", "type": "fact_check", "source": "Toy Fact Desk (synthetic)", "queries": ["title"],
         "text": "Fact-check note.", "facts": {"event": ev_key}},
    ]
    if has_quotes:
        evidence.append({"id": "ev3", "type": "news", "source": "Toy Transcript (synthetic)",
                         "queries": ["speech"], "text": "Official transcript.", "quotes": orig})
    manip = (["title.location"] if forge_loc else []) + (["title.time"] if forge_time else []) \
        + ([f"speech.s{altered_at + 1}"] if forge_speech and has_quotes else [])
    return Case.from_dict({
        "id": f"gen-{idx}", "synthetic": True, "title": title, "ocr": ocr,
        "frames": [{"id": "f1", "t0": 0.0, "t1": end, "visual": {"event": ev_key, "landmark": "n/a"}}],
        "speech": shown, "evidence": evidence, "meta": {"manipulation": manip}})


def generate_cases(n: int, seed: int) -> list[Case]:
    rng = random.Random(seed)
    return [generate_case(rng, i) for i in range(n)]
