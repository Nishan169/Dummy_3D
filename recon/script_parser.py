"""Crime description (free text) -> structured action script.

Uses Claude when ANTHROPIC_API_KEY is set; otherwise a small rule-based fallback that
handles the common 'enters -> walks up -> attacks -> leaves' pattern."""
import json
import os
import re

ACTIONS = ["enter", "walk_to", "attack", "collapse", "leave", "wait"]

SYSTEM = """You convert a short crime-scene description into a JSON action script for a 3D animation.
Return ONLY JSON (no prose, no code fences) shaped like:
{"actors":[{"id":"perp","role":"perpetrator","start":{"hidden":true}},
           {"id":"victim","role":"victim","start":{"at":"chair","pose":"sitting"}}],
 "events":[{"actor":"perp","action":"enter","target":"door"},
           {"actor":"perp","action":"walk_to","target":"victim"},
           {"actor":"perp","action":"attack","target":"victim","style":"stab"},
           {"actor":"perp","action":"leave","target":"door"}]}
Allowed actions: """ + ", ".join(ACTIONS) + """.
attack.style is "stab" or "strike". Movement events may add "pace":"run" (e.g. flees, runs out). Targets are an actor id or a simple noun for a
door/furniture (door, window, chair, bed, sofa, table). The victim's collapse after an
attack is automatic. Events run in order. Use the nouns the description uses."""


def rule_based(prompt):
    t = prompt.lower()
    m = re.search(r"(?:through|via|from) (?:the )?(door|window|gate)", t)
    entry = m.group(1) if m else "door"
    m = re.search(r"(?:sit|sits|sitting|seated|sleeping|lying)\s+(?:down\s+)?(?:on|in|at)\s+(?:the |a )?(\w+)", t)
    seat = m.group(1) if m else "chair"
    hit = re.search(r"\b(hit|hits|strike|strikes|beat|beats|club|clubs|bash|bashes)\b", t)
    style = "strike" if hit and "stab" not in t else "stab"
    run = re.search(r"\b(runs|ran|flees|fled|dashes|sprints|rushes)\b", t)
    return {
        "actors": [{"id": "perp", "role": "perpetrator", "start": {"hidden": True}},
                   {"id": "victim", "role": "victim", "start": {"at": seat, "pose": "sitting"}}],
        "events": [{"actor": "perp", "action": "enter", "target": entry},
                   {"actor": "perp", "action": "walk_to", "target": "victim"},
                   {"actor": "perp", "action": "attack", "target": "victim", "style": style},
                   {"actor": "perp", "action": "leave", "target": entry, **({"pace": "run"} if run else {})}],
    }


def parse_with_claude(prompt):
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=1500,
        system=SYSTEM, messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def parse_script(prompt):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return parse_with_claude(prompt)
        except Exception as e:  # network / bad JSON -> fall back
            print(f"[warn] Claude parsing failed ({e}); using rule-based parser")
    return rule_based(prompt)


def scene_labels(script):
    """Nouns that must be located in the photos (everything that isn't an actor)."""
    ids = {a["id"] for a in script["actors"]}
    out = []
    for a in script["actors"]:
        n = a.get("start", {}).get("at")
        if n:
            out.append(n)
    for e in script["events"]:
        if e.get("target") and e["target"] not in ids:
            out.append(e["target"])
    return list(dict.fromkeys(out))
