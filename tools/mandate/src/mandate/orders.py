"""Work orders: the plan applied to the registry.

One order is one stage on one group of units. Its id is a hash of everything that defines the
work: the stage, the units, the resolved decisions, the pinned versions and the exact upstream
results it reads. Change any of them and it is a different order; the old one stays in the
ledger, superseded. An ensemble order (a PCA basis, a night's calibration) lists its members
and the ones it dropped, so "which reference stars went into this basis" is a file, not a memory.
"""

from __future__ import annotations

import dataclasses
import json

from .model import Campaign, canonical, matches, option_names, sha, slug

SETTLED_OK = ("pass", "warn")
SETTLED_BAD = ("fail", "error", "blocked")


@dataclasses.dataclass
class Order:
    stage: str
    key: dict
    units: list[dict]
    decisions: dict = dataclasses.field(default_factory=dict)
    inputs: dict = dataclasses.field(default_factory=dict)
    dropped: dict = dataclasses.field(default_factory=dict)
    id: str | None = None
    state: str = "waiting"
    reason: str = ""
    attempts: list[dict] = dataclasses.field(default_factory=list)
    waiting: list[str] = dataclasses.field(default_factory=list)  # upstream orders not settled yet

    @property
    def key_str(self) -> str:
        return "__".join(slug(v) for v in self.key.values()) or "all"

    @property
    def ref(self) -> str:
        return f"{self.stage}/{self.id or self.key_str}"

    @property
    def digest(self) -> str | None:
        return self.attempts[-1].get("digest") if self.attempts else None

    def value(self, column: str) -> str | None:
        """The column's value when every unit of the order shares it."""
        values = {u.get(column, "") for u in self.units}
        return values.pop() if len(values) == 1 else None

    def number(self, column: str) -> float | None:
        try:
            values = [float(u[column]) for u in self.units]
        except (KeyError, ValueError):
            return None
        return sum(values) / len(values) if values else None


def excluded(c: Campaign, unit: dict, stage: str) -> dict | None:
    for e in c.active("exclude"):
        if (not e.get("stages") or stage in e["stages"]) and matches(unit, e["where"]):
            return e
    return None


def unit_hash(c: Campaign, stage: str, unit: dict) -> str:
    """The unit as this stage is given it. A corrected registry row is different work; columns
    the plan lists under ``ignore_columns`` and the ones this stage itself writes are left out."""
    st = c.stages[stage]
    skip = c.ignore_columns | set((st.get("promote") or {}).values())
    row = {k: v for k, v in c.open_row(unit, c.level(stage)).items() if k not in skip}
    return sha(canonical(row), 12)


def resolve_decisions(c: Campaign, stage: str, order: Order) -> dict:
    """Default, then the first rule every unit satisfies, then the docket. Each with its source."""
    out = {}
    for name, d in (c.stages[stage].get("decisions") or {}).items():
        out[name] = {"value": d.get("default"), "source": "default"}
        column = d.get("default_from")  # the default is a registry column's value, where every unit shares one
        if column:
            seen = {str(c.open_row(u, c.level(stage)).get(column, "") or "") for u in order.units}
            if len(seen) == 1 and seen != {""} and next(iter(seen)) in option_names(d):
                out[name] = {"value": next(iter(seen)), "source": f"column:{column}"}
        for i, rule in enumerate(d.get("rules") or []):
            if all(matches(c.open_row(u, c.level(stage)), rule.get("when") or {}) for u in order.units):
                out[name] = {"value": rule["use"], "source": f"rule:{i}"}
                break
    for e in c.active("decide"):
        if e["stage"] == stage and e["key"] == order.key_str:
            for name, value in e["set"].items():
                if name in out:
                    out[name] = {"value": value, "source": f"docket:{e['id']}"}
    return out


def stage_identity(c: Campaign, stage: str) -> dict:
    st = c.stages[stage]
    ident = {k: st.get(k) for k in ("run", "worker", "result", "status_key", "requires", "expects", "needs", "digest")}
    if st.get("prompt") and (c.root / st["prompt"]).exists():
        ident["prompt"] = sha((c.root / st["prompt"]).read_text(), 16)
    versions = c.plan.get("versions") or {}
    ident["pins"] = {n: versions[n].get("pin") for n in (st.get("uses") or versions) if n in versions}
    return ident


def upstream(name: str, need: dict | None) -> str:
    """The stage a need reads. Its name, unless it says ``stage:`` (two needs may read one stage)."""
    return (need or {}).get("stage", name)


def read_attempts(c: Campaign, stage: str, order_id: str) -> list[dict]:
    folder = c.root / "ledger" / stage / order_id
    return [json.loads(p.read_text()) for p in sorted(folder.glob("attempt-*.json"))] if folder.exists() else []


def expand(c: Campaign) -> dict[str, list[Order]]:
    """Every order of every stage, with its state: waiting, blocked, ready, parked or a result."""
    parked = {(e["stage"], e.get("key")) for e in c.pending() if e["kind"] == "request"}
    out: dict[str, list[Order]] = {}
    for stage in c.order_of_stages():
        st = c.stages[stage]
        by = st.get("by", [c.unit_id])
        groups: dict[tuple, Order] = {}
        for u in c.units:
            if not matches(u, st.get("over")) or excluded(c, u, stage):
                continue
            key = {col: u.get(col, "") for col in by}
            groups.setdefault(tuple(key.values()), Order(stage, key, [])).units.append(u)
        orders = sorted(groups.values(), key=lambda o: o.key_str)
        index: dict[str, dict[tuple, list[Order]]] = {}
        for name, need in (st.get("needs") or {}).items():
            match = (need or {}).get("match") or []
            index[name] = {}
            for p in out.get(upstream(name, need), []):
                index[name].setdefault(tuple(p.value(col) for col in match), []).append(p)
        gone: dict[str, bool] = {}  # upstream orders whose every unit is excluded from this stage
        for o in orders:
            _resolve(c, o, index, gone, parked)
        out[stage] = orders
    return out


def _resolve(c: Campaign, o: Order, index: dict, gone: dict[str, bool], parked: set) -> None:
    st = c.stages[o.stage]
    o.decisions = resolve_decisions(c, o.stage, o)
    waiting, short = [], []
    for name, need in (st.get("needs") or {}).items():
        need = need or {}
        up = upstream(name, need)
        mine = tuple(o.value(col) for col in need.get("match") or [])
        candidates = [] if None in mine else index[name].get(mine, [])
        if need.get("link"):
            # Only upstream orders with a unit the registry links to one of ours, in either direction.
            partners = set().union(*(c.linked(need["link"], u.get(c.unit_id, "")) for u in o.units))
            candidates = [p for p in candidates if any(u.get(c.unit_id) in partners for u in p.units)]
        members, dropped = [], []
        for p in candidates:
            if p.ref not in gone:
                gone[p.ref] = all(excluded(c, u, o.stage) for u in p.units)
            if gone[p.ref]:
                dropped.append({"stage": up, "key": p.key_str, "why": "excluded"})
            elif p.state in SETTLED_OK:
                members.append(p)
            elif p.state in SETTLED_BAD:
                dropped.append({"stage": up, "key": p.key_str, "why": p.state})
            else:
                waiting.append(p.ref)
        near = need.get("nearest")
        if near and not waiting:
            here = o.number(near["by"])
            ranked = [p for p in members if p.number(near["by"]) is not None and here is not None]
            ranked.sort(key=lambda p: (abs(p.number(near["by"]) - here), p.key_str))
            kept = ranked[: int(near.get("k", 1))]
            if near.get("within") is not None:
                kept = [p for p in kept if abs(p.number(near["by"]) - here) <= float(near["within"])]
            dropped += [{"stage": up, "key": p.key_str, "why": "not nearest"} for p in members if p not in kept]
            members = kept
        o.inputs[name] = [{"stage": up, "order": p.id, "key": p.key_str, "status": p.state, "digest": p.digest}
                          for p in members]
        o.dropped[name] = dropped
        if len(members) < int(need.get("min", 1)):
            short.append(f"{name}: {len(members)} of at least {need.get('min', 1)}")
    o.waiting = waiting
    if waiting:
        o.state, o.reason = "waiting", "on " + ", ".join(waiting[:3]) + (" ..." if len(waiting) > 3 else "")
        return
    if short:
        o.state, o.reason = "blocked", "; ".join(short)
        return
    identity = {
        "stage": o.stage,
        "def": stage_identity(c, o.stage),
        "key": o.key,
        "units": sorted((u.get(c.unit_id, ""), unit_hash(c, o.stage, u)) for u in o.units),
        "decisions": {k: v["value"] for k, v in o.decisions.items()},
        "inputs": {up: [(m["order"], m["digest"]) for m in ms] for up, ms in o.inputs.items()},
    }
    o.id = f"{o.key_str}-{sha(canonical(identity))}"
    o.attempts = read_attempts(c, o.stage, o.id)
    if (o.stage, o.key_str) in parked or (o.stage, None) in parked:
        o.state, o.reason = "parked", "an open request"
    elif o.attempts:
        o.state = o.attempts[-1]["status"]
    else:
        o.state = "ready"


def find(c: Campaign, ref: str, orders: dict[str, list[Order]] | None = None) -> Order:
    """An order by ``stage/id`` or ``stage/key``."""
    from .model import MandateError

    orders = orders or expand(c)
    stage, _, name = ref.partition("/")
    for o in orders.get(stage, []):
        if name in (o.id, o.key_str):
            return o
    raise MandateError(f"no current order {ref!r} (see `mandate orders`; it may be superseded)")


def superseded(c: Campaign, orders: dict[str, list[Order]]) -> list[str]:
    """Ledger orders that the plan no longer asks for: their inputs, decisions or stage changed."""
    current = {(s, o.id) for s, os_ in orders.items() for o in os_ if o.id}
    out = []
    ledger = c.root / "ledger"
    for folder in sorted(ledger.glob("*/*")) if ledger.exists() else []:
        if folder.is_dir() and (folder.parent.name, folder.name) not in current:
            out.append(f"{folder.parent.name}/{folder.name}")
    return out


def unassigned(c: Campaign) -> list[dict]:
    """Units no stage selects. Nothing should fall through silently."""
    return [u for u in c.units if not any(matches(u, s.get("over")) for s in c.stages.values())]
