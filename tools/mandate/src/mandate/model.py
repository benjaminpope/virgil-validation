"""The campaign on disk: the plan, the registry, the docket and git.

A campaign is one directory:

    mandate.yaml        the plan: stages, decisions, who may change what
    registry/units.csv  one row per unit of data (a snapshot of the group's sheet)
    notes/*.md          human notes and paper extracts, each marked open or sealed
    docket.jsonl        every departure from the plan's defaults, append-only
    ledger/             one small record per work order and per attempt
    STATUS.md           generated

Nothing here runs anything; see ``orders`` (what work exists) and ``work`` (doing it).
"""

from __future__ import annotations

import csv
import datetime
import difflib
import hashlib
import json
import os
import pathlib
import re
import subprocess

import yaml

PLAN = "mandate.yaml"
UNITS = "registry/units.csv"
LINKS = "registry/links.csv"
DOCKET = "docket.jsonl"
# Paths that define the mandate: a run refuses unless these are committed and clean.
MANDATE_PATHS = (PLAN, "registry")
KINDS = ("decide", "exclude", "annotate", "request", "resolve")
STAGE_KEYS = {
    "doc", "over", "by", "needs", "run", "worker", "prompt", "playbook", "decisions", "result", "status_key",
    "requires", "expects", "promote", "review", "sealed", "informed", "uses", "slurm", "digest",
}


class MandateError(Exception):
    """A refusal: the request is outside the plan, or the campaign is not in a state to run."""


def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(text)).strip("_") or "_"


def option_names(decision: dict) -> list[str]:
    """A decision's allowed values. ``options`` is a list, or a map of name to the text it puts
    in the command (``{paper_nights: "", all_nights: "--include-extra"}``)."""
    return [str(o) for o in decision.get("options", [decision.get("default")])]


def option_text(decision: dict, value) -> str:
    options = decision.get("options")
    return str(options[value]) if isinstance(options, dict) and value in options else str(value)


def _rank(level) -> int:
    return {"open": 0, "informed": 1, "sealed": 2}.get(level, 2 if level is True else 0)


def matches(row: dict, where: dict) -> bool:
    """A selector is ``{column: value}`` or ``{column: [values]}``; every column must match."""
    for col, want in (where or {}).items():
        have = str(row.get(col, "") or "")
        wants = want if isinstance(want, list) else [want]
        if have not in [str("" if w is None else w) for w in wants]:
            return False
    return True


def git(root: pathlib.Path, *args: str) -> str | None:
    """Output of a git command in ``root``, or None when it fails (no repository, unknown ref)."""
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


class Campaign:
    def __init__(self, root: str | os.PathLike = "."):
        self.root = pathlib.Path(root).resolve()
        if not (self.root / PLAN).exists():
            raise MandateError(f"no {PLAN} in {self.root}")
        self.plan = yaml.safe_load((self.root / PLAN).read_text()) or {}
        self.stages: dict[str, dict] = self.plan.get("stages") or {}
        self.unit_id: str = self.plan.get("unit_id", "id")
        self.sealed_columns = set(self.plan.get("sealed_columns") or [])
        self.informed_columns = set(self.plan.get("informed_columns") or [])
        self.humans = set(self.plan.get("humans") or [])
        runs = os.environ.get("MANDATE_RUNS") or self.plan.get("runs_root", "runs")
        self.runs_root = (self.root / runs).resolve()
        self.ignore_columns = set(self.plan.get("ignore_columns") or [])
        self.docket = self._read_docket()
        self._index()
        self.units = self._read_units()
        self.links = self._read_links()

    # ---- registry -------------------------------------------------------------------------

    def _read_units(self) -> list[dict]:
        path = self.root / UNITS
        if not path.exists():
            return []
        with open(path, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        for entry in self.active("annotate"):
            for row in rows:
                if matches(row, entry["where"]):
                    row.update({k: str(v) for k, v in entry["set"].items()})
        return rows

    def _read_links(self) -> list[dict]:
        """Typed links between units: ``from,to,kind`` and where each came from. The registry's
        columns say what a unit is; links say which other units it belongs with."""
        path = self.root / LINKS
        self._linked: dict[tuple[str, str], set[str]] = {}
        if not path.exists():
            return []
        with open(path, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        for r in rows:
            self._linked.setdefault((r.get("kind", ""), r.get("from", "")), set()).add(r.get("to", ""))
            self._linked.setdefault((r.get("kind", ""), r.get("to", "")), set()).add(r.get("from", ""))
        return rows

    def linked(self, kind: str, unit_id: str) -> set[str]:
        return self._linked.get((kind, unit_id), set())

    def level(self, stage: str) -> str:
        """What a stage may know: ``open`` (blind), ``informed`` or ``sealed``.

        Three levels of knowledge about the answer. *Open* is what a blind analysis may use: the data and facts
        about the data. *Informed* adds what a publication says the object is (a model family, a source class),
        so an informed stage's results are never blind results. *Sealed* adds the published values themselves and
        is for scoring only.
        """
        st = self.stages.get(stage) or {}
        return "sealed" if st.get("sealed") else ("informed" if st.get("informed") else "open")

    def open_row(self, row: dict, sealed: bool | str = False) -> dict:
        """A registry row as a stage of the given level may see it (``sealed`` is a level, or True for sealed)."""
        rank = _rank(sealed)
        hidden = (self.sealed_columns if rank < 2 else set()) | (self.informed_columns if rank < 1 else set())
        return {k: v for k, v in row.items() if k not in hidden}

    # ---- docket ---------------------------------------------------------------------------

    def _read_docket(self) -> list[dict]:
        path = self.root / DOCKET
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def needs_human(self, entry: dict) -> bool:
        """Does this entry take effect only once a person approves it?

        An entry made by someone the plan lists under ``humans`` is its own approval. ``by`` is
        declared, not proven: ``verify --base`` lists every such entry for the person who merges.
        """
        kind = entry["kind"]
        if kind == "request":
            return True
        if entry.get("by") in self.humans:
            return False
        if kind == "decide":
            menu = self.stages.get(entry.get("stage"), {}).get("decisions") or {}
            return any(menu.get(k, {}).get("authority", "human") != "agent" for k in entry["set"])
        if kind == "annotate":
            return entry.get("review", "human") != "none"
        if kind == "exclude":
            return self.plan.get("exclusions", "human") != "agent"
        return False

    def _index(self) -> None:
        self._resolved = {e["of"]: e for e in self.docket if e["kind"] == "resolve"}
        self._active: dict[str, list[dict]] = {}

    def resolution(self, entry: dict) -> dict | None:
        return self._resolved.get(entry["id"])

    def state(self, entry: dict) -> str:
        """``active`` (in force), ``proposed``/``open`` (awaiting a person), ``denied`` or ``closed``."""
        res = self.resolution(entry)
        if res:
            if entry["kind"] == "request":
                return "closed"
            return "active" if res["verdict"] == "approve" else "denied"
        if entry["kind"] == "request":
            return "open"
        return "proposed" if self.needs_human(entry) else "active"

    def active(self, kind: str) -> list[dict]:
        if kind not in self._active:
            self._active[kind] = [e for e in self.docket if e["kind"] == kind and self.state(e) == "active"]
        return self._active[kind]

    def pending(self) -> list[dict]:
        """Entries waiting for a person."""
        return [e for e in self.docket if e["kind"] != "resolve" and self.state(e) in ("proposed", "open")]

    def append(self, kind: str, why: str, by: str | None = None, **fields) -> dict:
        if kind not in KINDS:
            raise MandateError(f"unknown docket kind {kind!r}")
        if not (why or "").strip():
            raise MandateError("a docket entry needs a reason (--why)")
        if kind == "decide":
            self.check_decision(fields["stage"], fields["set"])
        if kind in ("decide", "request") and fields.get("stage") not in self.stages:
            raise MandateError(f"unknown stage {fields.get('stage')!r}")
        if kind == "resolve":
            target = next((e for e in self.docket if e["id"] == fields["of"] and e["kind"] != "resolve"), None)
            if target is None:
                raise MandateError(f"no docket entry {fields['of']!r}")
            if self.resolution(target):
                raise MandateError(f"{fields['of']} is already resolved")
            own = fields["verdict"] == "deny" and (by or actor()) == target.get("by")  # withdrawing one's own
            if self.humans and (by or actor()) not in self.humans and not own:
                raise MandateError(f"only the plan's humans ({', '.join(sorted(self.humans))}) resolve docket entries")
        entry = {"kind": kind, **fields, "why": why.strip(), "by": by or actor(), "at": now()}
        # A content hash, not a counter: two branches can append without clashing.
        entry = {"id": "d-" + sha(canonical(entry) + str(len(self.docket)), 6), **entry}
        with open(self.root / DOCKET, "a") as f:
            f.write(canonical(entry) + "\n")
        self.docket.append(entry)
        self._index()
        self.units = self._read_units()
        return entry

    def check_decision(self, stage: str, chosen: dict) -> None:
        """Refuse anything that is not on the stage's menu: that is a request, not a decision."""
        menu = self.stages.get(stage, {}).get("decisions") or {}
        for name, value in chosen.items():
            if name not in menu:
                raise MandateError(f"stage {stage!r} has no decision {name!r}; file a request instead")
            options = option_names(menu[name])
            if str(value) not in options:
                raise MandateError(f"{name}={value!r} is not an option ({', '.join(options)}); file a request")

    def sealed_root(self) -> pathlib.Path | None:
        """Where the answers are kept, when the plan keeps them outside the campaign."""
        return (self.root / self.plan["sealed_root"]).resolve() if self.plan.get("sealed_root") else None

    # ---- git ------------------------------------------------------------------------------

    def plan_commit(self) -> str:
        """The commit that last changed the plan or the registry. Refuses when either is uncommitted."""
        if git(self.root, "rev-parse", "--git-dir") is None:
            raise MandateError("the campaign is not in a git repository: commit the plan and registry first")
        dirty = git(self.root, "status", "--porcelain", "--", *MANDATE_PATHS)
        if dirty:
            raise MandateError("the plan or registry has uncommitted changes; commit them first:\n" + dirty)
        commit = git(self.root, "log", "-1", "--format=%H", "--", *MANDATE_PATHS)
        if not commit:
            raise MandateError("the plan and registry have never been committed")
        return commit

    def campaign_state(self) -> dict:
        """HEAD and whether anything but the ledger, the docket and the status page is uncommitted."""
        skip = [":!ledger", ":!STATUS.md", f":!{DOCKET}"]
        if self.runs_root.is_relative_to(self.root) and self.runs_root != self.root:
            skip.append(f":!{self.runs_root.relative_to(self.root).as_posix()}")
        status = git(self.root, "status", "--porcelain", "--", ".", *skip)
        return {"commit": git(self.root, "rev-parse", "HEAD"), "dirty": bool(status)}

    def versions(self, names: list[str] | None = None) -> dict:
        """Evaluate the plan's ``versions``; refuse a pinned one that does not match."""
        out = {}
        for name, spec in (self.plan.get("versions") or {}).items():
            if names is not None and name not in names:
                continue
            if "git" in spec:
                repo = (self.root / spec["git"]).resolve()
                value = git(repo, "rev-parse", "HEAD")
                dirty = bool(git(repo, "status", "--porcelain"))
                out[name] = {"value": value, "dirty": dirty}
            else:
                done = subprocess.run(spec["cmd"], shell=True, capture_output=True, text=True, cwd=self.root, check=False)
                value = done.stdout.strip() if done.returncode == 0 else None
                out[name] = {"value": value}
            pin = spec.get("pin")
            if pin is not None and (value is None or not str(value).startswith(str(pin))):
                raise MandateError(f"version {name!r} is {value!r} but the plan pins {pin!r}")
            if pin is not None and out[name].get("dirty"):
                raise MandateError(f"version {name!r} is pinned but its checkout has uncommitted changes")
        return out

    # ---- validation -----------------------------------------------------------------------

    def problems(self) -> list[str]:
        """Everything wrong with the plan, registry and docket that can be told without running."""
        out = []
        ids = [u.get(self.unit_id, "") for u in self.units]
        columns = set(self.units[0]) if self.units else set()
        for e in self.active("annotate"):
            columns |= set(e["set"])
        if self.units and self.unit_id not in self.units[0]:
            out.append(f"registry has no {self.unit_id!r} column (unit_id)")
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
            out.append(f"unit ids are not unique: {', '.join(dup)}")
        promoted = {c for s in self.stages.values() for c in (s.get("promote") or {}).values()}
        if not self.humans:
            out.append("the plan lists no humans: nobody can approve a proposal or close a request")
        known = set(self.plan.get("versions") or {})
        def hint(word: str, known: set) -> str:
            close = difflib.get_close_matches(str(word), sorted(known), n=1)
            return f" (did you mean {close[0]!r}?)" if close else ""

        for name, st in self.stages.items():
            for key in set(st) - STAGE_KEYS:
                out.append(f"stage {name}: unknown key {key!r}" + hint(key, STAGE_KEYS))
            worker = st.get("worker", "code")
            if worker not in ("code", "agent"):
                out.append(f"stage {name}: worker must be code or agent")
            if worker == "code" and not st.get("run"):
                out.append(f"stage {name}: a code stage needs a run command")
            if worker == "agent" and not st.get("prompt"):
                out.append(f"stage {name}: an agent stage needs a prompt file")
            for ref in ("prompt", "playbook"):
                if st.get(ref) and not (self.root / st[ref]).exists():
                    out.append(f"stage {name}: {ref} file {st[ref]} is missing")
            for col in list(st.get("over") or {}) + list(st.get("by") or []):
                if columns and col not in columns | promoted:
                    out.append(f"stage {name}: column {col!r} is not in the registry" + hint(col, columns | promoted))
            for col in (st.get("over") or {}):
                if col in self.sealed_columns and not st.get("sealed"):
                    out.append(f"stage {name}: selects on sealed column {col!r} but is not a sealed stage")
                if col in self.informed_columns and self.level(name) == "open":
                    out.append(f"stage {name}: selects on informed column {col!r} but is a blind stage")
            for label, need in (st.get("needs") or {}).items():
                up = (need or {}).get("stage", label)
                if up not in self.stages:
                    out.append(f"stage {name}: needs unknown stage {up!r}" + hint(up, set(self.stages)))
                elif self.stages[up].get("sealed") and not st.get("sealed"):
                    out.append(f"stage {name}: reads sealed stage {up!r} but is not sealed itself")
                elif _rank(self.level(up)) > _rank(self.level(name)):
                    out.append(f"stage {name}: is blind but reads informed stage {up!r}; its results would not be blind")
                kinds = {r.get("kind") for r in self.links}
                if (need or {}).get("link") and need["link"] not in kinds:
                    out.append(f"stage {name}: no link of kind {need['link']!r} in registry/links.csv"
                               + hint(need["link"], kinds))
                for col in (need or {}).get("match") or []:
                    if columns and col not in columns | promoted:
                        out.append(f"stage {name}: match column {col!r} is not in the registry")
            for dname, d in (st.get("decisions") or {}).items():
                options = option_names(d)
                picks = [d.get("default")] + [r.get("use") for r in d.get("rules") or []]
                for p in picks:
                    if str(p) not in options:
                        out.append(f"stage {name}: decision {dname} default/rule {p!r} is not one of its options")
            for v in st.get("uses") or []:
                if v not in known:
                    out.append(f"stage {name}: uses unknown version {v!r}")
        for r in self.links:
            for end in ("from", "to"):
                if ids and r.get(end) not in ids:
                    out.append(f"links.csv: {r.get(end)!r} ({r.get('kind')}) is not a unit")
                    break
        try:
            self.order_of_stages()
        except MandateError as e:
            out.append(str(e))
        for e in self.docket:
            if e["kind"] == "decide" and e.get("stage") in self.stages:
                try:
                    self.check_decision(e["stage"], e["set"])
                except MandateError as err:
                    out.append(f"docket {e['id']}: {err}")
        return out

    def order_of_stages(self) -> list[str]:
        """Stages with every stage after the ones it needs."""
        done: list[str] = []
        left = dict(self.stages)
        while left:
            def met(s: dict) -> bool:
                ups = [(need or {}).get("stage", label) for label, need in (s.get("needs") or {}).items()]
                return all(u in done or u not in self.stages for u in ups)

            ready = [n for n, s in left.items() if met(s)]
            if not ready:
                raise MandateError(f"stages need each other in a cycle: {', '.join(left)}")
            for n in ready:
                done.append(n)
                del left[n]
        return done


def actor() -> str:
    """Who is acting. Declared, not proven: set MANDATE_ACTOR for an agent."""
    return os.environ.get("MANDATE_ACTOR") or os.environ.get("USER") or "unknown"


def read_notes(c: Campaign, units: list[dict], stage: str, sealed: bool | str) -> list[dict]:
    """Notes that apply to these units at this stage. A note's ``visibility`` (open, informed, sealed) must not
    exceed the stage's level (``sealed``: a level, or True for sealed)."""
    sealed, level = _rank(sealed) == 2, _rank(sealed)
    out = []
    paths = sorted((c.root / "notes").glob("*.md"))
    vault = c.sealed_root()
    if sealed and vault is not None:
        paths += sorted((vault / "notes").glob("*.md"))
    for path in paths:
        text = path.read_text()
        meta, body = {}, text
        if text.startswith("---"):
            _, head, body = text.split("---", 2)
            meta = yaml.safe_load(head) or {}
        if _rank(meta.get("visibility", "open")) > level:
            continue
        if meta.get("stages") and stage not in meta["stages"]:
            continue
        where = meta.get("applies_to") or {}
        if not where or any(matches(u, where) for u in units):
            out.append({"file": path.name, "meta": meta, "body": body.strip()})
    return out
