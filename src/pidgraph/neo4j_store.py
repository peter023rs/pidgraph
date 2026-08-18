"""Live Neo4j GraphStore (ticket 09) — loads a plant graph into an
operator-controlled local database, selected by configuration behind the
GraphStore seam: PipelineConfig(graph_store="neo4j"). The Cypher-script
emitter stays the default; a run needs no live database to be useful.

The loader executes exactly the statements the offline emitter writes
(cypher_store.to_cypher_statements), so a live load conforms to s2_pml
(ADR-0001) precisely when the emitted script does — schema conformance
stays asserted offline, and the test suite never needs a running Neo4j.
Those statements MERGE on the unique tag under an IF NOT EXISTS
constraint, so loading the same run twice is idempotent.

Local-only, per ADR-0001's corpus sensitivity: any URI that is not a
loopback bolt://neo4j:// endpoint is refused at construction time, before
a connection is ever attempted — nothing leaves the operator's machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .cypher_store import to_cypher, to_cypher_statements

# unencrypted local schemes only — the +s/+ssc TLS variants signal a hosted
# endpoint (AuraDB etc.), which local-only loading exists to rule out
_LOCAL_SCHEMES = {"bolt", "neo4j"}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Neo4jSettings:
    """Connection settings for the operator's local database, read from the
    PIDGRAPH_NEO4J_* environment — deployment details stay out of
    PipelineConfig, which only selects the implementation."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str | None = None  # None -> connect without auth
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> Neo4jSettings:
        return cls(
            uri=os.environ.get("PIDGRAPH_NEO4J_URI", cls.uri),
            user=os.environ.get("PIDGRAPH_NEO4J_USER", cls.user),
            password=os.environ.get("PIDGRAPH_NEO4J_PASSWORD"),
            database=os.environ.get("PIDGRAPH_NEO4J_DATABASE", cls.database),
        )


def _require_local(uri: str) -> None:
    parts = urlsplit(uri)
    if parts.scheme not in _LOCAL_SCHEMES or (
            parts.hostname not in _LOOPBACK_HOSTS):
        raise ValueError(
            f"Neo4j URI {uri!r} is not a local endpoint; loading is "
            "local-only (ADR-0001) — use bolt:// or neo4j:// with one of "
            f"the loopback hosts {sorted(_LOOPBACK_HOSTS)}")


def open_driver(settings: Neo4jSettings) -> Any:
    """The official driver against the operator's local endpoint. Imported
    lazily so the default offline store never needs the neo4j package."""
    try:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "the live Neo4j loader needs the neo4j driver package; "
            "install it with: pip install 'pidgraph[neo4j]'") from exc
    auth = (settings.user, settings.password) \
        if settings.password is not None else None
    return GraphDatabase.driver(settings.uri, auth=auth)


class Neo4jGraphStore:
    """Live GraphStore: runs the offline emitter's statements against a
    local Neo4j, one auto-commit transaction each (CREATE CONSTRAINT cannot
    share a transaction with writes). Still writes the script artifact when
    an out_dir is given, so a live run stays operator-inspectable."""

    def __init__(self, settings: Neo4jSettings | None = None,
                 open_driver: Callable[[Neo4jSettings], Any] | None = None):
        self.settings = settings or Neo4jSettings.from_env()
        _require_local(self.settings.uri)
        self._open_driver = open_driver

    def store(self, graph: dict, out_dir: Path | None = None) -> dict:
        # local-only must hold at the moment the wire opens, not just at
        # construction
        _require_local(self.settings.uri)
        statements = to_cypher_statements(graph)
        script = to_cypher(graph)
        path = None
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "plant_graph.cypher"
            path.write_text(script, encoding="utf-8")
        # the open_driver global resolves at call time, so tests patching
        # pidgraph.neo4j_store.open_driver reach registry-built stores too
        opener = self._open_driver or open_driver
        with opener(self.settings) as driver:
            with driver.session(database=self.settings.database) as session:
                for statement in statements:
                    session.run(statement)
        return {
            "kind": "neo4j",
            "uri": self.settings.uri,
            "database": self.settings.database,
            "script": script,
            "path": str(path) if path else None,
            "nodes": len(graph["nodes"]),
            "relationships": len(graph["edges"]),
            "statements": len(statements),
        }
