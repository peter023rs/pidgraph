"""GraphStore seam, live side (ticket 09): the Neo4j loader executes
exactly the statements the offline script emitter writes, so schema
conformance stays asserted offline against the emitted Cypher and the
suite never requires a live Neo4j — a fake driver captures what would run.
Loading is local-only: any URI that is not a loopback endpoint is refused
before a connection is ever attempted."""

import sys

import pytest

from pidgraph import neo4j_store
from pidgraph.cypher_store import to_cypher, to_cypher_statements
from pidgraph.neo4j_store import Neo4jGraphStore, Neo4jSettings
from pidgraph.pipeline import digitize
from pidgraph.seams import PipelineConfig


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    """An operator's real PIDGRAPH_NEO4J_* shell config must never steer
    the suite — registry-built stores read the environment."""
    for var in ("PIDGRAPH_NEO4J_URI", "PIDGRAPH_NEO4J_USER",
                "PIDGRAPH_NEO4J_PASSWORD", "PIDGRAPH_NEO4J_DATABASE"):
        monkeypatch.delenv(var, raising=False)


def _graph():
    return {
        "nodes": [
            {"tag": "T-101", "name": "T-101", "equipment_type": "tank",
             "detection_confidence": 1.0, "attributes": {}},
            {"tag": "V-101", "name": "V-101", "equipment_type": "valve",
             "detection_confidence": 1.0, "attributes": {}},
        ],
        "edges": [
            {"source": "T-101", "target": "V-101", "line_tag": None,
             "attributes": {"direction": "known",
                            "direction_sources": ["arrow:p1n5"]}},
        ],
    }


class FakeSession:
    def __init__(self, statements):
        self.statements = statements

    def run(self, statement):
        self.statements.append(statement)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    """Stands in for neo4j.Driver: records every statement run and which
    database the session targeted."""

    def __init__(self):
        self.statements = []
        self.databases = []
        self.closed = False

    def session(self, database=None):
        self.databases.append(database)
        return FakeSession(self.statements)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_loader_executes_exactly_the_offline_script_statements():
    driver = FakeDriver()
    store = Neo4jGraphStore(settings=Neo4jSettings(),
                            open_driver=lambda settings: driver)

    summary = store.store(_graph(), None)

    assert driver.statements == to_cypher_statements(_graph())
    # and the statement list IS the offline script, minus comments — the
    # offline schema-conformance tests therefore cover live loads
    assert [s + ";" for s in driver.statements] == [
        line for line in to_cypher(_graph()).splitlines()
        if not line.startswith("//")]
    assert driver.closed
    assert summary["kind"] == "neo4j"
    assert summary["nodes"] == 2
    assert summary["relationships"] == 1


def test_loader_targets_the_configured_database():
    driver = FakeDriver()
    settings = Neo4jSettings(database="plant")
    store = Neo4jGraphStore(settings=settings,
                            open_driver=lambda s: driver)

    store.store(_graph(), None)

    assert driver.databases == ["plant"]


def test_live_run_still_writes_the_inspectable_script(tmp_path):
    driver = FakeDriver()
    store = Neo4jGraphStore(settings=Neo4jSettings(),
                            open_driver=lambda s: driver)

    summary = store.store(_graph(), tmp_path)

    path = tmp_path / "plant_graph.cypher"
    assert path.exists()
    assert summary["path"] == str(path)
    assert summary["script"] == to_cypher(_graph())
    assert path.read_text(encoding="utf-8") == summary["script"]


@pytest.mark.parametrize("uri", [
    "bolt://db.example.com:7687",
    "neo4j://10.0.0.5:7687",
    "neo4j+s://abc123.databases.neo4j.io",
    "bolt+s://localhost:7687",
    "http://localhost:7474",
])
def test_remote_or_non_bolt_endpoints_are_refused(uri):
    with pytest.raises(ValueError, match="local"):
        Neo4jGraphStore(settings=Neo4jSettings(uri=uri))


@pytest.mark.parametrize("uri", [
    "bolt://localhost:7687",
    "neo4j://localhost:7687",
    "bolt://127.0.0.1:7687",
    "bolt://[::1]:7687",
])
def test_loopback_endpoints_are_accepted(uri):
    Neo4jGraphStore(settings=Neo4jSettings(uri=uri))  # must not raise


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("PIDGRAPH_NEO4J_URI", "neo4j://127.0.0.1:9999")
    monkeypatch.setenv("PIDGRAPH_NEO4J_USER", "operator")
    monkeypatch.setenv("PIDGRAPH_NEO4J_PASSWORD", "s3cret")
    monkeypatch.setenv("PIDGRAPH_NEO4J_DATABASE", "plant")

    settings = Neo4jSettings.from_env()

    assert settings == Neo4jSettings(uri="neo4j://127.0.0.1:9999",
                                     user="operator", password="s3cret",
                                     database="plant")


def test_environment_defaults_to_local_bolt():
    # the autouse fixture has cleared PIDGRAPH_NEO4J_*
    assert Neo4jSettings.from_env() == Neo4jSettings()
    assert Neo4jSettings().uri == "bolt://localhost:7687"


def test_missing_driver_package_names_the_install_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "neo4j", None)  # forces ImportError
    store = Neo4jGraphStore(settings=Neo4jSettings())

    with pytest.raises(RuntimeError, match=r"pidgraph\[neo4j\]"):
        store.store(_graph(), None)


def test_selected_by_configuration_behind_the_seam(
        synthetic_document, synthetic_profile, monkeypatch, tmp_path):
    """digitize() with graph_store="neo4j" loads into the (fake) driver
    without the pipeline changing; the script emitter stays the default."""
    assert PipelineConfig().graph_store == "cypher-script"

    driver = FakeDriver()
    monkeypatch.setattr(neo4j_store, "open_driver", lambda s: driver)

    artifacts = digitize(synthetic_document, synthetic_profile,
                         config=PipelineConfig(graph_store="neo4j"),
                         out_dir=tmp_path)

    assert artifacts.store_summary["kind"] == "neo4j"
    assert driver.statements[0].startswith("CREATE CONSTRAINT")
    assert driver.statements == to_cypher_statements(artifacts.plant_graph)
    # run outputs unchanged: script artifact still on disk, still exposed
    assert artifacts.paths["plant_graph"].endswith("plant_graph.cypher")
    assert artifacts.cypher_script == to_cypher(artifacts.plant_graph)
