from swarm_optimizer import gold_version as gv


def test_current_points_to_sibling_repo(monkeypatch):
    monkeypatch.delenv("GOLD_VERSION", raising=False)
    p = gv.gold_paths()
    assert p["articles"].as_posix().endswith("gold_standard_v5/data/pilot_gold_articles.parquet")
    assert p["relations"].as_posix().endswith("gold_standard_v5/data/pilot_gold_final.parquet")
    assert p["unions_dir"].as_posix().endswith("gold_standard_v5/data/pilot_entity_unions")
    assert p["splits"].as_posix().endswith("results/swarm/splits.json")


def test_v3_points_local(monkeypatch):
    monkeypatch.setenv("GOLD_VERSION", "v3")
    p = gv.gold_paths()
    assert p["articles"].as_posix().endswith("data/gold_v3/articles.parquet")
    assert p["relations"].as_posix().endswith("data/gold_v3/gold_final.parquet")
    assert p["unions_dir"].as_posix().endswith("data/gold_v3/entity_unions")
    assert p["splits"].as_posix().endswith("data/gold_v3/splits.json")


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("GOLD_VERSION", "v3")
    p = gv.gold_paths("current")
    assert p["articles"].as_posix().endswith("gold_standard_v5/data/pilot_gold_articles.parquet")


def test_current_matches_legacy_hardcoded_paths(monkeypatch):
    """Con GOLD_VERSION ausente, las rutas deben ser EXACTAMENTE las viejas (no romper)."""
    monkeypatch.delenv("GOLD_VERSION", raising=False)
    from swarm_optimizer import splits, rubric
    p = gv.gold_paths()
    assert splits.gold_articles_path() == p["articles"]
    assert splits.splits_path() == p["splits"]
    assert rubric.unions_dir() == p["unions_dir"]
    assert rubric.gold_relations_path() == p["relations"]
