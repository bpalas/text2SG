from swarm_optimizer.subsets import registro_of, split_ids_by_registro


def test_formal_medios():
    for m in ("Emol", "La Tercera", "EFE", "Reuters", "T13", "El Mostrador"):
        assert registro_of(m) == "formal", m


def test_informal_medios():
    for m in ("La Cuarta", "The Clinic"):
        assert registro_of(m) == "informal", m


def test_biobio_mojibake_is_informal():
    # el parquet trae el carácter de reemplazo U+FFFD en lugar de í; "radio b" lo cubre igual
    assert registro_of("Radio B�o-B�o") == "informal"
    assert registro_of("Radio Bío-Bío") == "informal"


def test_empty_or_unknown_defaults_formal():
    assert registro_of("") == "formal"
    assert registro_of(None) == "formal"
    assert registro_of("Diario Desconocido") == "formal"


def test_split_ids_by_registro_partitions():
    medio_map = {"a": "Emol", "b": "La Cuarta", "c": "EFE", "d": "The Clinic"}
    groups = split_ids_by_registro(["a", "b", "c", "d"], medio_map)
    assert groups["formal"] == ["a", "c"]
    assert groups["informal"] == ["b", "d"]
