from text2sg.genome import Genome, AnalysisConfig


def test_genome_roundtrip_with_analysis():
    g = Genome(prompt_text="x", analysis=AnalysisConfig(emit_dossier=False, role_window=25))
    g2 = Genome.from_json(g.to_json())
    assert g2.analysis is not None
    assert g2.analysis.emit_dossier is False
    assert g2.analysis.role_window == 25


def test_genome_roundtrip_without_analysis():
    g = Genome(prompt_text="x")
    g2 = Genome.from_json(g.to_json())
    assert g2.analysis is None


from text2sg.analysis import (
    canon_act_type, _actor_dossier, _alias_map, _act_type_canon_block,
)


def _union():
    return {
        "U1": {"type": "roster_actor", "canonical_names": ["Luis Hermosilla"],
               "surfaces": ["Hermosilla", "Luis Hermosilla"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Juan Pablo Hermosilla"],
               "surfaces": ["Juan Pablo Hermosilla"]},
        "U9": {"type": "NIL", "canonical_names": ["Ruido"], "surfaces": ["ruido"]},
    }


def test_canon_act_type_maps_noncanonical():
    assert canon_act_type("kill") == "attacks"
    assert canon_act_type("CRITICIZES") == "accuses"
    assert canon_act_type("attacks") == "attacks"


def test_actor_dossier_excludes_nil_and_lists_aliases():
    out = _actor_dossier(_union())
    assert "Luis Hermosilla" in out
    assert "Juan Pablo Hermosilla" in out
    assert "Ruido" not in out


def test_alias_map_maps_surface_to_canonical():
    out = _alias_map(_union())
    assert "Hermosilla" in out and "Luis Hermosilla" in out


def test_act_type_canon_block_lists_canonical_types():
    out = _act_type_canon_block()
    assert "attacks" in out and "endorses" in out


from text2sg.analysis import _role_hints


def test_role_hints_disambiguates_shared_surname():
    union = _union()
    body = ("Juan Pablo Hermosilla, abogado defensor, intervino en la audiencia. "
            "Relleno neutral de la nota para separar bien las dos menciones del texto. "
            "El imputado Luis Hermosilla guardó silencio ante el tribunal.")
    out = _role_hints(union, body, None, 25)
    assert "AMBIGÜEDAD" in out
    assert "abogado/defensa" in out
    assert "imputado" in out


def test_role_hints_no_role_when_absent():
    union = {"U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"],
                    "surfaces": ["Boric"]}}
    out = _role_hints(union, "Gabriel Boric habló en La Moneda.", None, 80)
    assert "rol no detectado" in out


from text2sg.analysis import (
    _direction_scaffold, _main_speaker, _comention_pairs, _domain_gate,
)


def test_direction_scaffold_passive_voice():
    out = _direction_scaffold("Boric fue criticado por Matthei en la sesión.")
    assert "from=Matthei" in out and "to=Boric" in out


def test_direction_scaffold_no_passive_returns_empty():
    assert _direction_scaffold("Boric habló en La Moneda.") == ""


def test_main_speaker_picks_most_mentioned():
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Evelyn Matthei"], "surfaces": ["Matthei"]},
    }
    body = "Boric anunció. Boric defendió. Boric insistió. Matthei respondió."
    out = _main_speaker(union, body)
    assert "Gabriel Boric" in out


def test_comention_pairs_same_sentence():
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Evelyn Matthei"], "surfaces": ["Matthei"]},
    }
    out = _comention_pairs(union, "Boric y Matthei coincidieron en el acto.")
    assert "Gabriel Boric" in out and "Evelyn Matthei" in out


def test_domain_gate_flags_football():
    out = _domain_gate({}, "El club ganó el partido con un gol en el estadio.")
    assert out != "" and "deportivo" in out.lower()


from text2sg.analysis import build_analysis


def test_build_analysis_empty_union_returns_empty():
    assert build_analysis({}, "texto", AnalysisConfig()) == ""


def test_build_analysis_respects_gates():
    cfg = AnalysisConfig(
        emit_dossier=True, emit_alias_map=False, emit_role_hints=False,
        emit_direction_scaffold=False, emit_main_speaker=False,
        emit_comention_pairs=False, emit_act_type_canon=False, emit_domain_gate=False,
    )
    out = build_analysis(_union(), "Luis Hermosilla habló.", cfg)
    assert "ACTORES" in out
    assert "MAPA DE ALIAS" not in out
    assert "=== ANÁLISIS DE ACTORES ===" in out


def test_build_analysis_full_has_all_sections():
    out = build_analysis(_union(), "Juan Pablo Hermosilla, abogado, habló.", AnalysisConfig())
    assert "ACTORES" in out
    assert "MAPA DE ALIAS" in out
    assert "ROLES DETECTADOS" in out
    assert "ACT_TYPES CANÓNICOS" in out


from text2sg.extractor import build_prompt


def test_build_prompt_uses_analysis_block_when_present():
    g = Genome(prompt_text="INSTRUCCIONES", architecture="given_entities",
               analysis=AnalysisConfig())
    prompt = build_prompt(g, "Luis Hermosilla habló en la audiencia.", _union(), [])
    assert "=== ANÁLISIS DE ACTORES ===" in prompt
    assert "INSTRUCCIONES" in prompt


def test_build_prompt_falls_back_to_flat_list_without_analysis():
    g = Genome(prompt_text="INSTRUCCIONES", architecture="given_entities", analysis=None)
    prompt = build_prompt(g, "Luis Hermosilla habló.", _union(), [])
    assert "=== ANÁLISIS DE ACTORES ===" not in prompt
    assert "ACTORES PRESENTES EN EL ARTÍCULO:" in prompt


# ── regresiones del code-review (bugs C1/C2/I1/I2/I3) ──────────────────────── #


def test_main_speaker_no_double_count_nested_surface():
    # C1: el apellido dentro del nombre completo cuenta UNA vez. Boric (4 menciones
    # reales) debe ganar a Hermosilla (3), no perder por doble-conteo de alias anidado.
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Luis Hermosilla"],
               "surfaces": ["Hermosilla", "Luis Hermosilla"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Gabriel Boric"], "surfaces": ["Boric"]},
    }
    body = ("Luis Hermosilla habló. Luis Hermosilla insistió. Luis Hermosilla calló. "
            "Boric A. Boric B. Boric C. Boric D.")
    out = _main_speaker(union, body)
    assert "Gabriel Boric" in out
    assert "Luis Hermosilla" not in out.split(":")[1]   # no es el hablante principal


def test_role_hints_no_leak_across_sentences():
    # C2: 'imputado' (oración 1) no debe filtrarse a Juan Pablo (oración 2); y Luis,
    # cuyo único alias es el apellido pelado, SÍ debe recibir 'imputado'.
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Luis Hermosilla"], "surfaces": ["Hermosilla"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Juan Pablo Hermosilla"],
               "surfaces": ["Juan Pablo Hermosilla"]},
    }
    body = "El imputado Hermosilla guardó silencio. Juan Pablo Hermosilla, abogado defensor, intervino."
    out = _role_hints(union, body, None)
    luis_line = [l for l in out.splitlines() if l.strip().startswith("Luis Hermosilla")][0]
    jp_line = [l for l in out.splitlines() if l.strip().startswith("Juan Pablo Hermosilla")][0]
    assert "imputado" in luis_line and "abogado/defensa" not in luis_line
    assert "abogado/defensa" in jp_line and "imputado" not in jp_line


def test_alias_map_flags_ambiguous_shared_surface():
    # I1: un alias reclamado por dos actores es AMBIGUO, no se mapea a uno solo.
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Luis Hermosilla"], "surfaces": ["Hermosilla"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Juan Pablo Hermosilla"], "surfaces": ["Hermosilla"]},
    }
    out = _alias_map(union)
    assert "AMBIGUO" in out
    assert "Luis Hermosilla" in out and "Juan Pablo Hermosilla" in out


def test_comention_pairs_no_self_pair_from_substring():
    # I2: dos uids con el mismo canónico (o substring) no producen un par consigo mismo.
    union = {
        "U1": {"type": "roster_actor", "canonical_names": ["Gabriel Boric Font"], "surfaces": ["Boric"]},
        "U2": {"type": "roster_actor", "canonical_names": ["Gabriel Boric Font"],
               "surfaces": ["Gabriel Boric Font"]},
    }
    out = _comention_pairs(union, "Gabriel Boric Font firmó el decreto.")
    assert "↔" not in out   # sin par fantasma


def test_direction_scaffold_adverb_passive():
    # I3: pasiva con adverbio intercalado igual se detecta.
    out = _direction_scaffold("Boric fue duramente criticado por Matthei.")
    assert "from=Matthei" in out and "to=Boric" in out
