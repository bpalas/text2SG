from swarm_optimizer.fitness import f_beta, fitness, RECALL_FLOOR


def test_f_beta_weights_precision_over_recall():
    # F0.5: con precisión alta y recall bajo, F0.5 > F1
    p, r = 0.8, 0.2
    f05 = f_beta(p, r, beta=0.5)
    f1 = f_beta(p, r, beta=1.0)
    assert f05 > f1


def test_f_beta_zero_when_both_zero():
    assert f_beta(0.0, 0.0, 0.5) == 0.0


def _metrics(prec_rel, rec_rel, prec_ent=0.8, rec_ent=0.8, pol=0.8, act=0.7):
    return {
        "Precision_rel": prec_rel, "Recall_rel": rec_rel,
        "Precision_ent": prec_ent, "Recall_ent": rec_ent,
        "Polarity_acc": pol, "Act_acc": act,
    }


def test_recall_floor_penalizes_graded():
    above = fitness(_metrics(0.95, RECALL_FLOOR + 0.10), tokens_per_article=1000)
    just_below = fitness(_metrics(0.95, RECALL_FLOOR - 0.01), tokens_per_article=1000)
    far_below = fitness(_metrics(0.95, 0.0), tokens_per_article=1000)
    # monótona: cuanto menor el recall bajo el piso, peor (no se cancela en un -1.0 plano)
    assert above > just_below > far_below
    assert far_below < 0      # recall ~0 sí queda fuertemente penalizado


def test_cost_penalty_prefers_cheaper():
    cheap = fitness(_metrics(0.7, 0.5), tokens_per_article=500)
    pricey = fitness(_metrics(0.7, 0.5), tokens_per_article=8000)
    assert cheap > pricey


def test_higher_precision_scores_higher():
    lo = fitness(_metrics(0.4, 0.5), tokens_per_article=1000)
    hi = fitness(_metrics(0.9, 0.5), tokens_per_article=1000)
    assert hi > lo


def test_pro_penalized_more_than_flash_same_tokens():
    flash = fitness(_metrics(0.7, 0.5), tokens_per_article=3000, model="gemini-2.5-flash")
    pro = fitness(_metrics(0.7, 0.5), tokens_per_article=3000, model="gemini-2.5-pro")
    assert flash > pro      # mismo conteo de tokens, pero Pro cuesta más -> menor fitness
