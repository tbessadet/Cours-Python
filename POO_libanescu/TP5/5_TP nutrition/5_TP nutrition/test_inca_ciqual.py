import inca_ciqual as ic

def test_recherche_aliment():
    r = ic.rechercher_aliment("pain")
    assert len(r) == 1
    assert r.iloc[0]["code_aliment"] == 1001

def test_composition_pain():
    r = ic.composition_aliment(1001)
    assert len(r) == 5
    assert float(r.loc[r["code_const"] == 2, "teneur"].iloc[0]) == 8.5

def test_apport_proteines_pain():
    r = ic.apport_aliment(1001, 60)
    valeur = float(r.loc[r["code_const"] == 2, "apport"].iloc[0])
    assert abs(valeur - 5.1) < 1e-9
