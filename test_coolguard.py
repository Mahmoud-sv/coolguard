"""
CoolGuard - automated tests for the core decision logic.

Run:  pip install pytest
      pytest test_coolguard.py -v

Why this matters for your presentation: it proves the core calculations do
what you claim, on demand, not just "trust me." Being able to run this live
in front of a judge is a strong, simple way to show rigor.
"""

import sys
sys.path.insert(0, ".")
from coolguard_core import estimate_wbgt, wbgt_risk, hr_risk, symptom_score, rest_minutes, decide


def test_wbgt_increases_with_temperature():
    cooler = estimate_wbgt(temp_c=25, humidity_pct=40)
    hotter = estimate_wbgt(temp_c=40, humidity_pct=40)
    assert hotter > cooler


def test_wbgt_increases_with_humidity():
    drier = estimate_wbgt(temp_c=35, humidity_pct=20)
    humid = estimate_wbgt(temp_c=35, humidity_pct=80)
    assert humid > drier


def test_wbgt_risk_bands_are_ordered():
    assert wbgt_risk(25)[0] < wbgt_risk(29)[0] < wbgt_risk(31)[0] < wbgt_risk(35)[0]


def test_hr_risk_flags_large_rise():
    level, label = hr_risk(resting_hr=70, current_hr=115, age=30)
    assert level == 2
    assert label == "High strain"


def test_hr_risk_normal_when_close_to_resting():
    level, label = hr_risk(resting_hr=70, current_hr=75, age=30)
    assert level == 0


def test_symptom_score_none_is_zero():
    assert symptom_score(["None"]) == 0
    assert symptom_score([]) == 0


def test_symptom_score_sums_weights():
    score = symptom_score(["Headache", "Dizziness"])
    assert score == 2 + 3


def test_danger_symptom_scores_high():
    assert symptom_score(["Confusion"]) >= 5


def test_rest_minutes_zero_when_continuing():
    assert rest_minutes("🟢 CONTINUE", score=0) == 0


def test_rest_minutes_positive_when_stopping():
    assert rest_minutes("🔴 STOP", score=6) > 0


def test_decide_continue_in_mild_conditions():
    decision, score, rest = decide(wbgt=26, resting_hr=70, current_hr=75, age=25, symptoms=["None"])
    assert decision == "🟢 CONTINUE"
    assert rest == 0


def test_decide_stop_in_extreme_conditions_with_symptoms():
    decision, score, rest = decide(wbgt=34, resting_hr=70, current_hr=130, age=25,
                                   symptoms=["Dizziness", "Nausea"])
    assert decision == "🔴 STOP"
    assert rest > 0


def test_decide_stricter_for_new_worker():
    args = dict(wbgt=29, resting_hr=70, current_hr=95, age=25, symptoms=["None"])
    normal = decide(**args, new_worker=False)
    new_hire = decide(**args, new_worker=True)
    assert new_hire[1] >= normal[1]  # new worker's risk score should be >= the same reading for an experienced worker


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
