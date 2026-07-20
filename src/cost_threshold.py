COST_MISSED_WARNING = 50_000  # claim payout + safety risk, placeholder
COST_FALSE_ALARM = 200  # wasted underwriting attention, placeholder


def expected_cost(y_true, y_pred_proba, threshold: float) -> float:
    raise NotImplementedError


def select_threshold(y_true, y_pred_proba, thresholds=None) -> float:
    raise NotImplementedError
