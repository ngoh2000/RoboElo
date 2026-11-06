"""RoboElo: joint rating of robot-policy competency and task difficulty."""

# Logit <-> Elo scale. P(success) = 1 / (1 + 10^(-(R_policy - R_task)/400)) = sigmoid(theta - beta)
ELO_BASE = 1500.0
ELO_PER_LOGIT = 400.0 / 2.302585092994046  # 400 / ln(10) ~= 173.72


def to_elo(x):
    return ELO_BASE + ELO_PER_LOGIT * x
