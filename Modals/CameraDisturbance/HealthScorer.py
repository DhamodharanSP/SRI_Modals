class HealthScorer:
    def __init__(self, weights=None):
        self.weights = weights or {
            'brightness': 0.20,
            'blur': 0.25,
            'exposure': 0.20,
            'obstruction': 0.25,
            'network': 0.10
        }

    def score(self, metrics):
        total = 0.0
        for k, w in self.weights.items():
            total += w * float(metrics.get(k, 100.0))
        return float(max(0.0, min(100.0, total)))
