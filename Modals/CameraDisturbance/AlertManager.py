import time


class AlertManager:
    def __init__(self, light_seconds=5.0, warning_seconds=10.0):
        self.light_seconds = light_seconds
        self.warning_seconds = warning_seconds
        self.conditions = {}

    def update_condition(self, name, active):
        now = time.time()

        if active:
            if name not in self.conditions:
                self.conditions[name] = now
        else:
            if name in self.conditions:
                del self.conditions[name]

    def check_alerts(self):
        now = time.time()
        out = {}

        for name, start in list(self.conditions.items()):
            elapsed = now - start

            if elapsed >= self.warning_seconds:
                out[name] = "warning"
            elif elapsed >= self.light_seconds:
                out[name] = "light"

        return out
