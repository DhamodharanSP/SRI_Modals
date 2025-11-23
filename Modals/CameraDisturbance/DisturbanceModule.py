import time
import cv2
import numpy as np
from CameraDisturbance.Utils import (
    resize_for_check, brightness_mean, std_dev,
    laplacian_variance, histogram_spread, block_variances
)
from CameraDisturbance.HealthScorer import HealthScorer
from CameraDisturbance.AlertManager import AlertManager


class DisturbanceDetector:
    def __init__(self, proc_size=(128, 128), block_grid=(8, 8),
                 light_seconds=5.0, warning_seconds=10.0):

        self.proc_size = proc_size
        self.block_grid = block_grid

        self.alerts = AlertManager(light_seconds, warning_seconds)
        self.health_scorer = HealthScorer()

        self.uniform_std_threshold = 5.0
        self.block_low_var_threshold = 8.0
        self.blur_threshold = 50.0
        self.hist_spread_low = 10.0

        self.last_frame_time = None
        self.offline_timeout = 3.0

    def analyze_frame(self, frame):
        now = time.time()

        small_gray = resize_for_check(frame, self.proc_size)

        mean_brightness = brightness_mean(small_gray)
        global_std = std_dev(small_gray)
        lap_var = laplacian_variance(small_gray)
        hist_spread = histogram_spread(small_gray)

        bvars = block_variances(small_gray, self.block_grid)
        block_low_mask = (bvars < self.block_low_var_threshold)
        partial_fraction = float(block_low_mask.sum()) / bvars.size

        full_obstruction = (global_std < self.uniform_std_threshold)
        partial_occlusion = partial_fraction >= 0.1 and not full_obstruction
        over_blur = lap_var < self.blur_threshold
        poor_exposure = hist_spread < self.hist_spread_low

        self.last_frame_time = now

        brightness_score = min(100, (mean_brightness / 255.0) * 100)
        blur_score = min(100, (lap_var / 200.0) * 100)
        exposure_score = min(100, (hist_spread / 128.0) * 100)
        obstruction_score = 100 - (partial_fraction * 150)

        metrics = {
            "brightness": brightness_score,
            "blur": blur_score,
            "exposure": exposure_score,
            "obstruction": obstruction_score,
            "network": 100,
        }

        health = self.health_scorer.score(metrics)

        self.alerts.update_condition("full_obstruction", full_obstruction)
        self.alerts.update_condition("partial_occlusion", partial_occlusion)
        self.alerts.update_condition("blur", over_blur)
        self.alerts.update_condition("poor_exposure", poor_exposure)

        alerts = self.alerts.check_alerts()

        return {
            "timestamp": now,
            "brightness": mean_brightness,
            "std": global_std,
            "lap_var": lap_var,
            "hist_spread": hist_spread,
            "partial_fraction": partial_fraction,
            "full_obstruction": full_obstruction,
            "partial_occlusion": partial_occlusion,
            "blur": over_blur,
            "poor_exposure": poor_exposure,
            "metrics": metrics,
            "health": health,
            "alerts": alerts
        }

    def check_offline(self):
        if self.last_frame_time is None:
            return True
        return (time.time() - self.last_frame_time) > self.offline_timeout
