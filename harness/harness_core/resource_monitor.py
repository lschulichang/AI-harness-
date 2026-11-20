"""记录本地批次决策延迟、CPU 时间和采样内存。"""

import os
import time
from threading import Event, Thread

import psutil


def cpu_seconds():
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


class BatchMeter:
    def __init__(self, ledger):
        self.ledger = ledger
        self.process = psutil.Process()
        self.key = None

    def _sample(self):
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            return
        rss = 0
        for process in processes:
            try:
                rss += process.memory_info().rss
            except psutil.Error:
                if process is self.process:
                    return
        self.peak_rss = max(self.peak_rss, rss)

    def _sample_loop(self):
        while not self.stop.wait(0.05):
            self._sample()

    def start(self, batch, *, at_entry):
        key = f"{batch:02d}"
        if self.key == key:
            return
        self.close()
        previous = self.ledger.get("batch_profile", key)
        if previous and previous["sealed"]:
            return
        self.key = key
        self.previous = previous or {}
        self.started = time.perf_counter()
        self.cpu_started = cpu_seconds()
        self.complete = at_entry and previous is None
        self.peak_rss = 0
        self._sample()
        self.stop = Event()
        self.sampler = Thread(target=self._sample_loop, daemon=True)
        self.sampler.start()

    def checkpoint(self, *, sealed=False):
        if self.key is None:
            return None
        fields = {
            "decision_seconds": self.previous.get("decision_seconds", 0) + (time.perf_counter() - self.started),
            "cpu_seconds": self.previous.get("cpu_seconds", 0) + max(0, cpu_seconds() - self.cpu_started),
            "peak_rss_mib": max(self.previous.get("peak_rss_mib") or 0, self.peak_rss / 2**20) or None,
            "profile_complete": self.complete,
        }
        self.ledger.put("batch_profile", self.key, {**fields, "sealed": sealed})
        return fields

    def finish(self, batch):
        key = f"{batch:02d}"
        if self.key is None:
            return {k: v for k, v in self.ledger.get("batch_profile", key).items() if k != "sealed"}
        self.stop.set()
        self.sampler.join()
        self._sample()
        fields = self.checkpoint(sealed=True)
        self.key = None
        return fields

    def close(self):
        if self.key is not None:
            self.stop.set()
            self.sampler.join()
            self._sample()
            self.checkpoint()
            self.key = None
