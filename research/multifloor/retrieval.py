"""Rig-consistent DINO retrieval with per-level, time-diverse candidate budgets."""
from collections import Counter
import numpy as np
from .floors import floor_relation


def camera_scores(query, database):
    return np.einsum('avd,bwd->avbw', query, database).max(axis=(1, 3))


class RigRetrieval:
    def __init__(self, descriptors, nodes, times, labels, minimum_time_gap=35.):
        self.desc = np.asarray(descriptors)
        self.times = np.asarray(times, float)
        self.labels = labels
        self.gap = minimum_time_gap
        self.nodes = sorted(int(n) for n in nodes if np.isfinite(self.desc[n]).all()
                            and np.all(np.linalg.norm(self.desc[n], axis=-1) > .5))
        self.cp = {}
        for i in self.nodes:
            for j in self.nodes:
                if j >= i or self.times[i]-self.times[j] < self.gap:
                    continue
                self.cp[i,j] = camera_scores(self.desc[i], self.desc[j])
        values = np.array(list(self.cp.values()))
        self.mean = values.mean(0) if len(values) else np.zeros((4,4))
        self.std = np.maximum(values.std(0), 1e-6) if len(values) else np.ones((4,4))
        self.level_stats = {}
        for level in sorted({v for v in labels if v is not None}):
            level_values = np.array([cp for (i,j),cp in self.cp.items()
                                     if labels[i]==level and labels[j]==level])
            if len(level_values) >= 32:
                self.level_stats[level] = (level_values.mean(0), np.maximum(level_values.std(0),1e-6))

    def score(self, i, j):
        mean,std = self.level_stats.get(self.labels[i],(self.mean,self.std))
        cp = (self.cp[i,j]-mean)/std
        scores = [np.mean([cp[c,(c+s)%4] for c in range(4)]) for s in range(4)]
        shift = int(np.argmax(scores))
        return float(scores[shift]), shift

    def shortlist(self, i, limit=12, unknown_limit=3, bin_seconds=12., per_bin=2):
        ranked = sorted(((self.score(i,j), j) for j in self.nodes if (i,j) in self.cp),
                        key=lambda x: (-x[0][0], x[1]))
        selected, counts, audit = [], Counter(), Counter()
        origin = self.times[0]
        # Known same-level candidates get their own budget before unknowns.
        # Cross-level aliases never consume either budget.
        for desired, cap in [('same', limit), ('unknown', unknown_limit)]:
            if cap <= 0:
                continue
            count = 0
            for rank, ((score, shift), j) in enumerate(ranked, 1):
                relation = floor_relation(i,j,self.labels)
                if relation != desired:
                    continue
                bucket = (desired, int((self.times[j]-origin)//bin_seconds))
                if counts[bucket] >= per_bin:
                    continue
                selected.append(dict(query=i, match=j, global_rank=rank, score=score,
                                     shift=shift, floor_relation=relation,
                                     floor_id=self.labels[i], retrieval_bucket=bucket[1]))
                counts[bucket] += 1
                count += 1
                if count >= cap:
                    break
        for _, j in ranked:
            audit[floor_relation(i,j,self.labels)] += 1
        for rank, row in enumerate(selected, 1):
            row['rank'] = rank
        return selected, dict(audit)
