# SPDX-License-Identifier: Apache-2.0
"""Read a slice of a sharded checkpoint. No vLLM, no engine, no fleet.

The point is the SLICE. This checkpoint is 185 GB in 11 shards and 148,498
tensors; the serving stack needs four nodes and 4-8 minutes to have any of it in
memory, which is why every question in 40차 cost a fleet hold. safetensors can
open a shard and read one tensor out of it, so a few layers land on one GPU in
seconds, and a bug in a layer's shapes, dtypes or scale layout reproduces there
instead of on the fleet.

Layer names come from the index rather than from a hardcoded prefix: the model
puts them under `model.language_model.layers.N.`, and a checkpoint that spells
that differently should still slice.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

_LAYER = re.compile(r"(?P<prefix>.*\.layers\.)(?P<index>\d+)\.")


class Checkpoint:
    def __init__(self, path: str):
        self.path = path
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path) as fh:
            self.weight_map = json.load(fh)["weight_map"]
        self.layer_prefix, self.num_layers = self._probe_layers()

    def _probe_layers(self) -> tuple[str, int]:
        prefixes, highest = set(), -1
        for name in self.weight_map:
            m = _LAYER.match(name)
            if m:
                prefixes.add(m.group("prefix"))
                highest = max(highest, int(m.group("index")))
        if len(prefixes) != 1:
            raise ValueError(f"expected one layer prefix, found {sorted(prefixes)[:4]}")
        return prefixes.pop(), highest + 1

    def layer_of(self, name: str) -> int | None:
        m = _LAYER.match(name)
        return int(m.group("index")) if m else None

    def keys_for(self, layers: range | list[int], *, include_shared: bool = False) -> list[str]:
        """Tensor names for these layers; `include_shared` adds everything that
        belongs to no layer (embeddings, the final norm, the lm head)."""
        wanted = set(layers)
        out = []
        for name in self.weight_map:
            idx = self.layer_of(name)
            if idx is None:
                if include_shared:
                    out.append(name)
            elif idx in wanted:
                out.append(name)
        return sorted(out)

    def load(self, keys: list[str], *, device: str = "cpu", recorder=None) -> dict:
        """Read exactly these tensors, one shard open at a time.

        Grouping by shard is not a micro-optimisation: opening 11 shards once
        each and pulling the keys out beats reopening per tensor by orders of
        magnitude at 148k tensors, and it is the difference between a harness
        that runs in seconds and one nobody uses."""
        from safetensors import safe_open

        by_shard = defaultdict(list)
        for name in keys:
            by_shard[self.weight_map[name]].append(name)
        tensors, total = {}, 0
        for shard, names in sorted(by_shard.items()):
            full = os.path.join(self.path, shard)
            with safe_open(full, framework="pt", device=device) as fh:
                for name in names:
                    t = fh.get_tensor(name)
                    tensors[name] = t
                    total += t.numel() * t.element_size()
            if recorder is not None:
                recorder.count("shards", 1)
        if recorder is not None:
            recorder.gauge("tensors", len(tensors))
            recorder.gauge("GiB", round(total / (1 << 30), 3))
        return tensors

    def summary(self) -> dict:
        dtypes: dict = {}
        per_layer = defaultdict(int)
        for name, shard in self.weight_map.items():
            idx = self.layer_of(name)
            per_layer["shared" if idx is None else idx] += 1
        return {"tensors": len(self.weight_map), "shards": len(set(self.weight_map.values())),
                "layers": self.num_layers, "prefix": self.layer_prefix,
                "tensors_per_layer": per_layer.get(0, 0), "shared_tensors": per_layer.get("shared", 0),
                "dtypes": dtypes}
