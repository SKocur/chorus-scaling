"""Ember, the table optimiser of the Chorus paper (arXiv 2611.23386), for sparse tables.

After `ChorusMemory.oe_mfree_step` in `reference/chorus/chorus/memory_modules.py`: no
momentum, and the second moment of each table factored into one value per row and one per
column, kept as running means with beta2 0.95 and bias correction. A row read in a step
moves by lr * g / sqrt(r_row * c_col / mean(c)); rows not read do not move, but their row
statistic decays as if their gradient were zero, as in the authors' dense version. The
state is one float per row and one per column instead of Adam's two per parameter.

Our tables are flat: every N-gram order and head of a module (and Chorus's token table)
is a segment of one `nn.Embedding`. The authors keep each in its own table with its own
column statistic, so here the column statistic is kept per segment. One difference stays:
Chorus's token table is stored as four rows of width d/4 per token, so its column
statistic has d/4 entries where the authors' has d.
"""

from __future__ import annotations

import torch
from torch import nn


class Ember(torch.optim.Optimizer):
    def __init__(self, params, segments: list[list[int]], lr: float, beta2: float = 0.95,
                 eps: float = 1e-10):
        """`segments` gives, for each parameter in order, the first row of each of its
        segments, starting with 0."""
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps))
        flat = [p for g in self.param_groups for p in g["params"]]
        if len(segments) != len(flat):
            raise ValueError("one list of segment starts per parameter")
        for p, starts in zip(flat, segments):
            if starts[0] != 0 or sorted(starts) != list(starts) or starts[-1] >= p.size(0):
                raise ValueError(f"bad segment starts {starts[:4]}... for {p.size(0)} rows")
            self.state[p]["starts"] = torch.tensor(starts, dtype=torch.long, device=p.device)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta2, eps = group["lr"], group["beta2"], group["eps"]
            for p in group["params"]:
                state = self.state[p]
                starts = state["starts"]
                if "row" not in state:
                    state["step"] = 0
                    state["row"] = torch.zeros(p.size(0), dtype=torch.float32, device=p.device)
                    state["col"] = torch.zeros(len(starts), p.size(1), dtype=torch.float32,
                                               device=p.device)
                    ends = torch.cat([starts[1:], starts.new_tensor([p.size(0)])])
                    state["sizes"] = (ends - starts).float()
                if p.grad is None:
                    continue
                state["step"] += 1
                row, col = state["row"], state["col"]
                row.mul_(beta2)
                col.mul_(beta2)
                grad = p.grad.coalesce() if p.grad.is_sparse else p.grad.to_sparse().coalesce()
                idx, g = grad.indices()[0], grad.values().float()
                if idx.numel() == 0:
                    continue
                g2 = g.square()
                seg = torch.searchsorted(starts, idx, right=True) - 1
                row.index_add_(0, idx, (1 - beta2) * g2.mean(-1))
                col_sum = torch.zeros_like(col).index_add_(0, seg, g2)
                col.add_((1 - beta2) * col_sum / state["sizes"][:, None])
                bias = 1 - beta2 ** state["step"]
                rh = (row[idx] / bias).clamp_min(1e-30)
                ch = (col / bias).clamp_min(1e-30)
                cn = ch / ch.mean(-1, keepdim=True).clamp_min(1e-30)
                denom = (rh[:, None] * cn[seg]).sqrt() + eps
                p.index_add_(0, idx, (-lr * g / denom).to(p.dtype))


def table_segments(model: nn.Module, table: nn.Parameter) -> list[int]:
    """First row of each segment of a flat Engram or Chorus table."""
    for module in model.modules():
        weight = getattr(getattr(module, "table", None), "weight", None)
        if weight is table:
            starts = [int(s) for s in module.offsets]
            if hasattr(module, "token_offset"):
                starts.append(int(module.token_offset))
            return starts
    raise ValueError("no Engram or Chorus module owns this table")
