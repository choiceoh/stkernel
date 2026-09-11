"""Every lane says "I served", or the pipeline refuses (base).

D3's evidence is four `armed != serving` incidents and an IndexCache rule that
silently skipped its first layer. The fix is not more care; it is a contract:
a lane that a profile declares must REPORT, at serving time, that it ran --
with a value a reader can check -- and a run whose report is missing a
declared lane is not a run.

A Lane is declared by a profile or module. A Report is what the runner
collects while serving. `verify` is the gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lane:
    name: str            # e.g. "moe.nvfp4.b12x", "linear_attention.gdn.triton"
    owner: str           # module or profile that declared it
    expects: str         # what the served value must look like, for a reader


@dataclass
class Report:
    served: dict = field(default_factory=dict)   # lane -> evidence value

    def mark(self, lane: str, evidence) -> None:
        if lane in self.served and self.served[lane] != evidence:
            raise ValueError(f"lane {lane} reported twice with different evidence: "
                             f"{self.served[lane]!r} then {evidence!r}")
        self.served[lane] = evidence


class ProofError(SystemExit):
    pass


def verify(declared: "list[Lane]", report: Report) -> "list[str]":
    """Lines a reader can keep. Raises if any declared lane stayed silent."""
    silent = [l for l in declared if l.name not in report.served]
    if silent:
        raise ProofError("proof: declared lane(s) never reported serving -- "
                         + ", ".join(f"{l.name} ({l.owner}, expects {l.expects})" for l in silent))
    extra = sorted(set(report.served) - {l.name for l in declared})
    if extra:
        raise ProofError(f"proof: lane(s) served that nobody declared: {extra}")
    return [f"[proof] {l.name} served: {report.served[l.name]!r}" for l in declared]


def _selfcheck() -> None:
    lanes = [Lane("moe.nvfp4", "modules.quant", "kernel tag"), Lane("attn.sparse", "modules.sparse_attention", "topk")]
    r = Report(); r.mark("moe.nvfp4", "b12x_u"); r.mark("attn.sparse", 512)
    assert verify(lanes, r)[0].startswith("[proof] moe.nvfp4 served")
    r2 = Report(); r2.mark("moe.nvfp4", "b12x_u")
    try:
        verify(lanes, r2); raise AssertionError("silent lane must fail")
    except ProofError as e:
        assert "attn.sparse" in str(e)
    r3 = Report(); r3.mark("moe.nvfp4", "x"); r3.mark("attn.sparse", 1); r3.mark("ghost", 0)
    try:
        verify(lanes, r3); raise AssertionError("undeclared lane must fail")
    except ProofError:
        pass
    print("  proof: silent lane refused, undeclared lane refused OK")


if __name__ == "__main__":
    _selfcheck()
