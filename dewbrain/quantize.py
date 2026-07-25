"""
quantize.py — ship the learned identity net SMALL and FAST (edge/on-prem infra).

The moat (identity_net) only matters if it runs where the data lives: on Ahmet's
laptop, on George's air-gapped box, in a browser tab — no GPU, no cloud, no data
leaving. That is an INFRASTRUCTURE problem (CLAUDE.md §2: quantization / edge
runtime), not a modelling one. This module is that infra layer for our own net.

What it does, measured not claimed:
  1. QUANTIZE the trained float32 adapter to int8 (dynamic post-training quant on
     the Linear layers — the standard CPU-inference path). The adapter is the only
     thing shipped, so its bytes and its latency are the deploy cost per person.
  2. MEASURE the three numbers that decide if it ships:
       - size:    float32 bytes vs int8 bytes (download / storage per person)
       - latency: ms to project a batch, fp32 vs int8 (inference cost)
       - fidelity: does int8 KEEP the identity AUC? A fast net that lost the moat
                   (AUC collapsed to mpnet's 0.51) is worthless. The gate is
                   "fidelity held", per prove-don't-claim.

Why this is real infra and not a toy: this is exactly the LLM-weight-quantization
family from §2, at the scale that actually runs on a customer's own machine today.
No LLM in this file at all — strip the (non-existent) LLM and you still have a
quantized, benchmarked, deployable network. Wrapper test: passed by construction.

torch dynamic quantization is CPU-only and needs the qnnpack/fbgemm backend; we
select it explicitly and fall back honestly if unavailable (report, never fake).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import identity_net as idn


@dataclass
class QuantReport:
    fp32_bytes: int
    int8_bytes: int
    fp32_ms: float
    int8_ms: float
    fp32_auc: float
    int8_auc: float
    backend: str

    @property
    def size_ratio(self) -> float:
        return self.fp32_bytes / max(1, self.int8_bytes)

    @property
    def speedup(self) -> float:
        return self.fp32_ms / max(1e-9, self.int8_ms)

    @property
    def fidelity_held(self) -> bool:
        # int8 must keep essentially all the identity signal (allow tiny slack).
        return self.int8_auc >= self.fp32_auc - 0.02

    def as_dict(self) -> dict:
        return {"fp32_kb": round(self.fp32_bytes / 1024, 1),
                "int8_kb": round(self.int8_bytes / 1024, 1),
                "size_ratio": round(self.size_ratio, 2),
                "fp32_ms": round(self.fp32_ms, 2),
                "int8_ms": round(self.int8_ms, 2),
                "speedup": round(self.speedup, 2),
                "fp32_auc": round(self.fp32_auc, 4),
                "int8_auc": round(self.int8_auc, 4),
                "fidelity_held": self.fidelity_held,
                "backend": self.backend}


def _state_bytes(module: nn.Module) -> int:
    """On-disk size of the weights (what actually ships per person)."""
    import io
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


def _bench_ms(fn, X: torch.Tensor, iters: int = 50) -> float:
    """Mean ms per full-batch projection (CPU inference latency)."""
    with torch.no_grad():
        fn(X)  # warmup
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(X)
        return (time.perf_counter() - t0) / iters * 1000.0


def quantize_dynamic(net: idn.IdentityEncoder) -> tuple[nn.Module, str]:
    """Dynamic int8 quantization of the adapter's Linear layers (CPU inference).

    Dynamic PTQ quantizes weights to int8 offline and activations on-the-fly at
    run time — the right choice for a small MLP with no calibration data needed,
    so it works on a fresh customer machine with zero setup. Backend picked
    explicitly; qnnpack (arm/mac) or fbgemm (x86)."""
    backend = "qnnpack"
    try:
        torch.backends.quantized.engine = backend
    except Exception:
        backend = "fbgemm"
        torch.backends.quantized.engine = backend
    qnet = torch.quantization.quantize_dynamic(
        net.eval(), {nn.Linear}, dtype=torch.qint8)
    return qnet, backend


def evaluate(name: str = "default", seed: int = 0, log=print) -> QuantReport:
    """Quantize the trained net and report size / latency / fidelity on the
    held-out test split — the same honest AUC gate identity_net.train uses."""
    net, rep = idn.load(name)
    if net is None:
        raise SystemExit("once identity_net egit (python identity_net.py); agirlik yok.")

    data = idn.build_data(seed=seed)
    Xte = data.X[data.test_idx]
    gte = data.groups[data.test_idx]
    Xte_t = torch.from_numpy(Xte.astype(np.float32))

    qnet, backend = quantize_dynamic(net)

    # fidelity: AUC in each net's projected space, same test split.
    with torch.no_grad():
        Zf = net(Xte_t).numpy()
        Zq = qnet(Xte_t).numpy()
    fp32_auc = idn.pair_auc(Zf, gte, np.random.default_rng(seed))
    int8_auc = idn.pair_auc(Zq, gte, np.random.default_rng(seed))

    report = QuantReport(
        fp32_bytes=_state_bytes(net), int8_bytes=_state_bytes(qnet),
        fp32_ms=_bench_ms(lambda x: net(x), Xte_t),
        int8_ms=_bench_ms(lambda x: qnet(x), Xte_t),
        fp32_auc=fp32_auc, int8_auc=int8_auc, backend=backend)
    return report


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("kimlik agini nicemleme (int8, edge/on-prem infra)...")
    rep = evaluate()
    d = rep.as_dict()
    print(f"\n=== NICEMLEME SONUCU (kanit, backend={d['backend']}) ===")
    print(f"  boyut : fp32 {d['fp32_kb']} KB -> int8 {d['int8_kb']} KB  ({d['size_ratio']}x kucuk)")
    print(f"  hiz   : fp32 {d['fp32_ms']} ms -> int8 {d['int8_ms']} ms  ({d['speedup']}x)")
    print(f"  moat  : AUC fp32 {d['fp32_auc']} -> int8 {d['int8_auc']}  (korundu mu: {d['fidelity_held']})")
    if not d["fidelity_held"]:
        print("  -> int8 kimligi BOZDU (AUC dustu). Bu haliyle shiplenmez, fp32 kalir.")
    else:
        print("  -> int8 kimligi korudu. Kucuk+hizli+moat saglam = herkesin makinesinde kosar.")
