#!/usr/bin/env python3
"""Calculate the small matrix and numerical examples for the documentation."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def calculate():
    # High-precision arithmetic checks the log-sum-exp example.
    with localcontext() as ctx:
        ctx.prec = 60
        D = Decimal
        lse = D(1000) + (D(1) + (-D(1)).exp()).ln()
        probability = D(1) / (D(1) + (-D(1)).exp())
    Q = np.array([[2.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, 0.8])
    m1, m2, new_m1 = np.array([0.1, 0.0]), np.array([0.0, 0.2]), np.array([0.2, 0.1])
    u1, u2 = Q @ m1, Q @ m2
    fitted = u1 + u2
    residual = g - (fitted - u1)
    replacement = fitted - u1 + Q @ new_m1
    np.testing.assert_allclose(replacement, Q @ (new_m1 + m2), atol=2e-16, rtol=0)
    V, s2 = 0.2, 0.1
    beta = np.array([0.1, 0.2, 0.3])
    # This is scalar explanatory arithmetic. Use the scalar library operation so
    # NumPy ufunc changes do not flip the last displayed decimal at a rounding tie.
    constant = -0.5 * math.log(1 + V / s2)
    denominator = s2 * (s2 + V)
    bf = constant + 0.5 * beta**2 * V / denominator
    np.testing.assert_allclose(
        bf,
        [-0.5159728110007215, -0.4159728110007215, -0.2493061443340549],
        atol=3e-16,
        rtol=0,
    )
    return dict(
        cache=dict(
            u1=u1.tolist(),
            u2=u2.tolist(),
            fitted=fitted.tolist(),
            residual=residual.tolist(),
            new_m1=new_m1.tolist(),
            replacement=replacement.tolist(),
        ),
        ser=dict(
            V=V,
            s2=s2,
            constant=float(constant),
            denominator=denominator,
            bf=bf.tolist(),
        ),
        matrix_bytes=8 * 5000**2,
        matrix_MiB=8 * 5000**2 / 2**20,
        logsumexp=str(lse),
        normalized_first=str(probability),
    )


def snippets(x):
    c, ser = x["cache"], x["ser"]

    def vec(a):
        return "(" + ", ".join(f"{v:g}" for v in a) + ")"

    return {
        "cache": (
            f"The products are $\\mathbf u_1={vec(c['u1'])}"
            f"^T$ and "
            f"$\\mathbf u_2={vec(c['u2'])}"
            f"^T$, giving $\\mathbf f={vec(c['fitted'])}"
            f"^T$. "
            f"Component 1 uses residual **{vec(c['residual'])}"
            f"**. Replacing its mean by "
            f"$\\mathbf m_1^{{\\mathrm{{new}}}}={vec(c['new_m1'])}"
            f"^T$ changes the fitted "
            f"crossproduct to **{vec(c['replacement'])}"
            f"**.\n"
        ),
        "ser": (
            f"For $s^2={ser['s2']:g}"
            f"$, $V={ser['V']:g}"
            f"$ and marginal estimates "
            f"$(0.1,0.2,0.3)$, the common log term "
            f"is {ser['constant']:.6f}"
            f" and the "
            f"denominator is {ser['denominator']:.2f}"
            f". The log Bayes factors are "
            f"**{vec([round(v, 6) for v in ser['bf']])}"
            f"**.\n"
        ),
        "memory": f"At $p=5000$, $8p^2={x['matrix_bytes']:,}$ bytes, or "
        f"approximately **{x['matrix_MiB']:.1f} MiB**.\n",
        "logsum": (
            f"A 60-digit decimal calculation gives "
            f"**{float(x['logsumexp']):.12f}"
            f"**; "
            f"the first normalized weight is **{float(x['normalized_first']):.12f}"
            f"**.\n"
        ),
    }


def build():
    values = calculate()
    text = snippets(values)
    for name, keys in [
        ("implementation.md", ["logsum"]),
        ("optimization.md", ["cache", "ser", "memory"]),
    ]:
        path = ROOT / "docs" / name
        updated = path.read_text()
        for key in keys:
            start, end = (
                f"<!-- calculated:{key}:start -->",
                f"<!-- calculated:{key}:end -->",
            )
            assert updated.count(start) == updated.count(end) == 1
            before, remainder = updated.split(start)
            _, after = remainder.split(end)
            updated = before + start + "\n" + text[key] + end + after
        path.write_text(updated)
    print("Generated explanatory matrix and numerical calculations")


if __name__ == "__main__":
    build()
