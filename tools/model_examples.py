#!/usr/bin/env python3
"""Calculate the small probability/matrix examples and check their document text."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def calculate():
    alpha = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1]])
    pip = 1 - np.prod(1 - alpha, axis=0)
    # Independent rational arithmetic checks the probability example.
    with localcontext() as ctx:
        ctx.prec = 60
        D = Decimal
        exact_pip = [
            D(1) - (D(1) - D(a)) * (D(1) - D(b))
            for a, b in [("0.8", "0.1"), ("0.15", "0.8"), ("0.05", "0.1")]
        ]
        lse = D(1000) + (D(1) + (-D(1)).exp()).ln()
        probability = D(1) / (D(1) + (-D(1)).exp())
    np.testing.assert_allclose(pip, list(map(float, exact_pip)), atol=2e-16, rtol=0)
    Q = np.array([[2.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, 0.8])
    m1, m2, new_m1 = np.array([0.1, 0.0]), np.array([0.0, 0.2]), np.array([0.2, 0.1])
    u1, u2 = Q @ m1, Q @ m2
    fitted = u1 + u2
    residual = g - (fitted - u1)
    replacement = fitted - u1 + Q @ new_m1
    np.testing.assert_allclose(replacement, Q @ (new_m1 + m2), atol=2e-16, rtol=0)
    signed_Q = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    v, signs = np.array([0.2, 0.3, -0.1]), np.array([1.0, 1.0, -1.0])
    combined = signs @ v
    product = signed_Q @ v
    np.testing.assert_array_equal(product, signed_Q[:, 0] * combined)
    V, s2 = 0.2, 0.1
    beta = np.array([0.1, 0.2, 0.3])
    constant = -0.5 * np.log(1 + V / s2)
    denominator = s2 * (s2 + V)
    bf = constant + 0.5 * beta**2 * V / denominator
    np.testing.assert_allclose(
        bf,
        [-0.5159728110007215, -0.4159728110007215, -0.2493061443340549],
        atol=3e-16,
        rtol=0,
    )
    return dict(
        pip=pip.tolist(),
        cs_mass=float(np.array([0.50, 0.48]).sum()),
        cs_min_abs_corr=0.9,
        wald_z=5.0,
        wald_n=100,
        wald_adjusted=float(5 * np.sqrt(99 / 123)),
        wald_g=float(np.sqrt(99) * 5 * np.sqrt(99 / 123)),
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
        signed_coefficient=float(combined),
        signed_product=product.tolist(),
        logsumexp=str(lse),
        normalized_first=str(probability),
    )


def snippets(x):
    c, ser = x["cache"], x["ser"]
    vec = lambda a: "(" + ", ".join(f"{v:g}" for v in a) + ")"
    return {
        "wald": f"The adjusted statistic is **{x['wald_adjusted']:.6f}** and the working crossproduct is "
        f"**{x['wald_g']:.6f}** (with $Q_{{jj}}=99$ for unit LD diagonal).\n",
        "pip": f"The resulting PIPs are **{vec(x['pip'])}**, whose sum is {sum(x['pip']):g}.\n",
        "cs": f"The actual posterior mass is **{x['cs_mass']:.2f}**. If the two variants have signed "
        f"correlation $r=-{x['cs_min_abs_corr']:g}$, the minimum, mean and median absolute pairwise "
        f"correlation are all **{x['cs_min_abs_corr']:g}**, so the set passes <code>min_abs_corr=0.5</code>.\n",
        "cache": f"The products are $\\mathbf u_1={vec(c['u1'])}^T$ and "
        f"$\\mathbf u_2={vec(c['u2'])}^T$, giving $\\mathbf f={vec(c['fitted'])}^T$. "
        f"Component 1 uses residual **{vec(c['residual'])}**. Replacing its mean by "
        f"$\\mathbf m_1^{{\\mathrm{{new}}}}={vec(c['new_m1'])}^T$ changes the fitted "
        f"crossproduct to **{vec(c['replacement'])}**.\n",
        "ser": f"For $s^2={ser['s2']:g}$, $V={ser['V']:g}$ and marginal estimates "
        f"$(0.1,0.2,0.3)$, the common log term is {ser['constant']:.6f} and the "
        f"denominator is {ser['denominator']:.2f}. The log Bayes factors are "
        f"**{vec([round(v, 6) for v in ser['bf']])}**.\n",
        "memory": f"At $p=5000$, $8p^2={x['matrix_bytes']:,}$ bytes, or "
        f"approximately **{x['matrix_MiB']:.1f} MiB**.\n",
        "columns": f"The signed coefficient is $0.2+0.3-(-0.1)={x['signed_coefficient']:g}$, "
        f"and the product is **{vec(x['signed_product'])}**.\n",
        "logsum": f"A 60-digit decimal calculation gives **{float(x['logsumexp']):.12f}**; "
        f"the first normalized weight is **{float(x['normalized_first']):.12f}**.\n",
    }


def build(check=False):
    values = calculate()
    text = snippets(values)
    stale = []
    for name, keys in [
        ("model.md", ["wald", "pip", "cs"]),
        ("implementation.md", ["cache", "ser", "memory", "columns", "logsum"]),
    ]:
        path = ROOT / "docs" / name
        original = updated = path.read_text()
        for key in keys:
            start, end = (
                f"<!-- calculated:{key}:start -->",
                f"<!-- calculated:{key}:end -->",
            )
            assert updated.count(start) == updated.count(end) == 1
            before, remainder = updated.split(start)
            _, after = remainder.split(end)
            updated = before + start + "\n" + text[key] + end + after
        if check:
            if updated != original:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(updated)
    path = ROOT / "docs/assets/model_calculations.json"
    # These are displayed instructional values, not frozen inference references.
    # Ignore insignificant last-bit platform differences in elementary arithmetic.
    def display_values(value):
        if isinstance(value, float):
            return float(f"{value:.15g}")
        if isinstance(value, dict):
            return {key: display_values(item) for key, item in value.items()}
        if isinstance(value, list):
            return [display_values(item) for item in value]
        return value

    expected = json.dumps(display_values(values), indent=2, allow_nan=False) + "\n"
    if check:
        if not path.exists() or path.read_text() != expected:
            stale.append(str(path.relative_to(ROOT)))
    else:
        path.write_text(expected)
    if stale:
        raise SystemExit("Stale explanatory calculations: " + ", ".join(stale))
    print(
        "Checked explanatory probabilities, Wald adjustment, matrix products, SER and log-sum-exp"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)
