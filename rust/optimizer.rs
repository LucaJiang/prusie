// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from R 4.4 src/library/stats/src/optimize.c, Brent_fmin.
// Copyright (C) 1995, 1996 Robert Gentleman and Ross Ihaka.
// Copyright (C) 2003-2004 The R Foundation.
// Copyright (C) 1998-2023 The R Core Team.
// R's routine derives from Richard Brent, Algorithms for Minimization
// without Derivatives (1973). See LICENSE-R and THIRD_PARTY_NOTICES.md.

/// Minimize a scalar objective with the stopping and step rules used by
/// R's `optim(method = "Brent")`. The tolerance is R optim's default reltol,
/// not optimize's different default tolerance.
pub(super) fn minimize<F>(mut objective: F, lower: f64, upper: f64) -> Result<(f64, f64), String>
where
    F: FnMut(f64) -> f64,
{
    let golden = (3.0 - 5.0_f64.sqrt()) * 0.5;
    let eps = f64::EPSILON.sqrt();
    let mut a = lower;
    let mut b = upper;
    let mut x = a + golden * (b - a);
    let mut v = x;
    let mut w = x;
    let mut fx = objective(x);
    let mut fv = fx;
    let mut fw = fx;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    // This bound is defensive; the finite interval and positive tolerance
    // converge in far fewer steps for all finite likelihoods.
    for _ in 0..1000 {
        let midpoint = (a + b) * 0.5;
        let tolerance = eps * x.abs() + eps / 3.0;
        let twice_tolerance = 2.0 * tolerance;
        if (x - midpoint).abs() <= twice_tolerance - (b - a) * 0.5 {
            return Ok((x, fx));
        }
        let mut p = 0.0;
        let mut q = 0.0;
        let mut previous_e = 0.0;
        if e.abs() > tolerance {
            let r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            previous_e = e;
            e = d;
        }
        if p.abs() >= (q * 0.5 * previous_e).abs() || p <= q * (a - x) || p >= q * (b - x) {
            e = if x < midpoint { b - x } else { a - x };
            d = golden * e;
        } else {
            d = p / q;
            let candidate = x + d;
            if candidate - a < twice_tolerance || b - candidate < twice_tolerance {
                d = if x >= midpoint { -tolerance } else { tolerance };
            }
        }
        let u = x + if d.abs() >= tolerance {
            d
        } else if d > 0.0 {
            tolerance
        } else {
            -tolerance
        };
        let fu = objective(u);
        if !fu.is_finite() {
            return Err("Non-finite likelihood during prior variance optimization".into());
        }
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Err("Prior variance optimization exceeded its numerical iteration limit".into())
}

#[cfg(test)]
mod tests {
    use super::minimize;

    #[test]
    fn brent_interior_and_boundary() {
        let (x, fx) = minimize(|x| (x - 1.25).powi(2), -30.0, 15.0).expect("quadratic");
        assert!((x - 1.25).abs() < 1e-8);
        assert_eq!(fx, (x - 1.25).powi(2));
        let (x, fx) = minimize(|x| x, -30.0, 15.0).expect("linear");
        assert!((x + 30.0).abs() < 1e-6);
        assert_eq!(fx, x);
    }
}
