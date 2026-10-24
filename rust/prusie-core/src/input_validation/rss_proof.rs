//! Optional private-pool version of the unchanged sufficient RSS proof.
use super::RssPairLayout;
use rayon::{prelude::*, ThreadPool};

pub(crate) fn rss_proof_parallel(
    pool: &ThreadPool,
    values: &[f64],
    n: usize,
    scale: f64,
    _layout: RssPairLayout,
) -> Option<bool> {
    if n < 4
        || n.checked_mul(n) != Some(values.len())
        || values.len() > isize::MAX as usize / std::mem::size_of::<f64>()
    {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !(scale.abs() <= f64::MAX * 0.5) {
            return Some(false);
        }
        // Preserve the parent's complete diagonal conditions and arithmetic.
        for i in 0..n {
            let raw = values[i * n + i];
            let scaled = raw * scale;
            if !(raw.is_finite()
                && (raw - 1.0).abs() <= 1e-8
                && raw.abs() <= 1.0 + 1e-8
                && raw >= 0.0
                && scaled.is_finite()
                && scaled >= 0.0)
            {
                return Some(false);
            }
        }
        let threads = pool.current_num_threads().min(n.div_ceil(32).max(1));
        let ranks: Vec<bool> = pool.install(|| {
            (0..threads)
                .into_par_iter()
                .map(|rank| exact_contiguous_rss_pairs(values, n, rank, threads))
                .collect()
        });
        Some(ranks.into_iter().fold(true, |valid, rank| valid & rank))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (pool, scale);
        None
    }
}

#[cfg(target_arch = "x86_64")]
fn exact_contiguous_rss_pairs(values: &[f64], n: usize, rank: usize, threads: usize) -> bool {
    // Cyclic rows balance triangular work; within each row the original
    // 32×32/4×4 proof and complete checks remain unchanged.
    const BLOCK: usize = 32;
    for ii in (rank * BLOCK..n).step_by(BLOCK * threads) {
        for jj in (ii..n).step_by(BLOCK) {
            for i in (ii..(ii + BLOCK).min(n)).step_by(4) {
                for j in (jj.max(i)..(jj + BLOCK).min(n)).step_by(4) {
                    if j > i && i + 4 <= n && j + 4 <= n {
                        // SAFETY: both 4 × 4 blocks are inside the checked n × n
                        // contiguous slice; SSE2 is an x86-64 baseline feature.
                        if !unsafe { exact_block4(values.as_ptr(), n, i, j) } {
                            return false;
                        }
                    } else {
                        for row in i..(i + 4).min(n) {
                            for col in j.max(row + 1)..(j + 4).min(n) {
                                let x = values[row * n + col];
                                let y = values[col * n + row];
                                if !(x == y && x.abs() <= 1.0 + 1e-8) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

/// SSE2 is part of the x86-64 target baseline. The caller proves both complete
/// four-by-four blocks are in bounds and keeps the NumPy read borrow/GIL alive.
// SAFETY: base must address a live contiguous n*n f64 slice, and both
// transposed four-by-four blocks starting at (i,j) must lie within that slice.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn exact_block4(base: *const f64, n: usize, i: usize, j: usize) -> bool {
    use std::arch::x86_64::*;
    let abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7fff_ffff_ffff_ffff));
    let bound = _mm_set1_pd(1.0 + 1e-8);
    let mut valid = _mm_castsi128_pd(_mm_set1_epi64x(-1));
    for half in [0, 2] {
        let b0 = _mm_loadu_pd(base.add(j * n + i + half));
        let b1 = _mm_loadu_pd(base.add((j + 1) * n + i + half));
        let b2 = _mm_loadu_pd(base.add((j + 2) * n + i + half));
        let b3 = _mm_loadu_pd(base.add((j + 3) * n + i + half));
        let columns = [
            (_mm_unpacklo_pd(b0, b1), _mm_unpacklo_pd(b2, b3)),
            (_mm_unpackhi_pd(b0, b1), _mm_unpackhi_pd(b2, b3)),
        ];
        for (offset, (low, high)) in columns.into_iter().enumerate() {
            let row = i + half + offset;
            let x0 = _mm_loadu_pd(base.add(row * n + j));
            let x1 = _mm_loadu_pd(base.add(row * n + j + 2));
            valid = _mm_and_pd(valid, _mm_cmpeq_pd(x0, low));
            valid = _mm_and_pd(valid, _mm_cmpeq_pd(x1, high));
            valid = _mm_and_pd(valid, _mm_cmple_pd(_mm_and_pd(x0, abs_mask), bound));
            valid = _mm_and_pd(valid, _mm_cmple_pd(_mm_and_pd(x1, abs_mask), bound));
        }
    }
    _mm_movemask_pd(valid) == 3
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn oracle(values: &[f64], n: usize, scale: f64) -> bool {
        if !(scale.abs() <= f64::MAX * 0.5) {
            return false;
        }
        for i in 0..n {
            let raw = values[i * n + i];
            let scaled = raw * scale;
            if !(raw.is_finite()
                && (raw - 1.).abs() <= 1e-8
                && raw.abs() <= 1. + 1e-8
                && raw >= 0.
                && scaled.is_finite()
                && scaled >= 0.)
            {
                return false;
            }
            for j in i + 1..n {
                if !(values[i * n + j] == values[j * n + i] && values[i * n + j].abs() <= 1. + 1e-8)
                {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn rss_proof_parallel_matches_all_parent_conditions_at_tiles_and_tails() {
        for threads in [2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for n in [4, 5, 31, 32, 33, 34, 35, 63, 64, 65, 66, 67, 129, 1025] {
                let mut values = vec![0.125; n * n];
                for i in 0..n {
                    values[i * n + i] = 1.;
                }
                values[1] = -0.;
                values[n] = 0.;
                for scale in [
                    0.,
                    -0.,
                    1.,
                    -1.,
                    99.,
                    f64::MAX / 2.,
                    f64::MAX,
                    f64::from_bits(1),
                    f64::NAN,
                ] {
                    for layout in [RssPairLayout::C, RssPairLayout::F] {
                        assert_eq!(
                            rss_proof_parallel(&pool, &values, n, scale, layout),
                            Some(oracle(&values, n, scale))
                        );
                    }
                }
                for index in [0, 1, n * n / 2, n * n - 2, n * n - 1] {
                    let old = values[index];
                    for bad in [
                        -1.,
                        1. + 2e-8,
                        f64::MAX,
                        f64::INFINITY,
                        f64::NAN,
                        -f64::from_bits(1),
                    ] {
                        values[index] = bad;
                        assert_eq!(
                            rss_proof_parallel(&pool, &values, n, 99., RssPairLayout::C),
                            Some(oracle(&values, n, 99.)),
                            "n={n}, index={index}"
                        );
                    }
                    values[index] = old;
                }
            }
            assert!(rss_proof_parallel(&pool, &[1.; 15], 4, 99., RssPairLayout::C).is_none());
        }
    }
}
