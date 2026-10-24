//! Paired validation that initializes the caller's unique owned RSS repair.
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
pub enum RssPairLayout {
    C,
    F,
}

/// Independent predicates for the existing raw/scaled/repair error priority.
/// No predicate asserts that a caller or a preceding proof validated the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RssPairValidation {
    pub raw_finite: bool,
    pub raw_nonnegative: bool,
    pub scaled_finite: bool,
    pub scaled_nonnegative: bool,
    pub symmetric: bool,
    pub repair_finite: bool,
    pub repair_nonnegative: bool,
}

impl Default for RssPairValidation {
    fn default() -> Self {
        Self {
            raw_finite: true,
            raw_nonnegative: true,
            scaled_finite: true,
            scaled_nonnegative: true,
            symmetric: true,
            repair_finite: true,
            repair_nonnegative: true,
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl RssPairValidation {
    #[inline]
    fn pair(&mut self, x: f64, y: f64, scale: f64, diagonal: bool) -> f64 {
        self.raw_finite &= x.is_finite() & y.is_finite();
        let sx = x * scale;
        let sy = y * scale;
        self.scaled_finite &= sx.is_finite() & sy.is_finite();
        self.symmetric &= sx == sy;
        let value = (sx + sy) / 2.;
        self.repair_finite &= value.is_finite();
        if diagonal {
            self.raw_nonnegative &= x >= 0.;
            self.scaled_nonnegative &= sx >= 0.;
            self.repair_nonnegative &= value >= 0.;
        }
        value
    }
}

/// Validate all raw/scaled/repair entries and initialize every C-order output
/// cell exactly once. Input and output must be separate unique Rust borrows.
/// Unsupported dimensions/targets return None before any output is written.
pub fn rss_pair_validation_owned(
    values: &[f64],
    n: usize,
    scale: f64,
    layout: RssPairLayout,
    output: &mut [MaybeUninit<f64>],
) -> Option<RssPairValidation> {
    if n < 4
        || n.checked_mul(n) != Some(values.len())
        || values.len() != output.len()
        || values.len() > isize::MAX as usize / std::mem::size_of::<f64>()
    {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        Some(contiguous_pairs(
            values,
            n,
            scale,
            layout,
            OwnedRepair {
                values: output,
                #[cfg(test)]
                writes: None,
            },
        ))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (scale, layout);
        None
    }
}

struct OwnedRepair<'a> {
    values: &'a mut [MaybeUninit<f64>],
    #[cfg(test)]
    writes: Option<&'a mut [usize]>,
}
impl OwnedRepair<'_> {
    fn scalar(&mut self, index: usize, value: f64) {
        self.values[index].write(value);
        #[cfg(test)]
        if let Some(writes) = self.writes.as_deref_mut() {
            writes[index] += 1;
        }
    }
    // SAFETY: caller proves two contiguous output cells are in bounds. The
    // MaybeUninit borrow permits writes before any f64 reference is formed.
    #[cfg(target_arch = "x86_64")]
    unsafe fn pair(&mut self, index: usize, value: std::arch::x86_64::__m128d) {
        std::arch::x86_64::_mm_storeu_pd(self.values.as_mut_ptr().add(index).cast::<f64>(), value);
        #[cfg(test)]
        if let Some(writes) = self.writes.as_deref_mut() {
            writes[index] += 1;
            writes[index + 1] += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn contiguous_pairs(
    values: &[f64],
    n: usize,
    scale: f64,
    layout: RssPairLayout,
    mut output: OwnedRepair<'_>,
) -> RssPairValidation {
    use std::arch::x86_64::*;
    let mut result = RssPairValidation::default();
    for i in 0..n {
        let raw = values[i * n + i];
        let value = result.pair(raw, raw, scale, true);
        output.scalar(i * n + i, value);
    }
    // SAFETY: SSE2 is part of x86-64. Every full paired tile below has four
    // complete rows and columns inside the validated n*n immutable slice.
    unsafe {
        let mut masks = PairMasks {
            raw: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
            scaled: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
            symmetric: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
            repair: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
        };
        const BLOCK: usize = 32;
        for ii in (0..n).step_by(BLOCK) {
            for jj in (ii..n).step_by(BLOCK) {
                for i in (ii..(ii + BLOCK).min(n)).step_by(4) {
                    for j in (jj.max(i)..(jj + BLOCK).min(n)).step_by(4) {
                        if j > i && i + 4 <= n && j + 4 <= n {
                            pair_tile4(
                                values.as_ptr(),
                                n,
                                i,
                                j,
                                scale,
                                layout,
                                &mut masks,
                                &mut output,
                            );
                        } else {
                            for row in i..(i + 4).min(n) {
                                for column in j.max(row + 1)..(j + 4).min(n) {
                                    let (x, y) = match layout {
                                        RssPairLayout::C => {
                                            (values[row * n + column], values[column * n + row])
                                        }
                                        RssPairLayout::F => {
                                            (values[column * n + row], values[row * n + column])
                                        }
                                    };
                                    let value = result.pair(x, y, scale, false);
                                    output.scalar(row * n + column, value);
                                    output.scalar(column * n + row, value);
                                }
                            }
                        }
                    }
                }
            }
        }
        result.raw_finite &= _mm_movemask_pd(masks.raw) == 3;
        result.scaled_finite &= _mm_movemask_pd(masks.scaled) == 3;
        result.symmetric &= _mm_movemask_pd(masks.symmetric) == 3;
        result.repair_finite &= _mm_movemask_pd(masks.repair) == 3;
    }
    result
}

#[cfg(target_arch = "x86_64")]
struct PairMasks {
    raw: std::arch::x86_64::__m128d,
    scaled: std::arch::x86_64::__m128d,
    symmetric: std::arch::x86_64::__m128d,
    repair: std::arch::x86_64::__m128d,
}

// SAFETY: base addresses a live immutable n*n f64 slice; both transposed 4*4
// blocks beginning at (i,j) and (j,i) are complete and in bounds; SSE2 exists.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn pair_tile4(
    base: *const f64,
    n: usize,
    i: usize,
    j: usize,
    scale: f64,
    layout: RssPairLayout,
    masks: &mut PairMasks,
    output: &mut OwnedRepair<'_>,
) {
    use std::arch::x86_64::*;
    let abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7fff_ffff_ffff_ffff));
    let finite_bound = _mm_set1_pd(f64::MAX);
    let scale = _mm_set1_pd(scale);
    let two = _mm_set1_pd(2.);
    let mut tile = [[_mm_setzero_pd(); 2]; 4];
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
            let left = [
                _mm_loadu_pd(base.add(row * n + j)),
                _mm_loadu_pd(base.add(row * n + j + 2)),
            ];
            for (column, (physical_x, physical_y)) in left.into_iter().zip([low, high]).enumerate()
            {
                let (x, y) = match layout {
                    RssPairLayout::C => (physical_x, physical_y),
                    RssPairLayout::F => (physical_y, physical_x),
                };
                masks.raw = _mm_and_pd(
                    masks.raw,
                    _mm_and_pd(
                        _mm_cmple_pd(_mm_and_pd(x, abs_mask), finite_bound),
                        _mm_cmple_pd(_mm_and_pd(y, abs_mask), finite_bound),
                    ),
                );
                let sx = _mm_mul_pd(x, scale);
                let sy = _mm_mul_pd(y, scale);
                masks.scaled = _mm_and_pd(
                    masks.scaled,
                    _mm_and_pd(
                        _mm_cmple_pd(_mm_and_pd(sx, abs_mask), finite_bound),
                        _mm_cmple_pd(_mm_and_pd(sy, abs_mask), finite_bound),
                    ),
                );
                masks.symmetric = _mm_and_pd(masks.symmetric, _mm_cmpeq_pd(sx, sy));
                // Preserve both scaled terms and the parent's ordered average.
                let repaired = _mm_div_pd(_mm_add_pd(sx, sy), two);
                tile[half + offset][column] = repaired;
                output.pair(row * n + j + 2 * column, repaired);
                masks.repair = _mm_and_pd(
                    masks.repair,
                    _mm_cmple_pd(_mm_and_pd(repaired, abs_mask), finite_bound),
                );
            }
        }
    }
    // Store the same repaired bits to the transposed lower tile. Each pair
    // contains two adjacent low-row coordinates in one high output row.
    for half in [0, 2] {
        for columns in [0, 1] {
            let a = tile[half][columns];
            let b = tile[half + 1][columns];
            output.pair((j + 2 * columns) * n + i + half, _mm_unpacklo_pd(a, b));
            output.pair((j + 2 * columns + 1) * n + i + half, _mm_unpackhi_pd(a, b));
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    fn oracle(
        values: &[f64],
        n: usize,
        scale: f64,
        layout: RssPairLayout,
    ) -> (RssPairValidation, Vec<f64>) {
        let mut flags = RssPairValidation::default();
        let mut output = vec![0.; n * n];
        for ii in (0..n).step_by(32) {
            for jj in (ii..n).step_by(32) {
                for i in ii..(ii + 32).min(n) {
                    for j in jj.max(i)..(jj + 32).min(n) {
                        let (x, y) = match layout {
                            RssPairLayout::C => (values[i * n + j], values[j * n + i]),
                            RssPairLayout::F => (values[j * n + i], values[i * n + j]),
                        };
                        let value = flags.pair(x, y, scale, i == j);
                        output[i * n + j] = value;
                        output[j * n + i] = value;
                    }
                }
            }
        }
        (flags, output)
    }
    fn check(values: &[f64], n: usize, scale: f64, layout: RssPairLayout) {
        let original: Vec<_> = values.iter().map(|x| x.to_bits()).collect();
        let (expected, repair) = oracle(values, n, scale, layout);
        let mut output = vec![MaybeUninit::uninit(); n * n];
        let mut writes = vec![0; n * n];
        let actual = contiguous_pairs(
            values,
            n,
            scale,
            layout,
            OwnedRepair {
                values: &mut output,
                writes: Some(&mut writes),
            },
        );
        assert_eq!(actual, expected);
        assert!(writes.iter().all(|&count| count == 1), "n={n}");
        // Raw/scaled errors drop the private output before exposure. All
        // observable repairs, including overflowed repair entries, match bits.
        if expected.raw_finite && expected.scaled_finite {
            let bits: Vec<_> = output
                .iter()
                .map(|x| unsafe { x.assume_init() }.to_bits())
                .collect();
            assert_eq!(
                bits,
                repair.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "n={n},scale={scale}"
            );
        }
        assert_eq!(
            original,
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn owned_tiles_match_scalar_flags_bits_and_write_every_cell_once() {
        for n in [
            4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 34, 35, 63, 64, 65, 66, 67, 129,
        ] {
            let values: Vec<_> = (0..n * n)
                .map(|k| {
                    if k / n == k % n {
                        1.
                    } else {
                        ((k * 17 + k / n * 11) % 67) as f64 / 128. - 0.25
                    }
                })
                .collect();
            for layout in [RssPairLayout::C, RssPairLayout::F] {
                for scale in [
                    0.,
                    -0.,
                    1.,
                    -1.,
                    99.,
                    f64::MAX,
                    f64::MIN_POSITIVE,
                    f64::from_bits(1),
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                ] {
                    check(&values, n, scale, layout);
                }
            }
        }
    }
    #[test]
    fn every_position_retains_extremes_nonfinite_and_diagonal_flags() {
        let n = 9;
        let mut values: Vec<_> = (0..n * n)
            .map(|k| {
                if k / n == k % n {
                    1.
                } else {
                    (k % 11) as f64 / 32.
                }
            })
            .collect();
        for position in 0..n * n {
            let old = values[position];
            for x in [
                0.,
                -0.,
                f64::from_bits(1),
                -f64::from_bits(1),
                f64::MAX,
                f64::MIN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::from_bits(0x7ff8000000000042),
            ] {
                values[position] = x;
                for layout in [RssPairLayout::C, RssPairLayout::F] {
                    for scale in [0., -0., 1., 2., -2., f64::MAX, f64::from_bits(1)] {
                        check(&values, n, scale, layout);
                    }
                }
            }
            values[position] = old;
        }
    }
    #[test]
    fn unsupported_dimensions_do_not_write_any_output() {
        for n in [0, 1, 2, 3] {
            let input = vec![1.; n * n];
            let mut out = vec![MaybeUninit::new(123.); n * n];
            assert!(rss_pair_validation_owned(&input, n, 1., RssPairLayout::C, &mut out).is_none());
            assert!(out.iter().all(|x| unsafe { x.assume_init() } == 123.));
        }
        assert!(rss_pair_validation_owned(
            &[1.; 16],
            4,
            1.,
            RssPairLayout::F,
            &mut [MaybeUninit::uninit(); 15]
        )
        .is_none());
    }
}
