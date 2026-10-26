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
trait RepairWriter {
    fn scalar(&mut self, index: usize, value: f64);
    #[cfg(target_arch = "x86_64")]
    unsafe fn pair(&mut self, index: usize, value: std::arch::x86_64::__m128d);
}
impl RepairWriter for OwnedRepair<'_> {
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

// Each rank owns one disjoint set of *cells*, not a contiguous output slice.
// No mutable f64/slice reference is formed through this raw capability.
struct RankRepair<'a> {
    base: *mut MaybeUninit<f64>,
    n: usize,
    rank: usize,
    threads: usize,
    _borrow: std::marker::PhantomData<&'a mut [MaybeUninit<f64>]>,
    #[cfg(test)]
    writes: Option<&'a [std::sync::atomic::AtomicUsize]>,
}
// SAFETY: constructor validates the full unique output borrow and makes exactly
// one non-cloneable capability per rank. Rank r alone writes unordered pairs
// whose lower-coordinate macroblock row is congruent to r modulo threads. Each
// capability moves to one blocking job; all jobs finish before borrow release.
unsafe impl Send for RankRepair<'_> {}
impl RankRepair<'_> {
    fn owns(&self, index: usize) -> bool {
        index < self.n * self.n
            && ((index / self.n).min(index % self.n) / 32) % self.threads == self.rank
    }
}
impl RepairWriter for RankRepair<'_> {
    fn scalar(&mut self, index: usize, value: f64) {
        debug_assert!(self.owns(index));
        // SAFETY: the static upper-block traversal proves this bounded cell is
        // owned only by this rank, while the unique output borrow remains live.
        unsafe {
            self.base.add(index).write(MaybeUninit::new(value));
        }
        #[cfg(test)]
        if let Some(writes) = self.writes {
            writes[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn pair(&mut self, index: usize, value: std::arch::x86_64::__m128d) {
        debug_assert!(self.owns(index) && self.owns(index + 1));
        std::arch::x86_64::_mm_storeu_pd(self.base.add(index).cast::<f64>(), value);
        #[cfg(test)]
        if let Some(writes) = self.writes {
            writes[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            writes[index + 1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub(crate) fn rss_pairs_parallel(
    pool: &rayon::ThreadPool,
    values: &[f64],
    n: usize,
    scale: f64,
    layout: RssPairLayout,
    output: &mut [MaybeUninit<f64>],
) -> Option<RssPairValidation> {
    rss_pairs_parallel_impl(
        pool,
        values,
        n,
        scale,
        layout,
        output,
        #[cfg(test)]
        None,
        #[cfg(test)]
        None,
    )
}

// Tests add write-count and panic-injection arguments to the explicit repair inputs.
#[cfg_attr(test, allow(clippy::too_many_arguments))]
fn rss_pairs_parallel_impl(
    pool: &rayon::ThreadPool,
    values: &[f64],
    n: usize,
    scale: f64,
    layout: RssPairLayout,
    output: &mut [MaybeUninit<f64>],
    #[cfg(test)] writes: Option<&[std::sync::atomic::AtomicUsize]>,
    #[cfg(test)] panic_rank: Option<usize>,
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
        use rayon::prelude::*;
        let threads = pool.current_num_threads().min(n.div_ceil(32).max(1));
        let base = output.as_mut_ptr();
        let jobs: Vec<_> = (0..threads)
            .map(|rank| RankRepair {
                base,
                n,
                rank,
                threads,
                _borrow: std::marker::PhantomData,
                #[cfg(test)]
                writes,
            })
            .collect();
        let flags: Vec<_> = pool.install(|| {
            jobs.into_par_iter()
                .map(|output| {
                    #[cfg(test)]
                    let mut output = output;
                    let rank = output.rank;
                    #[cfg(test)]
                    if panic_rank == Some(rank) {
                        output.scalar(rank * 32 * n + rank * 32, 0.);
                        panic!("injected scoped repair task failure");
                    }
                    contiguous_pairs_rank(values, n, scale, layout, output, rank, threads)
                })
                .collect()
        });
        Some(
            flags
                .into_iter()
                .fold(RssPairValidation::default(), |mut total, rank| {
                    total.raw_finite &= rank.raw_finite;
                    total.raw_nonnegative &= rank.raw_nonnegative;
                    total.scaled_finite &= rank.scaled_finite;
                    total.scaled_nonnegative &= rank.scaled_nonnegative;
                    total.symmetric &= rank.symmetric;
                    total.repair_finite &= rank.repair_finite;
                    total.repair_nonnegative &= rank.repair_nonnegative;
                    total
                }),
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (pool, scale, layout);
        None
    }
}

#[cfg(target_arch = "x86_64")]
fn contiguous_pairs_rank(
    values: &[f64],
    n: usize,
    scale: f64,
    layout: RssPairLayout,
    mut output: RankRepair<'_>,
    rank: usize,
    threads: usize,
) -> RssPairValidation {
    use std::arch::x86_64::*;
    let mut result = RssPairValidation::default();
    for i in (rank * 32..n)
        .step_by(threads * 32)
        .flat_map(|start| start..(start + 32).min(n))
    {
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
        for ii in (rank * BLOCK..n).step_by(BLOCK * threads) {
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
// Tile coordinates, arithmetic inputs and disjoint repair state remain explicit.
#[allow(clippy::too_many_arguments)]
unsafe fn pair_tile4(
    base: *const f64,
    n: usize,
    i: usize,
    j: usize,
    scale: f64,
    layout: RssPairLayout,
    masks: &mut PairMasks,
    output: &mut impl RepairWriter,
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

/// Destructively validate and repair one uniquely borrowed C-order RSS buffer.
/// Every original unordered pair is loaded before either result is stored.
/// The caller must establish a finite scaled-asymmetry witness before choosing
/// this function: symmetric inputs must retain their original unscaled bytes.
/// This function independently computes every flag and does not trust a prior
/// validation result. Invalid dimensions return None before any write.
pub fn rss_pair_validation_in_place(
    values: &mut [f64],
    n: usize,
    scale: f64,
) -> Option<RssPairValidation> {
    if n == 0
        || n.checked_mul(n) != Some(values.len())
        || values.len() > isize::MAX as usize / std::mem::size_of::<f64>()
    {
        return None;
    }
    let base = values.as_mut_ptr();
    let mut result = RssPairValidation::default();
    // SAFETY: base derives from the only mutable slice, dimensions are checked,
    // and no shared slice or additional mutable reference is constructed.
    unsafe {
        for i in 0..n {
            let raw = *base.add(i * n + i);
            *base.add(i * n + i) = result.pair(raw, raw, scale, true);
        }
        #[cfg(target_arch = "x86_64")]
        let mut masks = {
            use std::arch::x86_64::*;
            PairMasks {
                raw: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
                scaled: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
                symmetric: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
                repair: _mm_castsi128_pd(_mm_set1_epi64x(-1)),
            }
        };
        for ii in (0..n).step_by(32) {
            for jj in (ii..n).step_by(32) {
                for i in (ii..(ii + 32).min(n)).step_by(4) {
                    for j in (jj.max(i)..(jj + 32).min(n)).step_by(4) {
                        #[cfg(target_arch = "x86_64")]
                        if j > i && i + 4 <= n && j + 4 <= n {
                            pair_tile4_in_place(base, n, i, j, scale, &mut masks);
                            continue;
                        }
                        for row in i..(i + 4).min(n) {
                            for column in j.max(row + 1)..(j + 4).min(n) {
                                let x = *base.add(row * n + column);
                                let y = *base.add(column * n + row);
                                let value = result.pair(x, y, scale, false);
                                *base.add(row * n + column) = value;
                                *base.add(column * n + row) = value;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            result.raw_finite &= _mm_movemask_pd(masks.raw) == 3;
            result.scaled_finite &= _mm_movemask_pd(masks.scaled) == 3;
            result.symmetric &= _mm_movemask_pd(masks.symmetric) == 3;
            result.repair_finite &= _mm_movemask_pd(masks.repair) == 3;
        }
    }
    Some(result)
}

// SAFETY: base addresses a live uniquely borrowed n*n f64 buffer; both transposed 4*4
// blocks beginning at (i,j) and (j,i) are complete and in bounds; SSE2 exists.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn pair_tile4_in_place(
    base: *mut f64,
    n: usize,
    i: usize,
    j: usize,
    scale: f64,
    masks: &mut PairMasks,
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
                let (x, y) = (physical_x, physical_y);
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
                masks.repair = _mm_and_pd(
                    masks.repair,
                    _mm_cmple_pd(_mm_and_pd(repaired, abs_mask), finite_bound),
                );
            }
        }
    }
    // All original terms in BOTH orientations have been loaded before the
    // first write. Complete paired tiles cannot overlap another visited pair.
    for (row, values) in tile.iter().enumerate() {
        _mm_storeu_pd(base.add((i + row) * n + j), values[0]);
        _mm_storeu_pd(base.add((i + row) * n + j + 2), values[1]);
    }
    // Store the same repaired bits to the transposed lower tile. Each pair
    // contains two adjacent low-row coordinates in one high output row.
    for half in [0, 2] {
        for columns in [0, 1] {
            let a = tile[half][columns];
            let b = tile[half + 1][columns];
            _mm_storeu_pd(
                base.add((j + 2 * columns) * n + i + half),
                _mm_unpacklo_pd(a, b),
            );
            _mm_storeu_pd(
                base.add((j + 2 * columns + 1) * n + i + half),
                _mm_unpackhi_pd(a, b),
            );
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn parallel_owned_tiles_match_parent_bits_flags_and_unique_writes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for threads in [2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for n in [4, 5, 31, 32, 33, 34, 35, 63, 64, 65, 66, 67, 129] {
                let mut values: Vec<f64> = (0..n * n)
                    .map(|k| {
                        if k / n == k % n {
                            1.
                        } else {
                            ((k * 17 + k / n * 11) % 67) as f64 / 128. - 0.25
                        }
                    })
                    .collect();
                for (index, value) in [
                    (1, -0.),
                    (n, 0.),
                    (n + 2, f64::from_bits(1)),
                    (2 * n + 1, -f64::from_bits(1)),
                    (n * n - 2, f64::MAX),
                ] {
                    values[index] = value;
                }
                let before: Vec<_> = values.iter().map(|x| x.to_bits()).collect();
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
                        f64::NAN,
                    ] {
                        let mut parent = vec![MaybeUninit::uninit(); n * n];
                        let expected =
                            rss_pair_validation_owned(&values, n, scale, layout, &mut parent)
                                .unwrap();
                        let mut actual = vec![MaybeUninit::uninit(); n * n];
                        let writes: Vec<_> = (0..n * n).map(|_| AtomicUsize::new(0)).collect();
                        let flags = rss_pairs_parallel_impl(
                            &pool,
                            &values,
                            n,
                            scale,
                            layout,
                            &mut actual,
                            Some(&writes),
                            None,
                        )
                        .unwrap();
                        assert_eq!(flags, expected, "n={n}, threads={threads}, scale={scale}");
                        assert!(
                            writes
                                .iter()
                                .all(|count| count.load(Ordering::Relaxed) == 1),
                            "n={n}, threads={threads}"
                        );
                        // SAFETY: successful complete traversals initialize every
                        // cell; the independent counts additionally verify this.
                        assert_eq!(
                            actual
                                .iter()
                                .map(|x| unsafe { x.assume_init() }.to_bits())
                                .collect::<Vec<_>>(),
                            parent
                                .iter()
                                .map(|x| unsafe { x.assume_init() }.to_bits())
                                .collect::<Vec<_>>(),
                            "n={n}, threads={threads}, scale={scale}"
                        );
                    }
                }
                assert_eq!(
                    values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    before
                );
            }
        }
    }

    #[test]
    fn parallel_owned_nonfinite_priorities_and_dimensions_match_parent() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let n = 67;
        let mut values = vec![0.125; n * n];
        for i in 0..n {
            values[i * n + i] = 1.;
        }
        for index in [0, 1, 32 * n + 33, 33 * n + 32, n * n - 1] {
            let old = values[index];
            for bad in [
                -1.,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::MAX,
                f64::from_bits(0x7ff0_0000_0000_0001),
            ] {
                values[index] = bad;
                for scale in [99., f64::MAX, f64::MIN_POSITIVE] {
                    for layout in [RssPairLayout::C, RssPairLayout::F] {
                        let mut parent = vec![MaybeUninit::uninit(); n * n];
                        let expected =
                            rss_pair_validation_owned(&values, n, scale, layout, &mut parent)
                                .unwrap();
                        let mut actual = vec![MaybeUninit::uninit(); n * n];
                        let flags =
                            rss_pairs_parallel(&pool, &values, n, scale, layout, &mut actual)
                                .unwrap();
                        assert_eq!(flags, expected);
                        assert_eq!(
                            actual
                                .iter()
                                .map(|x| unsafe { x.assume_init() }.to_bits())
                                .collect::<Vec<_>>(),
                            parent
                                .iter()
                                .map(|x| unsafe { x.assume_init() }.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
            values[index] = old;
        }
        let mut untouched = [MaybeUninit::new(0.375); 15];
        assert!(
            rss_pairs_parallel(&pool, &[1.; 16], 4, 99., RssPairLayout::C, &mut untouched)
                .is_none()
        );
        assert!(untouched
            .iter()
            .all(|value| unsafe { value.assume_init() } == 0.375));
    }

    #[test]
    fn parallel_owned_scoped_panic_finishes_writers_without_exposing_output() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = 67;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let values = vec![0.25; n * n];
        let writes: Vec<_> = (0..n * n).map(|_| AtomicUsize::new(0)).collect();
        let mut output = vec![MaybeUninit::uninit(); n * n];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rss_pairs_parallel_impl(
                &pool,
                &values,
                n,
                99.,
                RssPairLayout::C,
                &mut output,
                Some(&writes),
                Some(1),
            )
        }));
        assert!(result.is_err());
        assert!(writes
            .iter()
            .all(|count| count.load(Ordering::Relaxed) <= 1));
        assert!(
            writes
                .iter()
                .map(|count| count.load(Ordering::Relaxed))
                .sum::<usize>()
                < n * n
        );
        // Drop incomplete storage without f64 exposure; the blocking collect
        // joined every scoped writer before unwinding reached this caller.
        drop(output);
        assert_eq!(pool.broadcast(|_| 1).len(), 4);
    }
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
    fn destructive_tiles_match_independent_ordered_oracle_at_all_boundaries() {
        for n in [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 67, 129,
        ] {
            let mut values: Vec<f64> = (0..n * n)
                .map(|k| {
                    if k / n == k % n {
                        1.
                    } else {
                        ((k * 17 + k / n * 11) % 67) as f64 / 128. - 0.25
                    }
                })
                .collect();
            for index in [0, n * n / 2, n * n - 1] {
                let old = values[index];
                for raw in [
                    old,
                    -0.,
                    0.,
                    f64::from_bits(1),
                    f64::MAX,
                    f64::MIN,
                    f64::INFINITY,
                    f64::NAN,
                ] {
                    values[index] = raw;
                    for scale in [0., -0., 1., -1., 99., f64::MAX, f64::from_bits(1)] {
                        let (expected, repair) = oracle(&values, n, scale, RssPairLayout::C);
                        let mut actual = values.clone();
                        let flags = rss_pair_validation_in_place(&mut actual, n, scale).unwrap();
                        assert_eq!(flags, expected, "n={n},index={index},scale={scale}");
                        if expected.raw_finite && expected.scaled_finite {
                            assert_eq!(
                                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                repair.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "n={n},index={index},scale={scale}"
                            );
                        }
                    }
                }
                values[index] = old;
            }
        }
        let mut invalid = [1.; 15];
        assert!(rss_pair_validation_in_place(&mut invalid, 4, 99.).is_none());
        assert_eq!(invalid, [1.; 15]);
        assert!(rss_pair_validation_in_place(&mut [], 0, 99.).is_none());
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
