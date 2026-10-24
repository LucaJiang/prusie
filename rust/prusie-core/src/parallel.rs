//! Fit-local kernel execution without a process-global Rayon pool.
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};

/// A priori work cutoff: at least 1024 output rows and 1024² matrix entries.
/// It is a scheduling heuristic, not a fitted threshold or statistical filter.
pub(crate) fn applicable(rows: usize, columns: usize) -> bool {
    rows >= 1024
        && rows
            .checked_mul(columns)
            .is_some_and(|entries| entries >= 1024 * 1024)
}

pub(crate) fn pool(threads: usize, dimension: usize) -> Result<Option<ThreadPool>, String> {
    if !(1..=32).contains(&threads) {
        return Err("n_threads must be an integer between 1 and 32".into());
    }
    if threads == 1 || !applicable(dimension, dimension) {
        return Ok(None);
    }
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("prusie-kernel-{index}"))
        .build()
        .map(Some)
        .map_err(|error| format!("Cannot create the per-fit kernel pool: {error}"))
}

/// Fixed disjoint row chunks; scheduling never changes a row's arithmetic.
pub(crate) fn rows(
    pool: &ThreadPool,
    output: &mut [f64],
    compute: impl Fn(usize, &mut [f64]) + Send + Sync,
) {
    // Round chunk lengths to eight f64 values. Ownership remains disjoint,
    // including the final tail; no special alignment of caller data is assumed.
    let chunk = output
        .len()
        .div_ceil(pool.current_num_threads())
        .div_ceil(8)
        * 8;
    pool.install(|| {
        output
            .par_chunks_mut(chunk.max(1))
            .enumerate()
            .for_each(|(index, values)| compute(index * chunk, values));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_small_work_do_not_create_a_pool() {
        assert!(pool(1, 16384).unwrap().is_none());
        assert!(pool(4, 1023).unwrap().is_none());
        assert!(pool(0, 1024).is_err());
        assert!(pool(33, 1024).is_err());
    }
}
