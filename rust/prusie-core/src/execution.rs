//! A private public-fit lifetime shared by optional preparation and the solver.
//! No executor is constructed by the ordinary/default solver entry point.
use crate::{
    input_validation::{self, RssPairLayout, RssPairValidation},
    vector_math,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe},
    thread::{self, JoinHandle, ThreadId},
};

/// A failed scoped task may have written a private, incomplete repair. Its
/// caller must discard that allocation before using the serial parent helper.
pub enum PreparationAttempt<T> {
    NotApplicable,
    Complete(T),
    Aborted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    New,
    Ready,
    Prepared,
    Retired,
    Core,
    Closed,
}

struct JoinablePool {
    pool: Option<ThreadPool>,
    workers: Vec<JoinHandle<()>>,
}

struct BuildFailure {
    message: String,
    join_failed: bool,
}

impl JoinablePool {
    fn build(threads: usize) -> Result<Self, BuildFailure> {
        Self::build_inner(threads, None)
    }

    // The optional failure point is internal and used only by focused tests.
    fn build_inner(threads: usize, fail_after: Option<usize>) -> Result<Self, BuildFailure> {
        let mut workers = Vec::with_capacity(threads);
        let built = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("prusie-kernel-{index}"))
            .spawn_handler(|worker| {
                if fail_after.is_some_and(|limit| workers.len() >= limit) {
                    return Err(std::io::Error::other("injected worker creation failure"));
                }
                let mut builder = thread::Builder::new();
                if let Some(name) = worker.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = worker.stack_size() {
                    builder = builder.stack_size(size);
                }
                workers.push(builder.spawn(move || worker.run())?);
                Ok(())
            })
            .build();
        match built {
            Ok(pool) => Ok(Self {
                pool: Some(pool),
                workers,
            }),
            Err(error) => {
                // Rayon has signalled its partial registry's termination. It
                // does not join our handles; do that before returning inactive.
                let mut partial = Self {
                    pool: None,
                    workers,
                };
                let join_failed = partial.close().is_some();
                Err(BuildFailure {
                    message: format!("Cannot create the per-fit kernel pool: {error}"),
                    join_failed,
                })
            }
        }
    }

    fn get(&self) -> &ThreadPool {
        self.pool.as_ref().expect("live fit-local pool")
    }

    fn close(&mut self) -> Option<String> {
        drop(self.pool.take());
        let mut failed = false;
        for worker in self.workers.drain(..) {
            failed |= worker.join().is_err();
        }
        failed.then(|| "A per-fit kernel worker failed while joining".into())
    }
}

impl Drop for JoinablePool {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Sealed execution resources, never a certificate that an input is valid.
/// Python keeps this owner only in its admitted public call's local lifetime.
pub struct FitExecutor {
    origin: ThreadId,
    phase: Phase,
    requested: Option<usize>,
    dimension: usize,
    fingerprint: Option<u64>,
    pool: Option<JoinablePool>,
    started: bool,
    failure: Option<String>,
    #[cfg(test)]
    task_panic: bool,
}

impl Default for FitExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl FitExecutor {
    /// Inert: no pool, floating inspection, global state or input ownership.
    pub fn new() -> Self {
        Self {
            origin: thread::current().id(),
            phase: Phase::New,
            requested: None,
            dimension: 0,
            fingerprint: None,
            pool: None,
            started: false,
            failure: None,
            #[cfg(test)]
            task_panic: false,
        }
    }

    pub fn has_started(&self) -> bool {
        self.started
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    fn fail(&mut self, message: &str) {
        if self.failure.is_none() {
            self.failure = Some(message.into());
        }
    }

    fn correct_thread(&self) -> bool {
        self.origin == thread::current().id()
    }

    fn retire(&mut self) {
        if let Some(mut pool) = self.pool.take() {
            if let Some(error) = pool.close() {
                self.fail(&error);
            }
        }
        if self.phase != Phase::Closed {
            self.phase = Phase::Retired;
        }
    }

    /// Explicitly finish every worker, even when an earlier join failed.
    /// A failure is retained for the binding/core caller to preserve any primary
    /// exception and to forbid returning a posterior after a worker failure.
    pub fn close(&mut self) {
        self.retire();
        self.phase = Phase::Closed;
    }

    /// Called inside the private helper after extraction and input read borrow.
    /// False never bypasses parent validation; it selects the serial helper.
    pub fn begin_preparation(&mut self, threads: usize, p: usize) -> bool {
        if !self.correct_thread() || self.phase != Phase::New {
            return false;
        }
        self.requested = Some(threads);
        self.dimension = p;
        self.phase = Phase::Retired;
        if !(2..=32).contains(&threads) || !crate::parallel::applicable(p, p) {
            return false;
        }
        let Some(fingerprint) = vector_math::standard_fp_control_fingerprint() else {
            return false;
        };
        match JoinablePool::build(threads) {
            Ok(pool) => {
                self.started = true;
                self.fingerprint = Some(fingerprint);
                self.pool = Some(pool);
                if self.workers_match(fingerprint) {
                    self.phase = Phase::Ready;
                    true
                } else {
                    self.retire();
                    false
                }
            }
            Err(error) => {
                if error.join_failed {
                    self.fail(&error.message);
                }
                false
            }
        }
    }

    fn workers_match(&self, fingerprint: u64) -> bool {
        self.pool.as_ref().is_some_and(|pool| {
            catch_unwind(AssertUnwindSafe(|| {
                pool.get()
                    .broadcast(|_| vector_math::fp_control_fingerprint())
            }))
            .is_ok_and(|values| values.into_iter().all(|value| value == Some(fingerprint)))
        })
    }

    fn preparation_ready(&mut self, p: usize) -> bool {
        if !self.correct_thread() || self.failure.is_some() {
            return false;
        }
        if self.phase != Phase::Ready || p != self.dimension {
            self.retire();
            return false;
        }
        if self.fingerprint != vector_math::standard_fp_control_fingerprint() {
            self.retire();
            return false;
        }
        true
    }

    pub fn exact_rss_proof(
        &mut self,
        values: &[f64],
        p: usize,
        scale: f64,
        layout: RssPairLayout,
    ) -> PreparationAttempt<bool> {
        if !self.preparation_ready(p) {
            return PreparationAttempt::NotApplicable;
        }
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            input_validation::rss_proof_parallel(
                self.pool.as_ref().unwrap().get(),
                values,
                p,
                scale,
                layout,
            )
        }));
        match attempt {
            Ok(Some(valid)) => {
                if valid {
                    self.phase = Phase::Prepared;
                }
                PreparationAttempt::Complete(valid)
            }
            Ok(None) => {
                self.retire();
                PreparationAttempt::NotApplicable
            }
            Err(_) => {
                self.retire();
                PreparationAttempt::Aborted
            }
        }
    }

    pub fn validate_rss_owned(
        &mut self,
        values: &[f64],
        p: usize,
        scale: f64,
        layout: RssPairLayout,
        output: &mut [MaybeUninit<f64>],
    ) -> PreparationAttempt<RssPairValidation> {
        if !self.preparation_ready(p) {
            return PreparationAttempt::NotApplicable;
        }
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            if self.task_panic {
                if let Some(cell) = output.first_mut() {
                    cell.write(0.);
                }
                self.pool
                    .as_ref()
                    .unwrap()
                    .get()
                    .install(|| panic!("injected scoped task panic"));
            }
            input_validation::rss_pairs_parallel(
                self.pool.as_ref().unwrap().get(),
                values,
                p,
                scale,
                layout,
                output,
            )
        }));
        match attempt {
            Ok(Some(flags)) => {
                self.phase = Phase::Prepared;
                PreparationAttempt::Complete(flags)
            }
            Ok(None) => {
                self.retire();
                PreparationAttempt::NotApplicable
            }
            Err(_) => {
                self.retire();
                PreparationAttempt::Aborted
            }
        }
    }

    /// Only ibss's original post-validation pool-selection position calls this.
    pub(crate) fn core_pool(
        &mut self,
        threads: usize,
        p: usize,
    ) -> Result<Option<&ThreadPool>, String> {
        if !self.correct_thread() {
            return Err("A fit executor cannot be reused from another thread".into());
        }
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.requested != Some(threads)
            || !matches!(self.phase, Phase::Prepared | Phase::Retired)
        {
            return Err("Fit executor count or phase does not match this native fit".into());
        }
        if !(1..=32).contains(&threads) {
            return Err("n_threads must be an integer between 1 and 32".into());
        }
        if threads == 1 || !crate::parallel::applicable(p, p) {
            self.retire();
            self.phase = Phase::Core;
            return Ok(None);
        }
        let current = vector_math::standard_fp_control_fingerprint();
        let reuse = self.phase == Phase::Prepared
            && current.is_some()
            && current == self.fingerprint
            && self.workers_match(current.unwrap());
        if !reuse {
            // No overlap: the original generation is fully joined before a
            // parent-time replacement inherits the caller's current controls.
            self.retire();
            if let Some(error) = &self.failure {
                return Err(error.clone());
            }
            match JoinablePool::build(threads) {
                Ok(pool) => {
                    self.started = true;
                    self.pool = Some(pool);
                }
                Err(error) => {
                    if error.join_failed {
                        self.fail(&error.message);
                    }
                    return Err(error.message);
                }
            }
            // Inspect the replacement too. This comparison admits whatever
            // controls the parent-time caller actually selected; it never sets
            // worker controls or silently forces a standard/scalar mode.
            let inherited = vector_math::fp_control_fingerprint();
            if !inherited.is_some_and(|controls| self.workers_match(controls)) {
                self.retire();
                return Err(self.failure.clone().unwrap_or_else(|| {
                    "Cannot verify the per-fit replacement worker controls".into()
                }));
            }
        }
        self.phase = Phase::Core;
        Ok(self.pool.as_ref().map(JoinablePool::get))
    }
}

impl Drop for FitExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(all(test, has_vector_math))]
mod tests {
    use super::*;
    use crate::{diagnostics::FitControl, options::PriorMethod, FitOptions, FitOutput};
    unsafe extern "C" {
        fn fegetround() -> i32;
        fn fesetround(mode: i32) -> i32;
    }

    struct RoundGuard(i32);
    impl RoundGuard {
        fn set(mode: i32) -> Self {
            let saved = unsafe { fegetround() };
            assert_eq!(unsafe { fesetround(mode) }, 0);
            Self(saved)
        }
    }
    impl Drop for RoundGuard {
        fn drop(&mut self) {
            assert_eq!(unsafe { fesetround(self.0) }, 0);
        }
    }

    fn executor(threads: usize) -> FitExecutor {
        assert!(vector_math::standard_fp_control_fingerprint().is_some());
        let mut owner = FitExecutor::new();
        assert!(owner.begin_preparation(threads, 1024));
        owner
    }

    fn identity(n: usize) -> Vec<f64> {
        let mut values = vec![0.; n * n];
        for i in 0..n {
            values[i * n + i] = 1.;
        }
        values
    }

    fn settings(p: usize) -> FitOptions {
        FitOptions {
            l: 2,
            prior_variance: 0.2,
            residual_variance: 1.,
            prior_weights: vec![1. / p as f64; p],
            estimate_prior_variance: false,
            estimate_prior_method: PriorMethod::Optim,
            estimate_residual_variance: false,
            check_null_threshold: 0.,
            max_iter: 3,
            tol: 1e-9,
            check_prior: false,
            prior_tol: 1e-9,
            null_index: None,
        }
    }

    fn same_fit(actual: &FitOutput, parent: &FitOutput) {
        for (a, b) in [
            (&actual.alpha, &parent.alpha),
            (&actual.mu, &parent.mu),
            (&actual.mu2, &parent.mu2),
            (&actual.lbf_variable, &parent.lbf_variable),
            (&actual.lbf, &parent.lbf),
            (&actual.v, &parent.v),
            (&actual.kl, &parent.kl),
            (&actual.elbo, &parent.elbo),
            (&actual.xtxr, &parent.xtxr),
        ] {
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(actual.sigma2.to_bits(), parent.sigma2.to_bits());
        assert_eq!(actual.niter, parent.niter);
        assert_eq!(actual.converged, parent.converged);
        assert_eq!(actual.warnings, parent.warnings);
        assert_eq!(
            actual.diagnostics.exact_activation,
            parent.diagnostics.exact_activation
        );
    }

    #[test]
    fn executor_default_small_and_initial_nonstandard_never_start_workers() {
        for (threads, p) in [(1, 1024), (4, 1023), (0, 1024), (33, 1024)] {
            let mut owner = FitExecutor::new();
            assert!(!owner.has_started());
            assert!(!owner.begin_preparation(threads, p));
            assert!(!owner.has_started());
            owner.close();
            owner.close();
            assert!(owner.failure().is_none());
        }
        let before = vector_math::fp_control_fingerprint();
        {
            let _mode = RoundGuard::set(0x400); // FE_DOWNWARD, current test thread only.
            let controls = vector_math::fp_control_fingerprint();
            let mut owner = FitExecutor::new();
            assert!(!owner.begin_preparation(4, 1024));
            assert!(!owner.has_started());
            assert_eq!(vector_math::fp_control_fingerprint(), controls);
        }
        assert_eq!(vector_math::fp_control_fingerprint(), before);
    }

    #[test]
    fn executor_initial_ftz_daz_exception_masks_and_x87_precision_are_inactive() {
        let mut saved_mxcsr = 0u32;
        let mut saved_x87 = 0u16;
        unsafe {
            std::arch::asm!("stmxcsr [{0}]", in(reg) &mut saved_mxcsr, options(nostack, preserves_flags));
            std::arch::asm!("fnstcw [{0}]", in(reg) &mut saved_x87, options(nostack, preserves_flags));
        }
        for control in [
            saved_mxcsr | 0x8000,
            saved_mxcsr | 0x40,
            (saved_mxcsr & !0x80) & !0x3f,
        ] {
            // Test-only setters. Clear pending SSE status before unmasking;
            // restore controls before assertions or any floating arithmetic.
            unsafe {
                std::arch::asm!("ldmxcsr [{0}]", in(reg) &control, options(nostack, preserves_flags));
            }
            let mut owner = FitExecutor::new();
            let admitted = owner.begin_preparation(2, 1024);
            let started = owner.has_started();
            owner.close();
            unsafe {
                std::arch::asm!("ldmxcsr [{0}]", in(reg) &saved_mxcsr, options(nostack, preserves_flags));
            }
            assert!(!admitted && !started);
            assert!(owner.failure().is_none());
        }
        let double_precision = (saved_x87 & !0x0300) | 0x0200;
        unsafe {
            std::arch::asm!("fldcw [{0}]", in(reg) &double_precision, options(nostack, preserves_flags));
        }
        let mut owner = FitExecutor::new();
        let admitted = owner.begin_preparation(2, 1024);
        let started = owner.has_started();
        owner.close();
        unsafe {
            std::arch::asm!("fldcw [{0}]", in(reg) &saved_x87, options(nostack, preserves_flags));
        }
        assert!(!admitted && !started);
        assert!(owner.failure().is_none());
    }

    #[test]
    fn executor_same_controls_reuse_workers_and_changed_controls_replace_after_join() {
        let values = identity(1024);
        for changed in [false, true] {
            let mut owner = executor(2);
            assert!(matches!(
                owner.exact_rss_proof(&values, 1024, 99., RssPairLayout::C),
                PreparationAttempt::Complete(true)
            ));
            let old = owner
                .pool
                .as_ref()
                .unwrap()
                .get()
                .broadcast(|_| thread::current().id());
            let _guard = changed.then(|| RoundGuard::set(0x800)); // FE_UPWARD
            let fingerprint = vector_math::fp_control_fingerprint();
            let pool = owner.core_pool(2, 1024).unwrap().unwrap();
            let new = pool.broadcast(|_| thread::current().id());
            assert_eq!(
                pool.broadcast(|_| vector_math::fp_control_fingerprint()),
                vec![fingerprint; 2]
            );
            if changed {
                assert!(old.iter().all(|id| !new.contains(id)));
            } else {
                assert_eq!(old, new);
            }
            owner.close();
            assert!(owner.pool.is_none());
            assert!(owner.failure().is_none());
        }
    }

    #[test]
    fn executor_worker_control_mismatch_retires_before_core_replacement() {
        let values = identity(1024);
        let mut owner = executor(2);
        assert!(matches!(
            owner.exact_rss_proof(&values, 1024, 99., RssPairLayout::C),
            PreparationAttempt::Complete(true)
        ));
        let old = owner
            .pool
            .as_ref()
            .unwrap()
            .get()
            .broadcast(|_| thread::current().id());
        owner.pool.as_ref().unwrap().get().broadcast(|context| {
            if context.index() == 0 {
                // Test-only worker-local FTZ change. This worker is retired;
                // production contains no floating-control setter.
                let mut value = 0u32;
                unsafe {
                    std::arch::asm!("stmxcsr [{0}]", in(reg) &mut value, options(nostack, preserves_flags));
                    value |= 0x8000;
                    std::arch::asm!("ldmxcsr [{0}]", in(reg) &value, options(nostack, preserves_flags));
                }
            }
        });
        let pool = owner.core_pool(2, 1024).unwrap().unwrap();
        let new = pool.broadcast(|_| thread::current().id());
        assert!(old.iter().all(|id| !new.contains(id)));
        assert!(pool
            .broadcast(|_| vector_math::standard_fp_control_fingerprint())
            .iter()
            .all(Option::is_some));
        owner.close();
        assert!(owner.failure().is_none());
    }

    #[test]
    fn executor_changed_controls_before_owned_write_return_parent_without_stores() {
        let mut owner = executor(2);
        let mut values = identity(1024);
        values[1] = 0.25;
        assert!(matches!(
            owner.exact_rss_proof(&values, 1024, 99., RssPairLayout::C),
            PreparationAttempt::Complete(false)
        ));
        let _guard = RoundGuard::set(0x400);
        let mut output = vec![MaybeUninit::new(0.375); values.len()];
        assert!(matches!(
            owner.validate_rss_owned(&values, 1024, 99., RssPairLayout::C, &mut output),
            PreparationAttempt::NotApplicable
        ));
        assert!(output
            .iter()
            .all(|cell| unsafe { cell.assume_init() } == 0.375));
        assert_eq!(owner.phase, Phase::Retired);
        assert!(owner.pool.is_none());
        assert!(owner.failure().is_none());
        assert!(owner.core_pool(2, 1024).unwrap().is_some());
        owner.close();
        assert!(owner.failure().is_none());
    }

    #[test]
    fn executor_changed_controls_full_fit_keeps_parent_mode_bits() {
        let p = 1024;
        let values = identity(p);
        let xty: Vec<_> = (0..p).map(|j| (j % 37) as f64 / 64. - 0.25).collect();
        let mut owner = executor(2);
        assert!(matches!(
            owner.exact_rss_proof(&values, p, 99., RssPairLayout::C),
            PreparationAttempt::Complete(true)
        ));
        let _guard = RoundGuard::set(0x400);
        let control = FitControl {
            n_threads: 2,
            ..FitControl::default()
        };
        let expected =
            crate::fit_controlled(&values, &xty, 99., 100., &settings(p), true, None, &control)
                .unwrap();
        let actual = crate::fit_controlled_with_executor(
            &values,
            &xty,
            99.,
            100.,
            &settings(p),
            true,
            None,
            &control,
            &mut owner,
        )
        .unwrap();
        same_fit(&actual, &expected);
        assert_eq!(owner.phase, Phase::Closed);
        assert!(owner.pool.is_none());
        assert!(owner.failure().is_none());
    }

    #[test]
    fn executor_partial_spawn_and_scoped_panic_leave_no_live_partial_owner() {
        for limit in [0, 1, 2, 3] {
            let failure = match JoinablePool::build_inner(4, Some(limit)) {
                Err(error) => error,
                Ok(_) => panic!("failure injection did not fire"),
            };
            assert!(!failure.join_failed);
            assert!(failure
                .message
                .contains("Cannot create the per-fit kernel pool"));
        }
        let mut owner = executor(2);
        owner.task_panic = true;
        let values = identity(1024);
        let mut output = vec![MaybeUninit::uninit(); values.len()];
        assert!(matches!(
            owner.validate_rss_owned(&values, 1024, 99., RssPairLayout::C, &mut output),
            PreparationAttempt::Aborted
        ));
        assert!(owner.pool.is_none());
        assert_eq!(owner.phase, Phase::Retired);
        assert!(owner.has_started());
        assert!(owner.failure().is_none());
        drop(output); // partial storage is deliberately never exposed as f64
        assert!(owner.core_pool(2, 1024).unwrap().is_some());
        owner.close();
        assert!(owner.failure().is_none());
    }

    #[test]
    fn executor_private_misuse_and_join_failure_cannot_return_success() {
        let mut owner = executor(2);
        owner.phase = Phase::Prepared;
        assert!(owner.core_pool(4, 1024).is_err());
        owner.close();
        assert!(owner.pool.is_none());
        assert!(owner.core_pool(2, 1024).is_err());
        let mut owner = executor(2);
        owner
            .pool
            .as_mut()
            .unwrap()
            .workers
            .push(thread::spawn(|| panic!("injected joined thread failure")));
        owner.close();
        assert!(owner.pool.is_none());
        assert!(owner.failure().is_some());
        owner.close();
        assert!(owner.failure().is_some());
        let mut owner = executor(2);
        let moved = thread::spawn(move || {
            assert!(!owner.begin_preparation(2, 1024));
            assert!(owner.core_pool(2, 1024).is_err());
            owner.close();
            assert!(owner.failure().is_none());
        });
        moved.join().unwrap();
    }

    #[test]
    fn executor_owned_preparation_and_full_fit_keep_same_count_parent_bits() {
        let p = 1024;
        let raw: Vec<f64> = (0..p * p)
            .map(|k| {
                if k / p == k % p {
                    1.
                } else {
                    0.01 + (k % p < k / p) as u8 as f64 * 0.001
                }
            })
            .collect();
        let before: Vec<_> = raw.iter().map(|x| x.to_bits()).collect();
        let xty: Vec<_> = (0..p).map(|j| (j % 37) as f64 / 16. - 1.).collect();
        for threads in [2, 4] {
            let mut owner = executor(threads);
            assert!(matches!(
                owner.exact_rss_proof(&raw, p, 99., RssPairLayout::C),
                PreparationAttempt::Complete(false)
            ));
            let mut parent = vec![MaybeUninit::uninit(); p * p];
            let expected = input_validation::rss_pair_validation_owned(
                &raw,
                p,
                99.,
                RssPairLayout::C,
                &mut parent,
            )
            .unwrap();
            let mut actual = vec![MaybeUninit::uninit(); p * p];
            let flags = match owner.validate_rss_owned(&raw, p, 99., RssPairLayout::C, &mut actual)
            {
                PreparationAttempt::Complete(flags) => flags,
                _ => panic!("no complete repair"),
            };
            assert_eq!(flags, expected);
            let parent: Vec<_> = parent
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();
            let actual: Vec<_> = actual
                .into_iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                parent.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let control = FitControl {
                n_threads: threads,
                ..FitControl::default()
            };
            let expected =
                crate::fit_controlled(&parent, &xty, 99., 100., &settings(p), true, None, &control)
                    .unwrap();
            let result = crate::fit_controlled_with_executor(
                &actual,
                &xty,
                99.,
                100.,
                &settings(p),
                true,
                None,
                &control,
                &mut owner,
            )
            .unwrap();
            same_fit(&result, &expected);
            assert_eq!(owner.phase, Phase::Closed);
            assert!(owner.pool.is_none());
            assert!(owner.failure().is_none());
        }
        assert_eq!(raw.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), before);
    }

    #[test]
    fn executor_core_validation_error_preserves_primary_and_joins() {
        let mut owner = executor(2);
        owner
            .pool
            .as_mut()
            .unwrap()
            .workers
            .push(thread::spawn(|| panic!("injected cleanup failure")));
        let result = crate::fit_controlled_with_executor(
            &[],
            &[],
            1.,
            2.,
            &settings(1),
            true,
            None,
            &FitControl::default(),
            &mut owner,
        );
        assert_eq!(
            result.err().unwrap(),
            "Invalid sufficient-statistic dimensions or number of effects"
        );
        assert_eq!(owner.phase, Phase::Closed);
        assert!(owner.pool.is_none());
        assert!(owner.failure().is_some());
    }

    #[test]
    fn executor_join_failure_discards_an_otherwise_successful_posterior() {
        let p = 1024;
        let values = identity(p);
        let mut owner = executor(2);
        assert!(matches!(
            owner.exact_rss_proof(&values, p, 99., RssPairLayout::C),
            PreparationAttempt::Complete(true)
        ));
        owner
            .pool
            .as_mut()
            .unwrap()
            .workers
            .push(thread::spawn(|| panic!("injected final join failure")));
        let control = FitControl {
            n_threads: 2,
            ..FitControl::default()
        };
        let result = crate::fit_controlled_with_executor(
            &values,
            &vec![0.125; p],
            99.,
            100.,
            &settings(p),
            true,
            None,
            &control,
            &mut owner,
        );
        assert_eq!(
            result.err().unwrap(),
            "A per-fit kernel worker failed while joining"
        );
        assert_eq!(owner.phase, Phase::Closed);
        assert!(owner.pool.is_none());
    }
}
