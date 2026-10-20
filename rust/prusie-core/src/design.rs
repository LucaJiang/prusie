//! Matrix-free individual-level regression with dense or canonical CSC designs.
//! Includes susieR's trend-filtering operator (BSD-3-Clause; see THIRD_PARTY_NOTICES.md).
//!
//! Centering and scaling are applied as linear operators. A null column is
//! represented virtually, so neither sparse input nor XᵀX is ever densified.
use crate::{
    diagnostics::FitControl, ibss::fit_operator, input_validation, numerical::sum,
    operator::GramOperator, FitOptions, FitOutput,
};

enum Storage<'a> {
    Dense {
        values: &'a [f64],
        column_major: bool,
    },
    Csc {
        values: &'a [f64],
        indices: &'a [i64],
        indptr: &'a [i64],
    },
}

struct Design<'a> {
    storage: Storage<'a>,
    nrow: usize,
    ncol: usize,
    means: &'a [f64],
    scales: &'a [f64],
    diagonal: Vec<f64>,
    fitted: Vec<f64>,
}

impl Design<'_> {
    fn raw_value(&self, row: usize, column: usize) -> f64 {
        match self.storage {
            Storage::Dense {
                values,
                column_major,
            } => {
                values[if column_major {
                    column * self.nrow + row
                } else {
                    row * self.ncol + column
                }]
            }
            Storage::Csc { .. } => unreachable!("raw_value only used for dense storage"),
        }
    }

    fn initialize_diagonal(&mut self) {
        for j in 0..self.ncol {
            let center = self.means[j];
            let norm2 = match self.storage {
                Storage::Dense { .. } => {
                    sum((0..self.nrow).map(|i| (self.raw_value(i, j) - center).powi(2)))
                }
                Storage::Csc { values, indptr, .. } => {
                    let start = indptr[j] as usize;
                    let end = indptr[j + 1] as usize;
                    sum(values[start..end].iter().map(|&x| (x - center).powi(2)))
                        + (self.nrow - (end - start)) as f64 * center.powi(2)
                }
            };
            self.diagonal[j] = norm2 / self.scales[j].powi(2);
        }
    }
}

impl GramOperator for Design<'_> {
    fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    fn apply(&mut self, vector: &[f64], output: &mut [f64]) {
        let shift = sum((0..self.ncol).map(|j| self.means[j] * vector[j] / self.scales[j]));
        self.fitted.fill(-shift);
        for (j, &coefficient) in vector[..self.ncol].iter().enumerate() {
            let coefficient = coefficient / self.scales[j];
            if coefficient == 0. {
                continue;
            }
            match self.storage {
                Storage::Dense {
                    values,
                    column_major,
                } => {
                    for (i, result) in self.fitted.iter_mut().enumerate() {
                        let index = if column_major {
                            j * self.nrow + i
                        } else {
                            i * self.ncol + j
                        };
                        *result += values[index] * coefficient;
                    }
                }
                Storage::Csc {
                    values,
                    indices,
                    indptr,
                } => {
                    for k in indptr[j] as usize..indptr[j + 1] as usize {
                        self.fitted[indices[k] as usize] += values[k] * coefficient;
                    }
                }
            }
        }
        let fitted_sum = sum(self.fitted.iter().copied());
        output.fill(0.);
        for (j, result) in output[..self.ncol].iter_mut().enumerate() {
            let product = match self.storage {
                Storage::Dense { .. } => {
                    sum((0..self.nrow).map(|i| self.raw_value(i, j) * self.fitted[i]))
                }
                Storage::Csc {
                    values,
                    indices,
                    indptr,
                } => sum((indptr[j] as usize..indptr[j + 1] as usize)
                    .map(|k| values[k] * self.fitted[indices[k] as usize])),
            };
            *result = (product - self.means[j] * fitted_sum) / self.scales[j];
        }
    }
}

fn validate_design(
    nrow: usize,
    ncol: usize,
    means: &[f64],
    scales: &[f64],
    p: usize,
    options: &FitOptions,
) -> Result<(), String> {
    let null = p == ncol + 1 && options.null_index == Some(ncol);
    if nrow <= 1
        || ncol == 0
        || (p != ncol && !null)
        || means.len() != ncol
        || scales.len() != ncol
        || !input_validation::all_finite(means)
        || scales.iter().any(|&s| !s.is_finite() || s <= 0.)
    {
        return Err("Invalid design dimensions, centering, scaling, or virtual null column".into());
    }
    Ok(())
}

/// Fit a dense Gram factor in working centered/scaled coordinates.
/// `means` and `scales` describe the raw `ncol` columns. If Xty has one extra
/// entry and `null_index == ncol`, it denotes a virtual all-zero null column.
/// The likelihood sample size `n` may differ from the factor's `nrow`, as when
/// reference genotypes provide an RSS Gram factor for a larger association study.
#[allow(clippy::too_many_arguments)]
pub fn fit_design_dense(
    values: &[f64],
    nrow: usize,
    ncol: usize,
    column_major: bool,
    means: &[f64],
    scales: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    control: &FitControl,
) -> Result<FitOutput, String> {
    validate_design(nrow, ncol, means, scales, xty.len(), options)?;
    if nrow.checked_mul(ncol) != Some(values.len()) || !input_validation::all_finite(values) {
        return Err("Dense design must have nrow × ncol finite entries".into());
    }
    let mut design = Design {
        storage: Storage::Dense {
            values,
            column_major,
        },
        nrow,
        ncol,
        means,
        scales,
        diagonal: vec![0.; xty.len()],
        fitted: vec![0.; nrow],
    };
    design.initialize_diagonal();
    fit_operator(&mut design, xty, yty, n, options, control)
}

/// Fit a canonical CSC design without constructing a dense design or Gram
/// matrix. Each column's row indices must be strictly increasing and in range.
/// The likelihood sample size `n` is independent of the number of factor rows.
#[allow(clippy::too_many_arguments)]
pub fn fit_design_csc(
    values: &[f64],
    indices: &[i64],
    indptr: &[i64],
    nrow: usize,
    ncol: usize,
    means: &[f64],
    scales: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    control: &FitControl,
) -> Result<FitOutput, String> {
    validate_design(nrow, ncol, means, scales, xty.len(), options)?;
    if values.len() != indices.len()
        || indptr.len() != ncol + 1
        || indptr[0] != 0
        || indptr[ncol] < 0
        || indptr[ncol] as usize != values.len()
        || !input_validation::all_finite(values)
    {
        return Err("Invalid canonical CSC arrays".into());
    }
    for j in 0..ncol {
        let start = indptr[j];
        let end = indptr[j + 1];
        if start < 0 || end < start || end as usize > values.len() {
            return Err("CSC column pointers must be ordered and in range".into());
        }
        let rows = &indices[start as usize..end as usize];
        if rows.iter().any(|&i| i < 0 || i as usize >= nrow)
            || rows.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("CSC rows must be strictly increasing and in range in each column".into());
        }
    }
    let mut design = Design {
        storage: Storage::Csc {
            values,
            indices,
            indptr,
        },
        nrow,
        ncol,
        means,
        scales,
        diagonal: vec![0.; xty.len()],
        fitted: vec![0.; nrow],
    };
    design.initialize_diagonal();
    fit_operator(&mut design, xty, yty, n, options, control)
}

/// Structured trend-filtering design, X = (−upper-triangular-ones)^(order+1).
struct TrendDesign<'a> {
    order: usize,
    means: &'a [f64],
    scales: &'a [f64],
    diagonal: Vec<f64>,
    fitted: Vec<f64>,
}

impl GramOperator for TrendDesign<'_> {
    fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }
    fn apply(&mut self, vector: &[f64], output: &mut [f64]) {
        let n = self.means.len();
        for (i, x) in self.fitted.iter_mut().enumerate() {
            *x = vector[i] / self.scales[i];
        }
        for _ in 0..=self.order {
            let mut total = 0.;
            for x in self.fitted.iter_mut().rev() {
                total += *x;
                *x = -total;
            }
        }
        let shift = sum((0..n).map(|j| self.means[j] * vector[j] / self.scales[j]));
        for x in &mut self.fitted {
            *x -= shift;
        }
        let fitted_sum = sum(self.fitted.iter().copied());
        output[..n].copy_from_slice(&self.fitted);
        for _ in 0..=self.order {
            let mut total = 0.;
            for x in &mut output[..n] {
                total += *x;
                *x = -total;
            }
        }
        for (j, x) in output[..n].iter_mut().enumerate() {
            *x = (*x - self.means[j] * fitted_sum) / self.scales[j];
        }
        output[n..].fill(0.);
    }
}

/// Fit a trend-filtering design in O(n) storage, without constructing X or XᵀX.
#[allow(clippy::too_many_arguments)]
pub fn fit_design_trend(
    nrow: usize,
    order: usize,
    means: &[f64],
    scales: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    control: &FitControl,
) -> Result<FitOutput, String> {
    validate_design(nrow, nrow, means, scales, xty.len(), options)?;
    if n != nrow as f64 {
        return Err("Invalid trend-filtering sample size or order".into());
    }
    let mut sums = vec![1.; nrow];
    let mut base = vec![1.0_f64; nrow];
    for _ in 0..=order {
        let mut total = 0.;
        for x in &mut sums {
            total += *x;
            *x = -total;
        }
    }
    for _ in 0..order {
        let mut total = 0.;
        for x in &mut base {
            total += *x;
            *x = total;
        }
    }
    let mut norm = 0.;
    let mut diagonal = vec![0.; xty.len()];
    for j in 0..nrow {
        norm += base[j].powi(2);
        diagonal[j] =
            ((norm - 2. * means[j] * sums[j] + n * means[j].powi(2)) / scales[j].powi(2)).max(0.);
    }
    let mut design = TrendDesign {
        order,
        means,
        scales,
        diagonal,
        fitted: vec![0.; nrow],
    };
    fit_operator(&mut design, xty, yty, n, options, control)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn centered_sparse_and_dense_operators_match_explicit_gram() {
        let dense = [1., 0., 2., 0., 0., 3., 0., 4.]; // 4 × 2, column-major
        let means = [0.75, 1.75];
        let scales = [2., 0.5];
        let vector = [2., -1., 17.];
        let mut a = Design {
            storage: Storage::Dense {
                values: &dense,
                column_major: true,
            },
            nrow: 4,
            ncol: 2,
            means: &means,
            scales: &scales,
            diagonal: vec![0.; 3],
            fitted: vec![0.; 4],
        };
        let mut b = Design {
            storage: Storage::Csc {
                values: &[1., 2., 3., 4.],
                indices: &[0, 2, 1, 3],
                indptr: &[0, 2, 4],
            },
            nrow: 4,
            ncol: 2,
            means: &means,
            scales: &scales,
            diagonal: vec![0.; 3],
            fitted: vec![0.; 4],
        };
        a.initialize_diagonal();
        b.initialize_diagonal();
        let mut av = [0.; 3];
        let mut bv = [0.; 3];
        a.apply(&vector, &mut av);
        b.apply(&vector, &mut bv);
        assert_eq!(av, bv);
        assert_eq!(a.diagonal, b.diagonal);
        assert_eq!(av[2], 0.);
        for j in 0..2 {
            let expected = sum((0..2).map(|k| {
                vector[k]
                    * sum((0..4).map(|i| {
                        (dense[j * 4 + i] - means[j]) * (dense[k * 4 + i] - means[k])
                            / (scales[j] * scales[k])
                    }))
            }));
            assert!((av[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn rss_factor_sample_size_is_independent_of_reference_rows() {
        use crate::options::PriorMethod;
        let dense = [1., 0., 2., 0., 0., 3., 0., 4.];
        let means = [0.75, 1.75];
        let scales = [2., 0.5];
        let gram: Vec<f64> = (0..4)
            .map(|index| {
                let (i, j) = (index % 2, index / 2);
                sum((0..4).map(|row| {
                    (dense[i * 4 + row] - means[i]) * (dense[j * 4 + row] - means[j])
                        / (scales[i] * scales[j])
                }))
            })
            .collect();
        let options = FitOptions {
            l: 2,
            prior_variance: 0.2,
            residual_variance: 1.,
            prior_weights: vec![0.5; 2],
            estimate_prior_variance: false,
            estimate_prior_method: PriorMethod::None,
            estimate_residual_variance: true,
            check_null_threshold: 0.,
            max_iter: 100,
            tol: 1e-9,
            check_prior: false,
            prior_tol: 1e-9,
            null_index: None,
        };
        let control = FitControl::default();
        let xty = [1., -2.];
        let expected =
            crate::fit_controlled(&gram, &xty, 10., 17., &options, true, None, &control).unwrap();
        let dense_result = fit_design_dense(
            &dense, 4, 2, true, &means, &scales, &xty, 10., 17., &options, &control,
        )
        .unwrap();
        let sparse_result = fit_design_csc(
            &[1., 2., 3., 4.],
            &[0, 2, 1, 3],
            &[0, 2, 4],
            4,
            2,
            &means,
            &scales,
            &xty,
            10.,
            17.,
            &options,
            &control,
        )
        .unwrap();
        for actual in [dense_result, sparse_result] {
            for (actual, expected) in actual.alpha.iter().zip(&expected.alpha) {
                assert!((actual - expected).abs() <= 1e-12);
            }
            assert!((actual.sigma2 - expected.sigma2).abs() <= 1e-12);
        }
    }
}

#[cfg(test)]
mod trend_tests {
    use super::*;
    #[test]
    fn cumulative_operator_matches_independent_binomial_design() {
        let n = 6;
        let means = [0.2, -0.1, 0.3, 0.4, -0.5, 0.7];
        let scales = [1., 2., 0.5, 3., 0.25, 1.5];
        let vector = [1., -2., 3., 0.5, 0., -1.];
        for order in 0..=3 {
            let sign = if order % 2 == 0 { -1. } else { 1. };
            let mut dense = vec![0.; n * n];
            for j in 0..n {
                for i in 0..=j {
                    let mut choose = 1.;
                    for k in 1..=order {
                        choose *= (j - i + k) as f64 / k as f64;
                    }
                    dense[j * n + i] = sign * choose;
                }
            }
            let mut expected = Design {
                storage: Storage::Dense {
                    values: &dense,
                    column_major: true,
                },
                nrow: n,
                ncol: n,
                means: &means,
                scales: &scales,
                diagonal: vec![0.; n],
                fitted: vec![0.; n],
            };
            let mut actual = TrendDesign {
                order,
                means: &means,
                scales: &scales,
                diagonal: vec![0.; n],
                fitted: vec![0.; n],
            };
            let mut a = [0.; 6];
            let mut b = [0.; 6];
            actual.apply(&vector, &mut a);
            expected.apply(&vector, &mut b);
            for (a, b) in a.iter().zip(b) {
                assert!((a - b).abs() < 1e-9, "{a} != {b}");
            }
        }
    }
}
