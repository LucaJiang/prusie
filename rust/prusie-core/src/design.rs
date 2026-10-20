//! Matrix-free individual-level regression with dense or canonical CSC designs.
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

/// Fit a dense individual-level design in working centered/scaled coordinates.
/// `means` and `scales` describe the raw `ncol` columns. If Xty has one extra
/// entry and `null_index == ncol`, it denotes a virtual all-zero null column.
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
    if n != nrow as f64 {
        return Err("Individual-level sample size must equal the number of design rows".into());
    }
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
    if n != nrow as f64 {
        return Err("Individual-level sample size must equal the number of design rows".into());
    }
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
}
