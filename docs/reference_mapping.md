# Reference source mapping

prusie follows the supported Gaussian paths of susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`.

| prusie responsibility | susieR source |
| --- | --- |
| RSS transformation | susie_constructors.R: summary_stats_constructor and summary_stats_working_quantities |
| Sufficient statistics and scaling | sufficient_stats_constructor; sufficient_stats_methods.R |
| Single-effect Bayes factors and moments | single_effect_regression.R: gaussian_ser_lbf and gaussian_ser_moments |
| Prior variance and Gaussian EM | prior-update hooks and generic_methods.R |
| IBSS, objective and stopping | susie_workhorse.R, model_methods.R and sufficient_stats_methods.R |
| Credible sets and PIP | susie_get_functions.R and susie_utils.R |

The scalar optimizer derives from R's `stats/src/optimize.c`; its license and
copyright remain in the [third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md).
Read [implementation and optimizations](implementation.md) for the executed path,
[model support](compatibility.md) for the API scope, and
[numerical accuracy](numerical_accuracy.md) for measured differences.
