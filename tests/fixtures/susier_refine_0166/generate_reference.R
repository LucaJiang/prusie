# Regenerates refine_convergence.json with susieR 0.16.6 (commit 8e56a8e).
# Usage: Rscript generate_reference.R inputs.json refine_convergence.json
# inputs.json holds prusie.load_convergence_example()["inputs"] (XtX, Xty, yty, n).
args <- commandArgs(TRUE)
suppressPackageStartupMessages({library(susieR); library(jsonlite)})
stopifnot(as.character(packageVersion("susieR")) == "0.16.6")
d <- fromJSON(args[1], simplifyVector = TRUE)
fit <- function(...) suppressWarnings(suppressMessages(
  susie_ss(d$XtX, d$Xty, d$yty, d$n, L = 10, max_iter = 100, tol = 1e-3, ...)))
base <- fit(convergence_method = "pip")
weights <- rep(1 / length(d$Xty), length(d$Xty))
weights[base$sets$cs[[1]]] <- 0
weights <- weights / sum(weights)
cases <- list(
  pip_zero_weights = list(options = list(convergence_method = "pip", prior_weights = weights)),
  refine_pip = list(options = list(convergence_method = "pip", refine = TRUE)),
  refine_mixture = list(options = list(prior_variance_grid = c(0.02, 0.2, 1),
                                       mixture_weights = c(0.2, 0.5, 0.3), refine = TRUE)),
  refine_NIG = list(options = list(estimate_residual_method = "NIG", refine = TRUE)))
for (id in names(cases)) {
  f <- do.call(fit, cases[[id]]$options)
  cases[[id]]$alpha <- unname(f$alpha)
  cases[[id]]$elbo <- tail(f$elbo, 1)
}
write_json(list(susieR = "0.16.6", commit = "8e56a8e038e989856d106d9ca5175cc664fea9d2",
                common = list(L = 10, max_iter = 100, tol = 1e-3), cases = cases),
           args[2], digits = NA, auto_unbox = TRUE)
