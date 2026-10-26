args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args)) args[[1]] else
    "tests/fixtures/susier_parity_0166/reference.json"
suppressPackageStartupMessages(library(susieR))
stopifnot(as.character(packageVersion("susieR")) == "0.16.6")
description <- packageDescription("susieR")
commit <- "8e56a8e038e989856d106d9ca5175cc664fea9d2"
stopifnot(!is.null(description$RemoteSha), description$RemoteSha == commit)
fixture <- jsonlite::fromJSON(path, simplifyVector = TRUE)
stopifnot(fixture$reference$version == "0.16.6", fixture$reference$commit == commit)
data <- fixture$datasets$N2finemapping
common <- list(L = 3, max_iter = 100, tol = 0.001,
               estimate_residual_variance = FALSE, residual_variance = 1,
               n_purity = "auto")
inputs <- list(
    individual_default = list(X = data$X, y = data$y),
    suff_stat_default = list(XtX = data$XtX, Xty = data$Xty, yty = data$yty, n = data$n),
    rss_default = list(z = data$z, R = data$R, n = data$n)
)
fitters <- list(individual_default = susie, suff_stat_default = susie_ss,
                rss_default = susie_rss)
for (name in names(inputs)) {
    fit <- suppressWarnings(do.call(fitters[[name]], c(inputs[[name]], common)))
    expected <- fixture$cases[[name]]$reference$alpha
    stopifnot(identical(dim(fit$alpha), dim(expected)),
              all(is.finite(fit$alpha)), all(fit$alpha >= 0), all(fit$alpha <= 1),
              max(abs(rowSums(fit$alpha) - 1)) <= 1e-10)
    error <- max(abs(fit$alpha - expected))
    stopifnot(is.finite(error), error <= 1e-5)
    cat(sprintf("%s: complete alpha max absolute error %.9g\n", name, error))
}
