# Regenerates ld_oracles.json and kriging_certified.json with susieR 0.16.6 (commit 8e56a8e).
# Usage: Rscript generate_reference.R <output directory>
args <- commandArgs(TRUE)
out <- args[1]
suppressPackageStartupMessages({library(susieR); library(jsonlite)})
stopifnot(as.character(packageVersion("susieR")) == "0.16.6")
sha <- "8e56a8e038e989856d106d9ca5175cc664fea9d2"
ns <- asNamespace("susieR")

p <- 6L
spd <- outer(seq_len(p), seq_len(p), function(i, j) 0.65^abs(i - j))
singular <- spd
singular[6, ] <- singular[5, ]
singular[, 6] <- singular[, 5]
cases <- list(
  spd = list(R = spd, z = c(3.2, 2.7, -1.2, -1.8, 0.4, 0.2)),
  singular_consistent = list(R = singular, z = c(3.2, 2.7, -1.2, -1.8, 0.4, 0.4)),
  singular_flip = list(R = singular, z = c(3.2, 2.7, -1.2, -1.8, 4, -4)))

# kriging_rss with its final expression replaced so that it also returns the
# internal grid and mixture weights; the numerical operations are unchanged.
kriging <- get("kriging_rss", ns)
body_parts <- as.list(body(kriging))
body_parts[[length(body_parts)]] <- quote(list(conditional_dist = res, a_grid = a_grid,
  weights = w, logl0 = logl0mix, logl1 = logl1mix, s = s))
body(kriging) <- as.call(body_parts)

ld <- list()
for (name in names(cases)) for (n in c(200, Inf)) {
  x <- cases[[name]]
  sample <- if (is.finite(n)) list(n = n) else list()
  methods <- c("null-mle", "null-partialmle", "null-pseudomle")
  estimates <- setNames(lapply(methods, function(m)
    do.call(estimate_s_rss, c(list(z = x$z, R = x$R, method = m), sample))), methods)
  fits <- list()
  for (s in list(NULL, 0.1, 0)) {
    call_args <- c(list(z = x$z, R = x$R), sample)
    if (!is.null(s)) call_args$s <- s
    fits[[if (is.null(s)) "estimated" else paste0("s_", s)]] <- tryCatch(
      suppressWarnings(do.call(kriging, call_args)),
      error = function(e) list(error = conditionMessage(e)))
  }
  ld[[paste0(name, "_n", n)]] <- list(R = x$R, z = x$z, n = if (is.finite(n)) n else NULL,
    estimates = estimates, kriging = fits)
}
write_json(list(susieR = "0.16.6", source_commit = sha, R = R.version.string, cases = ld),
  file.path(out, "ld_oracles.json"), auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null")

# Certified global optimum of the full-likelihood normal mixture for each kriging
# fit: enumerate grid vertices and, if needed, all two-component mixtures with
# stats::optimize, then confirm optimality with the full simplex KKT condition.
raw <- fromJSON(file.path(out, "ld_oracles.json"))
answers <- list()
for (name in names(raw$cases)) for (kind in names(raw$cases[[name]]$kriging)) {
  x <- raw$cases[[name]]$kriging[[kind]]
  if (!is.null(x$error)) next
  d <- x$conditional_dist
  ll <- dnorm(d$z - d$condmean, sd = outer(sqrt(d$condvar), x$a_grid), log = TRUE)
  ll <- ll - apply(ll, 1, max)
  L <- exp(ll)
  k <- ncol(L)
  metric <- function(w) {
    density <- as.vector(L %*% w)
    list(loglik = sum(log(density)),
         kkt_violation = max(as.vector(crossprod(L, 1 / density))) - nrow(L), weights = w)
  }
  w <- numeric(k)
  w[which.max(colSums(ll))] <- 1
  best <- metric(w)
  if (best$kkt_violation > 1e-7) {
    for (i in seq_len(k - 1L)) for (j in seq.int(i + 1L, k)) {
      a <- L[, i]
      b <- L[, j]
      fit <- optimize(function(t) -sum(log(t * a + (1 - t) * b)), c(0, 1), tol = 1e-12)
      if (-fit$objective > best$loglik) {
        w <- numeric(k)
        w[i] <- fit$minimum
        w[j] <- 1 - fit$minimum
        best <- metric(w)
      }
    }
  }
  default <- metric(x$weights)
  log_mixture <- function(residual) {
    l <- dnorm(residual, sd = outer(sqrt(d$condvar), x$a_grid), log = TRUE)
    top <- apply(l, 1, max)
    as.vector(log(exp(l - top) %*% (best$weights + 1e-15)) + top)
  }
  logLR <- log_mixture(d$z + d$condmean) - log_mixture(d$z - d$condmean)
  answers[[paste0(name, "/", kind)]] <- list(s = x$s, certificate_passed = best$kkt_violation <= 1e-6,
    certified = best, default = default, logLR = logLR)
}
stopifnot(all(vapply(answers, function(a) a$certificate_passed, logical(1))))
write_json(list(susieR = "0.16.6", source_commit = sha, R = R.version.string, cases = answers),
  file.path(out, "kriging_certified.json"), auto_unbox = TRUE, pretty = TRUE, digits = 17)
