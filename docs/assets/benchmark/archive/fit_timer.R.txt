# Same-call fitting-stage timer for the pinned susieR 0.16.6 reference.
# The installed namespace is inspected, never modified. Source files in pinned/
# retain upstream notices; source_manifest.json records their immutable origin.
make_susier_fit_timer <- function(clock, source_file, evidence_dir = NULL) {
  stopifnot(is.function(clock), length(source_file) == 1L)
  source_file <- normalizePath(source_file, mustWork = TRUE)
  namespace <- asNamespace("susieR")
  if (as.character(getRversion()) != "4.4.0")
    stop("The timer requires pinned R 4.4.0.")
  if (as.character(utils::packageVersion("susieR")) != "0.16.6")
    stop("The timer requires pinned susieR 0.16.6.")

  sha256 <- function(path) {
    output <- system2("sha256sum", shQuote(normalizePath(path, mustWork = TRUE)),
                      stdout = TRUE, stderr = TRUE)
    if (!is.null(attr(output, "status")) || length(output) != 1L ||
        !grepl("^[0-9a-f]{64}  ", output))
      stop("Cannot obtain SHA256 for ", path)
    substr(output, 1L, 64L)
  }
  source_hashes <- c(
    "susie.R.txt" = "13a8a1144ad14025350e5c9a0fa28136b0171bd4e542d35e217b973de2c9b975",
    "susie_workhorse.R.txt" = "c75de9bcc93811a631c434f28ce7038f5c2513d28b9ed2d51264559ed13cbef3",
    "susie_constructors.R.txt" = "1dc44c68632e82485c8f19fbcdbf2775e601efe4c5b6cbdc49aa3b9dc1151361",
    "iterative_bayesian_stepwise_selection.R.txt" = "b5cfea0703b88d0b555e9a2b92c64b13a1b978a6bb36adc77b22284c2c03df9f",
    "sufficient_stats_methods.R.txt" = "b10c6c3a2ebbf249aaf02579f3cb7fd68586b22b3b1662fcba514daef628ef73",
    "susie_get_functions.R.txt" = "cac45881f0fd8044e1a568469a243f2245d0637d90dc8be57510157134c3728e",
    "susie_utils.R.txt" = "e10d53e5ac5c4d1f9a10ac8540b13e7f1e59e9a7cbaa863f1a66622c3d41eb52",
    "model_methods.R.txt" = "96ce911755a5ae00c3f66848ce41da59d27a26ea2fee8c21e55000179c7c03f5"
  )
  if (basename(source_file) != "susie.R.txt")
    stop("source_file must identify the run-local pinned/susie.R.txt.")
  pinned_dir <- dirname(source_file)
  source_paths <- file.path(pinned_dir, names(source_hashes))
  actual_hashes <- vapply(source_paths, sha256, character(1))
  if (!identical(unname(actual_hashes), unname(source_hashes)))
    stop("A pinned timing-boundary source file has changed.")

  # Guard the complete public function and inspected lower-level boundaries.
  # Their functions continue to resolve in the unmodified installed namespace.
  guards <- list(
    "susie.R.txt" = "susie_rss",
    "susie_workhorse.R.txt" = "susie_workhorse",
    "susie_constructors.R.txt" = c("summary_stats_constructor", "sufficient_stats_constructor"),
    "iterative_bayesian_stepwise_selection.R.txt" = c("ibss_initialize.default", "ibss_finalize"),
    "sufficient_stats_methods.R.txt" = c("configure_data.ss", "initialize_susie_model.ss", "get_cs.ss", "[.scaled_XtX"),
    "susie_get_functions.R.txt" = "susie_get_cs",
    "susie_utils.R.txt" = "safe_cov2cor",
    "model_methods.R.txt" = "initialize_matrices.default"
  )
  originals <- list()
  for (filename in names(guards)) {
    expressions <- parse(file.path(pinned_dir, filename), keep.source = FALSE)
    for (name in guards[[filename]]) {
      matches <- Filter(function(x) is.call(x) && length(x) == 3L &&
                          identical(x[[1L]], as.name("<-")) &&
                          identical(x[[2L]], as.name(name)) &&
                          is.call(x[[3L]]) &&
                          identical(x[[3L]][[1L]], as.name("function")),
                        as.list(expressions))
      if (length(matches) != 1L)
        stop("Expected exactly one pinned function definition: ", name)
      definition <- matches[[1L]][[3L]]
      original <- get(name, envir = namespace, inherits = FALSE)
      if (!is.function(original) ||
          !identical(formals(original), definition[[2L]]) ||
          !identical(body(original), definition[[3L]], ignore.srcref = TRUE))
        stop("Installed namespace differs from pinned function: ", name)
      originals[[name]] <- original
    }
  }
  original <- originals[["susie_rss"]]
  original_body <- body(original)
  original_environment <- environment(original)
  statements <- as.list(original_body)
  workhorse_statement <- quote(model <- susie_workhorse(susie_objects$data, susie_objects$params))
  final_statement <- quote(return(model))
  workhorse_index <- which(vapply(statements, identical, logical(1), y = workhorse_statement))
  return_index <- which(vapply(statements, identical, logical(1), y = final_statement))
  if (length(workhorse_index) != 1L || length(return_index) != 1L ||
      return_index != length(statements) || workhorse_index >= return_index)
    stop("Pinned susie_rss timing boundaries are not unique or have changed.")
  reserved <- c(".prusie_fit_timing_state", ".prusie_fit_timing_clock")
  if (any(reserved %in% all.names(original_body)))
    stop("Timing bindings collide with pinned function symbols.")

  state <- new.env(parent = emptyenv())
  state$start <- NA_real_
  state$end <- NA_real_
  timer_environment <- new.env(parent = original_environment)
  timer_environment$.prusie_fit_timing_state <- state
  timer_environment$.prusie_fit_timing_clock <- clock
  lockEnvironment(timer_environment, bindings = TRUE)

  start_statement <- quote(.prusie_fit_timing_state$start <- .prusie_fit_timing_clock())
  end_statement <- quote(.prusie_fit_timing_state$end <- .prusie_fit_timing_clock())
  instrumented_statements <- list(statements[[1L]])
  for (i in seq.int(2L, length(statements))) {
    if (i == workhorse_index)
      instrumented_statements[[length(instrumented_statements) + 1L]] <- start_statement
    if (i == return_index)
      instrumented_statements[[length(instrumented_statements) + 1L]] <- end_statement
    instrumented_statements[[length(instrumented_statements) + 1L]] <- statements[[i]]
  }
  instrumented <- original
  body(instrumented) <- as.call(instrumented_statements)
  environment(instrumented) <- timer_environment
  # The installed reference is byte-compiled. Compile the copy as well; do not
  # use a fully interpreted public wrapper for complete-call measurements.
  instrumented <- compiler::cmpfun(instrumented)

  reset <- function() {
    state$start <- NA_real_
    state$end <- NA_real_
    invisible(NULL)
  }
  last <- function() {
    if (length(state$start) != 1L || length(state$end) != 1L ||
        !is.finite(state$start) || !is.finite(state$end) ||
        state$end < state$start)
      stop("No successful complete fit has been timed since reset().")
    list(fit_seconds = unname(state$end - state$start),
         fit_start_monotonic = unname(state$start),
         fit_end_monotonic = unname(state$end))
  }
  # Remove exactly the two inserted statements and demand the original AST.
  restored <- as.list(body(instrumented))
  start_added <- workhorse_index
  end_added <- return_index + 1L
  if (!identical(restored[[start_added]], start_statement) ||
      !identical(restored[[end_added]], end_statement) ||
      !identical(as.call(restored[-c(start_added, end_added)]), original_body,
                 ignore.srcref = TRUE) ||
      !identical(formals(instrumented), formals(original)))
    stop("Instrumented AST differs by more than the two timestamp statements.")
  namespace_unchanged <- all(vapply(names(originals), function(name)
    identical(get(name, envir = namespace, inherits = FALSE), originals[[name]]),
    logical(1))) && identical(environment(original), original_environment) &&
    identical(body(original), original_body, ignore.srcref = TRUE)
  if (!namespace_unchanged)
    stop("Installed susieR namespace changed during instrumentation.")

  provenance <- list(
    package = "susieR", version = as.character(utils::packageVersion("susieR")),
    pinned_commit = "8e56a8e038e989856d106d9ca5175cc664fea9d2",
    installed_package_path = normalizePath(find.package("susieR")),
    source_hashes = as.list(source_hashes),
    source_namespace_ast_identical = TRUE,
    guarded_functions = names(originals), namespace_unchanged = TRUE,
    parent_namespace_identical = identical(parent.env(environment(instrumented)), original_environment),
    added_statements = 2L, original_statement_count = length(statements) - 1L,
    instrumented_statement_count = length(instrumented_statements) - 1L,
    compiler = "compiler::cmpfun with the active R default compiler options",
    timing_clock = "Caller-provided monotonic clock; use existing prusie_monotonic probe",
    start_boundary = "Immediately before model <- susie_workhorse(susie_objects$data, susie_objects$params), pinned R/susie.R:808",
    end_boundary = "Immediately before final return(model), after optional RSS multi-panel metadata, pinned R/susie.R:826",
    reset_getter_outside_complete_call = TRUE,
    limitations = "Frozen single-panel z/R path only; init_only is not a timed fit; timer is sequential and not reentrant. Scalar residual-variance checks within solver initialization and CS-specific diagonal/submatrix normalization remain included."
  )
  if (!is.null(evidence_dir)) {
    dir.create(evidence_dir, recursive = TRUE, showWarnings = FALSE)
    evidence_dir <- normalizePath(evidence_dir, mustWork = TRUE)
    canonical <- function(fn) {
      expression <- as.call(list(as.name("function"), formals(fn), body(fn)))
      c("susie_rss <-", deparse(expression, width.cutoff = 500L,
                               control = c("keepInteger", "keepNA", "showAttributes")))
    }
    original_path <- file.path(evidence_dir, "susie_rss.original.canonical.R.txt")
    instrumented_path <- file.path(evidence_dir, "susie_rss.instrumented.canonical.R.txt")
    writeLines(canonical(original), original_path, useBytes = TRUE)
    writeLines(canonical(instrumented), instrumented_path, useBytes = TRUE)
    diff_path <- file.path(evidence_dir, "susie_rss.instrumentation.diff")
    diff <- suppressWarnings(system2("diff", c("-u", "--label", "pinned/susie_rss.canonical", "--label",
                             "instrumented/susie_rss.canonical", shQuote(original_path),
                             shQuote(instrumented_path)), stdout = TRUE, stderr = TRUE))
    if (!identical(attr(diff, "status"), 1L))
      stop("Expected a nonempty canonical instrumentation diff.")
    writeLines(diff, diff_path, useBytes = TRUE)
    provenance$canonical_original_sha256 <- sha256(original_path)
    provenance$canonical_instrumented_sha256 <- sha256(instrumented_path)
    provenance$canonical_diff_sha256 <- sha256(diff_path)
  }
  list(fit = instrumented, reset = reset, last = last,
       original = original, provenance = provenance)
}
