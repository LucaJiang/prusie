# Execute with the already verified pinned susieR library; no source modifications.
args <- commandArgs(TRUE)
stopifnot(length(args)==4)
.libPaths(c(args[3], args[4], .libPaths()))
library(susieR); library(jsonlite)
stopifnot(as.character(packageVersion('susieR'))=='0.16.6')
cases <- fromJSON(args[1], simplifyVector=FALSE)
results <- list()
for (case in cases) {
  a <- case$args
  for (key in names(a)) {
    if (key %in% c('XtX','R')) a[[key]] <- do.call(rbind,lapply(a[[key]],unlist))
    else if (is.list(a[[key]])) a[[key]] <- unlist(a[[key]])
  }
  if (case$kind=='rss') {
    check <- if (is.null(a$check_prior)) TRUE else a$check_prior
    a$check_prior <- NULL
    a$control <- susie_rss_control(check_prior=check)
    fun <- susie_rss
  } else fun <- susie_ss
  warnings <- character()
  results[[case$name]] <- tryCatch(withCallingHandlers({
    f <- do.call(fun,a)
    # Explicit full-pair purity avoids independent RNG subsampling.
    cs <- list()
    if (length(f$sets$cs)) for (i in seq_along(f$sets$cs)) {
      cs[[i]] <- list(component=unname(f$sets$cs_index[i]-1),
          members=unname(f$sets$cs[[i]]-1), coverage=unname(f$sets$coverage[i]),
          purity=unname(as.numeric(f$sets$purity[i,])))
    }
    list(status='returned', fields=f[c('alpha','mu','mu2','pip','V','sigma2','elbo',
         'KL','lbf','lbf_variable')], niter=f$niter, converged=f$converged,
         stop_reason=if(is.null(f$convergence_reason)) NULL else f$convergence_reason,
         sets=cs, warnings=warnings)
  }, warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
  error=function(e)list(status='error',error=conditionMessage(e),warnings=warnings))
}
write_json(list(reference=list(version=as.character(packageVersion('susieR')),
    path=find.package('susieR'),R=R.version.string,session=capture.output(sessionInfo())),
    cases=results),args[2],digits=NA,auto_unbox=TRUE,matrix='rowmajor',pretty=TRUE,null='null',na='null')
cat('Executed',length(results),'pinned R cases\n')
