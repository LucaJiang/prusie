#!/usr/bin/env Rscript
# One complete official SuSiE-RSS case, separate validation/time/memory modes.
# R_LIBS_USER and the selected R/BLAS installation are supplied by the caller.
argv <- commandArgs(TRUE)
stopifnot(length(argv) %% 2L == 0L)
arg <- setNames(as.list(argv[seq.int(2L,length(argv),2L)]),
                sub('^--','',argv[seq.int(1L,length(argv),2L)]))
stopifnot(all(c('case','mode','output') %in% names(arg)),
          arg$mode %in% c('validate','time','memory'))
suppressPackageStartupMessages(library(susieR))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('susieR')) == '0.16.6')
thread_names <- c('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','PRUSIE_NUM_THREADS')
stopifnot(all(Sys.getenv(thread_names) == '1'))
if (requireNamespace('RcppParallel',quietly=TRUE))
    RcppParallel::setThreadOptions(numThreads=1L)
case <- fromJSON(arg$case,simplifyVector=FALSE)
absolute <- function(p) {
    if (startsWith(p,'/')) p else file.path(dirname(normalizePath(arg$case)),p)
}
read_doubles <- function(path,n) {
    stopifnot(file.info(path)$size == 8*n)
    con <- file(path,'rb'); on.exit(close(con))
    readBin(con,'double',n=n,size=8L,endian='little')
}
load_case <- function(case) {
    m <- as.integer(case$n_variants)
    z <- read_doubles(absolute(case$z_path),m)
    if (!is.null(case$R_column_major_path)) {
        R <- read_doubles(absolute(case$R_column_major_path),m*m)
        dim(R) <- c(m,m)
    } else {
        R <- matrix(read_doubles(absolute(case$R_path),m*m),m,m,byrow=TRUE)
    }
    ids <- readLines(absolute(case$variant_ids_path))
    stopifnot(length(ids)==m,!anyDuplicated(ids),all(nchar(ids)>0))
    dimnames(R) <- list(ids,ids); names(z) <- ids
    params <- case$parameters
    for (key in intersect(c('L','max_iter','n_purity'),names(params)))
        params[[key]] <- as.integer(params[[key]])
    if (!is.null(params$prior_weights)) params$prior_weights <- unlist(params$prior_weights)
    moved <- intersect(c('r_tol','check_input','check_prior'),names(params))
    params$control <- do.call(susieR::susie_rss_control,params[moved])
    params[moved] <- NULL
    # Explicit current Gaussian semantics; no alternative model or LD repair.
    params$estimate_residual_method <- 'MoM'
    params$z_method <- 'wald'
    params$unmappable_effects <- 'none'
    params$convergence_method <- 'elbo'
    params$R_mismatch <- 'none'
    list(arguments=c(list(z=z,R=R),params),variant_ids=ids)
}
run_case <- function(inputs) do.call(susieR::susie_rss,inputs$arguments)
summarize_fit <- function(fit) list(
    niter=fit$niter,converged=fit$converged,
    cs_count=length(fit$sets$cs_index),
    cs_component_ids=I(as.integer(fit$sets$cs_index)-1L),
    status=if (fit$converged) 'converged' else 'nonconverged')
export_fit <- function(fit,ids) {
    fields <- c('pip','alpha','mu','mu2','lbf_variable','lbf','V','sigma2','elbo','KL')
    out <- fit[fields]
    for (key in fields) if (is.null(dim(out[[key]])) && key != 'sigma2')
        out[[key]] <- I(as.vector(out[[key]]))
    css <- list()
    if (length(fit$sets$cs_index)) for (k in seq_along(fit$sets$cs_index)) {
        purity <- as.list(fit$sets$purity[k,,drop=FALSE])
        names(purity) <- gsub('.','_',names(purity),fixed=TRUE)
        css[[k]] <- list(component_id_0based=as.integer(fit$sets$cs_index[k])-1L,
                          members_0based=I(as.integer(fit$sets$cs[[k]])-1L),
                          coverage=as.numeric(fit$sets$coverage[k]),purity=purity)
    }
    c(out,list(variant_ids=I(ids),cs=css,
               cs_return_order=I(as.integer(fit$sets$cs_index)-1L),
               requested_coverage=case$parameters$coverage,
               null_index=if (is.null(fit$null_index) || fit$null_index==0) NULL else fit$null_index-1L,
               niter=fit$niter,converged=fit$converged,params=case$parameters))
}
peak_rss <- function() {
    line <- readLines('/proc/self/status')
    as.numeric(sub('^VmHWM:\\s*([0-9]+).*','\\1',line[grepl('^VmHWM:',line)]))/1024
}
identity <- function() {
    maps <- readLines('/proc/self/maps')
    mapped <- unique(sub('^.* +(/[^ ]+)$','\\1',maps[grepl('/.*(blas|mkl|lapack|libmvec)',maps)]))
    probe <- NULL
    if (!is.null(arg[['blas-probe']])) {
        dyn.load(arg[['blas-probe']])
        probe <- .Call('prusie_blas_info')
    }
    list(version=as.character(packageVersion('susieR')),
         expected_reference_commit='8e56a8e038e989856d106d9ca5175cc664fea9d2',
         R=R.version.string,platform=R.version$platform,
         package_path=find.package('susieR'),
         package_description=unclass(packageDescription('susieR')),
         blas=extSoftVersion()[['BLAS']],blas_probe=probe,
         mapped_numerical_libraries=I(mapped),session=I(capture.output(sessionInfo())),
         threads=as.list(Sys.getenv(thread_names)),pid=Sys.getpid(),
         process_status=I(readLines('/proc/self/status')))
}
record <- list(case_id=case$case_id,stage='C',backend='susieR',mode=arg$mode,
               n_variants=case$n_variants,L=case$parameters$L,parameters=case$parameters,
               status='error',time_seconds=NULL,peak_rss_mib=NULL,numerical_errors=NULL,
               matrix_load=if (is.null(case$R_column_major_path)) 'row-major conversion' else 'column-major direct')
exit_code <- 1L
tryCatch({
    inputs <- load_case(case)
    if (!is.null(arg[['blas-probe']])) dyn.load(arg[['blas-probe']])
    if (arg$mode=='time' && is.null(arg[['blas-probe']]))
        stop('Timing requires --blas-probe for the high-resolution monotonic clock')
    clock <- if (arg$mode=='time') function() .Call('prusie_monotonic') else function() proc.time()[['elapsed']]
    repeats <- if (is.null(arg$repeats)) 7L else as.integer(arg$repeats)
    if (arg$mode=='time') stopifnot(repeats>=7L)
    indices <- if (arg$mode=='time') 0:repeats else 0L
    samples <- list(); fit <- NULL
    for (i in indices) {
        fit <- NULL; gc(verbose=FALSE)
        warns <- character()
        start_epoch <- as.numeric(Sys.time()); start <- clock()
        fit <- withCallingHandlers(run_case(inputs),
          warning=function(w) {warns<<-c(warns,conditionMessage(w));invokeRestart('muffleWarning')},
          message=function(w) {warns<<-c(warns,conditionMessage(w));invokeRestart('muffleMessage')})
        elapsed <- clock()-start; end_epoch <- as.numeric(Sys.time())
        peak <- peak_rss()
        samples[[length(samples)+1L]] <- c(list(repeat_index=i,'repeat'=i,
          warmup=arg$mode=='time' && i==0L,
          time_seconds=if (arg$mode=='time') elapsed else NULL,
          peak_rss_mib=if (arg$mode=='memory') peak else NULL,
          start_epoch=start_epoch,end_epoch=end_epoch,warnings=I(warns)),summarize_fit(fit))
    }
    record$status <- 'ok'; record$samples <- samples
    record$niter <- fit$niter; record$converged <- fit$converged
    if (arg$mode=='memory') record$peak_rss_mib <- samples[[1L]]$peak_rss_mib
    if (arg$mode=='validate') {
        record$fit <- export_fit(fit,inputs$variant_ids)
        a <- fit$alpha; pip <- as.vector(fit$pip)
        finite <- all(vapply(fit[c('alpha','mu','mu2','lbf_variable','lbf','V','sigma2','pip','elbo','KL')],
                             function(x) all(is.finite(x)),logical(1)))
        valid <- finite && all(a>=0 & a<=1) && all(pip>=0 & pip<=1) &&
            max(abs(rowSums(a)-1))<=1e-10 && all(fit$V>=0) && fit$sigma2>0 &&
            length(pip)==case$n_variants && identical(as.character(colnames(a)),inputs$variant_ids)
        record$validity <- list(passed=valid,alpha_mass_max_abs_error=max(abs(rowSums(a)-1)))
        if (!valid) record$status <- 'invalid'
        # Original official object for independent same-fit coloc pairing.
        # Export is separate from fit clocks and memory measurement.
        rds <- if (is.null(arg[['fit-rds']])) paste0(arg$output,'.fit.rds') else arg[['fit-rds']]
        dir.create(dirname(rds),recursive=TRUE,showWarnings=FALSE)
        saveRDS(fit,rds,version=3)
        record$fit_rds <- rds
    }
    record$environment <- identity()
    if (!is.null(record$environment$blas_probe$threads))
        stopifnot(is.na(record$environment$blas_probe$threads) || record$environment$blas_probe$threads==1L)
    exit_code <- if (record$status=='ok') 0L else 1L
},error=function(e) {record$status<<-'error';record$error<<-conditionMessage(e)})
dir.create(dirname(arg$output),recursive=TRUE,showWarnings=FALSE)
write_json(record,arg$output,auto_unbox=TRUE,digits=NA,pretty=TRUE,null='null',na='null',matrix='rowmajor')
quit(status=exit_code)
