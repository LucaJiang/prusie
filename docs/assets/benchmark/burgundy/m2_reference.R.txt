# Fresh official susieR 0.16.6 fit, integrity, complete-call time and process HWM.
argv <- commandArgs(TRUE)
stopifnot(length(argv) %% 2L == 0L)
arg <- setNames(as.list(argv[seq.int(2L,length(argv),2L)]),
                sub('^--','',argv[seq.int(1L,length(argv),2L)]))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(susieR))
stopifnot(as.character(packageVersion('susieR')) == '0.16.6')
request <- fromJSON(arg$request,simplifyVector=FALSE)
protocol <- fromJSON(request$protocol_path,simplifyVector=FALSE)
task <- request$task
case <- fromJSON(task$case_path,simplifyVector=FALSE)
base <- dirname(task$case_path)
absolute <- function(p) if (startsWith(p,'/')) p else file.path(base,p)
thread_names <- c('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','PRUSIE_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')
stopifnot(all(Sys.getenv(thread_names)=='1'))
if (requireNamespace('RcppParallel',quietly=TRUE)) RcppParallel::setThreadOptions(numThreads=1L)
dyn.load(request$blas_probe)
clock <- function() .Call('prusie_monotonic')
fit_timer <- NULL
fit_function <- susieR::susie_rss
if (!is.null(request$fit_timing)) {
    source(request$fit_timing$R_helper)
    fit_timer <- make_susier_fit_timer(clock, request$fit_timing$R_source_file)
    fit_function <- fit_timer$fit
}
probe <- .Call('prusie_blas_info')
stopifnot(!is.na(probe$threads) && probe$threads==1L)
stopifnot(normalizePath(find.package('susieR'))==normalizePath(request$R_expected$path),
          normalizePath(trimws(extSoftVersion()[['BLAS']]))==normalizePath(trimws(request$R_expected$blas)),
          identical(probe$config,request$R_expected$blas_probe$config))
peak_rss <- function() {
    line <- readLines('/proc/self/status')
    as.numeric(sub('^VmHWM:\\s*([0-9]+).*','\\1',line[grepl('^VmHWM:',line)]))*1024
}
read_doubles <- function(path,n,compressed=FALSE) {
    connection <- if (compressed) gzfile(path,'rb') else file(path,'rb')
    on.exit(close(connection))
    value <- readBin(connection,'double',n=n,size=8L,endian='little')
    stopifnot(length(value)==n,length(readBin(connection,'raw',n=1L))==0L)
    value
}
record <- list(status='error',backend='susieR',version=as.character(packageVersion('susieR')),
               mode=arg$mode,signature_sha256=request$signature_sha256,samples=list(),
               expected_reference_commit='8e56a8e038e989856d106d9ca5175cc664fea9d2',
               environment=list(R=R.version.string,package_path=find.package('susieR'),
                 blas=extSoftVersion()[['BLAS']],blas_probe=probe,
                 session=I(capture.output(sessionInfo())),threads=as.list(Sys.getenv(thread_names)),
                 process_status=I(readLines('/proc/self/status')),
                 process_limits=I(readLines('/proc/self/limits'))))
tryCatch({
    input_started <- clock()
    full_p <- as.integer(case$n_variants)
    p <- if (is.null(task$window)) full_p else as.integer(task$window)
    z <- read_doubles(absolute(case$z_path),full_p)
    if (!is.null(task$window) && is.null(case$R_gzip_path)) {
        # Materialize only the declared nested p by p window; loading the full
        # locus first would contaminate the controlled-memory series HWM.
        column_major <- !is.null(case$R_column_major_path)
        path <- if (column_major) case$R_column_major_path else case$R_path
        connection <- file(absolute(path),'rb')
        R <- matrix(0,p,p)
        for (j in seq_len(p)) {
            seek(connection,where=8*(j-1)*full_p,origin='start')
            values <- readBin(connection,'double',n=p,size=8L,endian='little')
            stopifnot(length(values)==p)
            if (column_major) R[,j] <- values else R[j,] <- values
        }
        close(connection)
    } else if (!is.null(case$R_gzip_path)) {
        R <- read_doubles(absolute(case$R_gzip_path),full_p*full_p,TRUE); dim(R) <- c(full_p,full_p)
    } else if (!is.null(case$R_column_major_path)) {
        R <- read_doubles(absolute(case$R_column_major_path),full_p*full_p); dim(R) <- c(full_p,full_p)
    } else {
        R <- matrix(read_doubles(absolute(case$R_path),full_p*full_p),full_p,full_p,byrow=TRUE)
    }
    ids <- readLines(absolute(case$variant_ids_path))
    stopifnot(length(ids)==full_p,!anyDuplicated(ids),all(nchar(ids)>0),all(is.finite(z)))
    if (!is.null(task$window)) {
        z <- z[seq_len(p)]; ids <- ids[seq_len(p)]
        if (!identical(dim(R),c(p,p))) R <- R[seq_len(p),seq_len(p),drop=FALSE]
    }
    dimnames(R) <- list(ids,ids); names(z) <- ids
    params <- case$parameters
    for (key in intersect(c('L','max_iter','n_purity'),names(params))) params[[key]] <- as.integer(params[[key]])
    if (!is.null(params$prior_weights)) params$prior_weights <- unlist(params$prior_weights)
    moved <- intersect(c('r_tol','check_input','check_prior'),names(params))
    params$control <- do.call(susieR::susie_rss_control,params[moved]); params[moved] <- NULL
    params$estimate_residual_method <- 'MoM'
    params$z_method <- 'wald'; params$unmappable_effects <- 'none'
    params$convergence_method <- 'elbo'; params$R_mismatch <- 'none'
    inputs <- c(list(z=z,R=R),params)
    record$input_load_seconds <- clock()-input_started
    fit <- NULL
    for (i in if (arg$mode=='time') 0:as.integer(protocol$timing$retained_calls) else 0L) {
        fit <- NULL; gc(verbose=FALSE)
        warns <- character()
        if (!is.null(fit_timer)) fit_timer$reset()
        start_epoch <- as.numeric(Sys.time()); start <- clock()
        fit <- withCallingHandlers(do.call(fit_function,inputs),
          warning=function(w) {warns<<-c(warns,conditionMessage(w));invokeRestart('muffleWarning')},
          message=function(w) {warns<<-c(warns,conditionMessage(w));invokeRestart('muffleMessage')})
        elapsed <- clock()-start; end_epoch <- as.numeric(Sys.time()); peak <- peak_rss()
        a <- unname(fit$alpha); pip <- as.vector(fit$pip)
        valid <- identical(dim(a),c(as.integer(min(case$parameters$L,p)),p)) &&
          identical(as.character(colnames(fit$alpha)),ids) && length(pip)==p &&
          all(is.finite(a) & a>=0 & a<=1) && all(is.finite(pip) & pip>=0 & pip<=1) &&
          max(abs(rowSums(a)-1))<=1e-10 && all(is.finite(fit$V) & fit$V>=0) &&
          is.finite(fit$sigma2) && fit$sigma2>0 && fit$niter>=1 && fit$niter<=case$parameters$max_iter
        sample <- list('repeat'=i,warmup=arg$mode=='time' && i==0L,
          time_seconds=if (arg$mode=='time') elapsed else NULL,
          peak_rss_bytes=if (arg$mode=='memory') peak else NULL,
          start_epoch=start_epoch,end_epoch=end_epoch,niter=fit$niter,converged=fit$converged,
          cs_count=length(fit$sets$cs_index),warnings=I(warns),validity_pass=valid)
        if (!is.null(fit_timer)) sample$fitting <- fit_timer$last()
        record$samples[[length(record$samples)+1L]] <- sample
        stopifnot(valid)
    }
    record$alpha_shape <- I(dim(a)); record$n_variants <- p
    record$parameters <- case$parameters
    if (arg$mode=='time') {
        connection <- file(paste0(arg$output,'.alpha.bin'),'wb');writeBin(as.vector(a),connection,size=8L,endian='little');close(connection)
        connection <- file(paste0(arg$output,'.pip.bin'),'wb');writeBin(pip,connection,size=8L,endian='little');close(connection)
    }
    record$peak_rss_bytes <- peak
    record$status <- 'ok'
},error=function(e) {record$status<<-'error';record$error<<-conditionMessage(e)})
write_json(record,arg$output,auto_unbox=TRUE,digits=NA,pretty=TRUE,null='null',na='null',matrix='rowmajor')
quit(status=if (record$status=='ok') 0L else 1L)
