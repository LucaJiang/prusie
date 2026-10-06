# Deterministic export of existing, pinned official teaching data. No generation.
# Usage: Rscript --vanilla export_reference.R COLOC_TEST_DATA_RDA OUTPUT_DIRECTORY
# Set R_LIBS_USER/R_LIBS to the libraries matching reference.lock.json first.
args <- commandArgs(TRUE)
stopifnot(length(args)==2)
suppressPackageStartupMessages(library(coloc))
suppressPackageStartupMessages(library(susieR))
library(jsonlite)
stopifnot(as.character(packageVersion('coloc'))=='6.0.3',
          as.character(packageVersion('susieR'))=='0.16.6')
options(digits=17)
out <- args[2]; dir.create(out, recursive=TRUE, showWarnings=FALSE)
jwrite <- function(x,name) write_json(x,file.path(out,name),auto_unbox=TRUE,digits=NA,
                                    pretty=FALSE,na='null',null='null',matrix='rowmajor')
load(args[1]); ds <- coloc_test_data[paste0('D',1:4)]
metadata <- list()
for(k in names(ds)) {
 d<-ds[[k]]; m<-length(d$snp)
 stopifnot(m==500, d$N==1000,d$type=='quant',!anyDuplicated(d$snp),
  identical(d$snp,rownames(d$LD)),identical(d$snp,colnames(d$LD)),
  all(is.finite(d$beta)),all(d$varbeta>0),all(is.finite(d$LD)),
  max(abs(d$LD-t(d$LD)))<1e-14,all(diag(d$LD)==1))
 metadata[[k]]<-list(N=d$N,sdY=d$sdY,type=d$type,m=m,
   snp=I(d$snp),position=I(d$position),causal_indices_1based=I(coloc_test_data$causals[[k]]),
   nature='existing official synthetic teaching fixture',genome_build=NULL,ancestry=NULL,
   effect_allele=NULL,other_allele=NULL,position_meaning='synthetic variable index, not base pairs')
 write.table(data.frame(snp=d$snp,position=d$position,beta=d$beta,varbeta=d$varbeta,MAF=d$MAF),
  file.path(out,paste0(k,'.tsv')),sep='\t',row.names=FALSE,quote=FALSE)
 for(field in c('beta','varbeta','MAF','LD')) writeBin(as.double(d[[field]]),file.path(out,paste0(k,'_',field,'.f64')),size=8,endian='little')
 write.table(d$LD,file.path(out,paste0(k,'_LD.tsv')),sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
}
jwrite(metadata,'metadata.json')
base <- list(L=10L,max_iter=100L,tol=1e-3,coverage=.95,min_abs_corr=.5,n_purity=500L,
 standardize=TRUE,scaled_prior_variance=.2,estimate_residual_variance=FALSE,
 estimate_prior_variance=TRUE,estimate_prior_method='optim',check_null_threshold=0,
 prior_tol=1e-9,null_weight=0,refine=FALSE,track_fit=FALSE,verbose=FALSE)
# Predeclared by dataset/parameters only, before native results are inspected.
cases <- list(D1=list(dataset='D1'),D2=list(dataset='D2'),D3=list(dataset='D3'),D4=list(dataset='D4'),
 D1_partial_zero=list(dataset='D1',prior_weights=as.numeric((1:500) %% 7 != 0)),
 D3_strict_purity=list(dataset='D3',min_abs_corr=1),
 D3_one_iteration=list(dataset='D3',max_iter=1L))
fits<-list(); fitmeta<-list(); params<-list()
for(k in names(cases)) {
 case<-cases[[k]]; d<-ds[[case$dataset]]; p<-modifyList(base,case[names(case)!='dataset'])
 params[[k]]<-c(list(dataset=case$dataset,input_kind='z',n=d$N,z_definition='beta/sqrt(varbeta)'),p)
 warnings<-character()
 f<-withCallingHandlers(do.call(susieR::susie_rss,c(list(z=d$beta/sqrt(d$varbeta),R=d$LD,n=d$N),p)),
    warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
 f<-coloc::annotate_susie(f,d$snp,d$LD);fits[[k]]<-f
 saveRDS(f,file.path(out,paste0('raw_fit_',k,'.rds')),version=3)
 arrays<-f[intersect(c('alpha','mu','mu2','lbf_variable','lbf','V','sigma2','pip','elbo','KL'),names(f))]
 # JSON arrays preserve matrix row/column shape; portable conversion uses float64.
 jwrite(arrays,paste0('arrays_',k,'.json'))
 css<-list()
 if(length(f$sets$cs)) for(i in seq_along(f$sets$cs)) {
  members<-unname(f$sets$cs[[i]])
  css[[length(css)+1L]]<-list(name=names(f$sets$cs)[i],component_id_0based=unname(f$sets$cs_index[i])-1L,
    component_id_1based=unname(f$sets$cs_index[i]),members_0based=I(members-1L),
    members_1based=I(members),coverage=unname(f$sets$coverage[i]),
    purity=as.list(f$sets$purity[i,,drop=FALSE]),
    independently_summed_alpha=sum(f$alpha[f$sets$cs_index[i],members]))
 }
 fitmeta[[k]]<-list(dataset=case$dataset,converged=f$converged,niter=f$niter,
  warnings=I(warnings),cs=css,requested_coverage=p$coverage,
  cs_coverage_available=length(css)>0,raw_sets=f$sets,
  array_fields=I(names(arrays)),snp=I(d$snp),component_index_base=0,
  raw_reference_component_index_base=1)
 cat(k,'niter',f$niter,'converged',f$converged,'CS',length(css),'\n')
}
jwrite(fitmeta,'r_fits.json');jwrite(params,'parameters.json')
coloc_ref<-list()
call_capture<-function(fun) {
 w<-character();e<-NULL
 value<-tryCatch(withCallingHandlers(fun(),warning=function(x){w<<-c(w,conditionMessage(x));invokeRestart('muffleWarning')}),error=function(x){e<<-conditionMessage(x);NULL})
 list(result=value,warnings=I(w),error=e)
}
cp<-list(p1=1e-4,p2=1e-4,p12=5e-6)
for(pair in list(c('D1','D2'),c('D3','D4'))) {
 key<-paste(pair,collapse='_')
 coloc_ref[[paste0('abf_',key)]]<-call_capture(function()do.call(coloc::coloc.abf,c(list(dataset1=ds[[pair[1]]],dataset2=ds[[pair[2]]]),cp)))
 coloc_ref[[paste0('susie_',key)]]<-call_capture(function()do.call(coloc::coloc.susie,c(list(dataset1=fits[[pair[1]]],dataset2=fits[[pair[2]]]),cp,list(overlap.min=.5,trim_by_posterior=TRUE))))
}
coloc_ref[['susie_strict_purity_D4']]<-call_capture(function()do.call(coloc::coloc.susie,c(list(dataset1=fits$D3_strict_purity,dataset2=fits$D4),cp)))
# Fixed software counterexample, not a synthetic study. Enumerated independently in Python.
bf<-setNames(c(0,0),c('a','b'))
coloc_ref[['weighted_equal_bf_counterexample']]<-call_capture(function()do.call(coloc::coloc.bf_bf,c(list(bf1=bf,bf2=bf,prior_weights1=c(9,1),prior_weights2=c(1,1)),cp)))
coloc_ref[['partial_zero_bf_counterexample']]<-call_capture(function()do.call(coloc::coloc.bf_bf,c(list(bf1=bf,bf2=bf,prior_weights1=c(1,0),prior_weights2=c(1,1)),cp)))
# Nonuniform positive weights fixed by index, independent of association statistics.
w1<-1+((1:500)%%5);w2<-1+((1:500)%%3)
coloc_ref[['abf_D1_D2_weighted']]<-call_capture(function()do.call(coloc::coloc.abf,c(list(dataset1=ds$D1,dataset2=ds$D2,prior_weights1=w1,prior_weights2=w2),cp)))
jwrite(list(coloc_parameters=cp,weighted_rule=list(w1='1+(one_based_index %% 5)',w2='1+(one_based_index %% 3)'),cases=coloc_ref),'r_coloc.json')
env<-list(R=R.version.string,platform=R.version$platform,lib_paths=I(.libPaths()),
 coloc_version=as.character(packageVersion('coloc')),coloc_path=find.package('coloc'),
 susieR_version=as.character(packageVersion('susieR')),susieR_path=find.package('susieR'),
 namespaces=lapply(setNames(loadedNamespaces(),loadedNamespaces()),function(k)list(version=as.character(packageVersion(k)),path=find.package(k))),
 session_info=I(capture.output(sessionInfo())),BLAS=extSoftVersion()[['BLAS']],
 susie_rss_formals=I(capture.output(dput(formals(susieR::susie_rss)))))
jwrite(env,'reference_environment.private.json')
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
