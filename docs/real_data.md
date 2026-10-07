# Real GWAS and eQTL recipes

The examples below describe two real-data entry points. They require separately
obtained association data and allele-aligned signed LD. The packaged tutorial
is synthetic; the downloadable benchmark records contain results, not these
human association or genotype inputs.

## GWAS: FinnGen R10 inflammatory bowel disease

The evaluated GWAS is `K11_IBD_STRICT` from FinnGen R10, with 9083 cases and
403098 controls (n=412181). FinnGen provides a
[download access form and R10 instructions](https://finngen.gitbook.io/documentation/r10/data-download).
Follow its access and use conditions, acknowledge participants and investigators,
and cite the FinnGen publication. A server returning a public URL does not
replace those data-use requirements.

The [R10 manifest](https://storage.googleapis.com/finngen-public-data-r10/summary_stats/R10_manifest.tsv)
identifies the release-specific file. After obtaining access under those terms,
download the endpoint named by the manifest and retain its checksum. The
[R10 field description](https://finngen.gitbook.io/documentation/r10/data-description)
defines `alt` as the effect allele; `beta` is log odds ratio and `sebeta` its
standard error. Calculate **signed z = beta / sebeta**. Record chromosome,
position, reference/alternate alleles and genome build for every variant.
The Gaussian summary model here uses n=412181, not an inferred effective sample
size or the number of LD samples.

Select a declared genomic window before inspecting fit outcomes, retain all
eligible variants in it, and align against a Finnish reference panel. The
benchmark used 99 recorded FIN founders with mean-imputed centered dosages and
Pearson signed r. A new reference release or sample selection creates a new
analysis; record those differences in its manifest.

## eQTL: BLUEPRINT monocytes

The evaluated expression dataset is eQTL Catalogue `QTD000021`, BLUEPRINT
monocytes, n=191. Its summary data are
[CC BY 4.0](https://www.ebi.ac.uk/eqtl/License/); cite the Catalogue and original
study and retain their attribution. The Catalogue's
[data-access page](https://www.ebi.ac.uk/eqtl/Data_access/) supplies dataset
metadata, file formats and indexed downloads. Its former REST API is retired.

A concrete source example from the Catalogue queries rs4239702 on chromosome
20. With HTSlib/tabix installed, query the documented index:

```sh
tabix https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/QTS000002/QTD000021/QTD000021.all.tsv.gz \
  20:46120612-46120613
```

This query is an association-record example, not a complete fine-mapping region.
For fitting, choose one gene and a prespecified cis window, fetch all its nominal
records in that window, and use beta/SE with ALT as the effect allele. Deduplicate
by full chromosome/position/REF/ALT identity for that gene, rather than by rsID
alone. Do not use an unsigned z reconstructed from a two-sided P value. Respect
the Catalogue's guidance on request frequency when making regional downloads.

Use an ancestry-matched LD reference on the same genome build. The benchmark
used 525 recorded EUR founders. Record expression quantification, normalization,
sample size and gene ID. The z-based RSS route does not require an invented
phenotype variance; provide `var_y` only when known for the actual phenotype.

## Align summaries and LD before fitting

1. Confirm build and chromosome/position/allele identities in both sources.
   State the reference sample selection and dosage-counted allele.
2. Restrict to the prespecified window, resolve duplicates, and document absent,
   multiallelic or ambiguous variants. Exclude unresolved palindromic orientation.
3. If the effect allele is swapped relative to counted dosage, multiply z by −1.
   If recoding an existing LD matrix with signs d, apply Rnew = diag(d) R diag(d).
4. Apply one final variant order to z, both R axes, IDs and metadata. Verify
   finite values, symmetry, unit diagonal, correlation range and the resulting
   byte/order hashes before fitting.

1000 Genomes reference data are available through
[IGSR](https://www.internationalgenome.org/data). Check the collection's
[data reuse policy](https://www.internationalgenome.org/faq/is-there-any-fee-associated-with-using-andor-reproducing-the-data/)
and retain sample/build provenance. LD is Pearson **r**, not r². Reference sample
size is separate from the association study's n.

Once prepared, both examples use the same API:

```python
import prusie

fit = prusie.susie_rss(z=signed_z, R=signed_ld, n=study_sample_size,
                       variant_ids=ordered_variant_ids, L=5,
                       max_iter=100, tol=0.001,
                       estimate_residual_variance=False)
```

The placeholders are the aligned arrays from your own preparation. The portable
case format and backend drivers are described in the
[benchmark guide](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md).
The source endpoints and indexed eQTL file were accessible by bounded HTTP
range requests during documentation preparation. That check verifies download
access, not a newly executed end-to-end real-data fit. Exact reruns of the
retained panel still require its original frozen inputs and permission to use
them; see [reproducibility](reproducibility.md).
