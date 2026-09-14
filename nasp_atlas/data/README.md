# Bundled reference gene sets

`reference_sets.gmt` contains unsigned human Reactome and MSigDB Hallmark
signatures. `reference_sets.json` records official identifiers and names,
releases, source URLs, retrieval date, raw-response SHA-256 and gene counts.
These are **reference gene sets, not KEGG pathways** and not additions to the
curated compendium. No network access or writable cache is used during scoring.

## Sources and processing

- Reactome release 97, retrieved 2026-09-09. For each requested stable pathway
  ID, the official Content Service `data/query/{id}` supplied its name and
  species; `data/participants/{id}/referenceEntities` supplied participating
  reference entities (including nested events/complexes). We retained UniProt
  `ReferenceGeneProduct` entities with gene names, took their first gene name,
  and sorted/deduplicated the symbols. Chemical entities and unnamed proteins
  were excluded. Release identity was checked before and after retrieval.
- MSigDB `2026.1.Hs`, retrieved 2026-09-09, from the official
  [human Hallmark symbol GMT](https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt).
  The four requested records were selected by exact identifier, with sorted,
  deduplicated gene symbols. The recorded raw hash identifies the full source
  GMT. No orthologue conversion, synonym expansion, or direction assignment was
  performed.

Important naming correction: requested `R-HSA-975138` is officially **TRAF6
mediated induction of NFkB and MAP kinases upon TLR7/8 or 9 activation**, not
TRIF-mediated TLR3/TLR4 signaling. This bundle preserves the requested ID under
its official name; it must not be interpreted as a TRIF signature.

## Interpretation and updates

Pathway participation is not evidence that each member activates the pathway.
Every member contributes positively to its reference signature, including
inhibitors. These scores are expression summaries, not directional activation
scores. Shared genes and nested Reactome pathways make agreement non-independent;
the analysis exports overlap alongside donor/context-centered score agreement.
Gene coverage is measured against the input feature symbols, with unavailable
signatures explicitly recorded. A measured gene can still be undetected.

To update, repeat the source extraction, review names/membership, replace both
small files, and rerun scoring. Their combined hash is recorded in scores and
the completion manifest; incompatible scores cannot silently resume.

## Attribution and redistribution

Reactome data are released under [CC0](https://reactome.org/license).
MSigDB gene sets are redistributed under
[CC BY 4.0 and the MSigDB license terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp).
MSigDB copyright 2004-2025 Broad Institute, Inc., Massachusetts Institute of
Technology, and Regents of the University of California. This bundle changes
formatting and selects records, but does not curate additional gene members.
Please cite the source databases and
[Liberzon et al., Hallmark gene sets](https://doi.org/10.1016/j.cels.2015.12.004)
when publishing results using these signatures.
