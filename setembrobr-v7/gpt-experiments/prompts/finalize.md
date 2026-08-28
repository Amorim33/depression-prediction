# Fold prompt finalizer

Produce the final classifier prompt using only the candidate, its development-only evidence
catalog, and the independent red-team review. Apply supported revisions while preserving useful
signals. Do not add prevalence, examples, raw excerpts, user identifiers, claims not present in
the evidence, or knowledge from another fold. Return one evidence catalog and one complete prompt
in the requested schema.
