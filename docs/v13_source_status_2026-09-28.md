# V13 source snapshot: current qualification status

This commit publishes source code for the V13 paper–dataset pipeline and its
Scheduler V2 runtime. The public name remains **V13**; `v13r2` identifies a
separately frozen repair condition and does not rename it V14.

V13 combines MinerU-prepared paper input, the shared four-category taxonomy,
taxonomy classification, grouped schema extraction, deterministic dataset
parsing, and evidence-bound packet checks. MinerU and the optional structural
helper have independent switches. The structural helper is off in the current
V13 run. The frontier soft reference is deferred. Dataset content is excluded
from paper classification and extraction requests.

Revision 2 binds the actual classification grammar to production requests,
uses typed `W...` evidence locators in extraction, narrows each extraction
request to one target page with adjacent pages for context, and reserves a
separate final-answer budget for Muse. Distinct table views retain distinct
source identities and bounded citation lists. The source includes group-level
dispatch and recovery, source and output replay checks, qualification probes,
and notification helpers. See the [repair record](v13_revision2_repair_2026_09_28.md).

At this source snapshot, all 826 all-corpus CPU request constructions passed
context and grammar compilation checks. An earlier fixed GPU probe found a
table-citation bound omission and an unresolved Muse field omission. The
corrected candidate has completed all-corpus CPU grammar/context preflight and
has entered its separate GPU output probes. At this source snapshot, no final
candidate-02 output qualification has been recorded. The
ten-paper production rerun, independent `G_visible` annotation, correspondence
review, and formal fidelity results are **not complete**. A parseable response
or dataset-name match is not evidence of high-fidelity extraction.

The public Docker Hub `0.1.0-cuda13` image remains bound to its earlier source.
No V13 image, data, credentials, model weights, or private run outputs are
published with this commit. Site-specific scripts require the user's own
source bundle and qualified model deployments.

The public export normalizes text line endings and adapts tests to this
repository's package layout. Its byte inventory describes this Git checkout.
The private candidate's own frozen source, input, model and image manifests
remain authoritative for replaying any Mercury result; this commit is not a
retroactive substitute for those manifests.
