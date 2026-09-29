# V13 revision 2: classification, evidence and output-channel repairs

Public pipeline name: **V13**. This is a separate, auditable input condition within V13, not V14. Previous V13 requests, returned answers (including rejected and truncated answers), and evaluation decisions remain historical records.

This is the operator's dated repair record. Its local experiment directories and Mercury paths are provenance references; source publication does not redistribute the underlying PDFs, datasets, model weights, private results, or credentials. Public readers can inspect the code and tests, while claims about individual GPU outputs require the retained private audit artifacts.

## Scope and scientific boundary

The same ten paper–dataset matches, eleven physical dataset assets, three local model weights, three replicate seeds and MinerU preprocessing are retained. Structural auxiliary classification remains disabled. Taxonomy classification remains enabled. Soft reference remains deferred. Dataset evidence never enters the paper classifier or extraction requests.

These repairs address execution and citation contracts. Contract acceptance does not establish schema fidelity, semantic entailment, dataset correspondence, or recall over `G_visible`. Those require the independent evidence evaluation defined by the study. Narrower context is a new condition: distant definitions may require uncertainty rather than guessing or declaring absence.

## Repairs

| Issue | V13 revision 2 behavior | Verification |
|---|---|---|
| Classification schema compiled in preflight but absent from production | Production, replay and new preflight share `production_request`; per-group grammar fixes entry order, unit IDs, state/category compatibility and quote IDs | Real container compilation and fixed classifier probes, plus mock production/replay tests |
| Competing integer unit/window namespaces | Model-facing extraction evidence uses typed `W000000` IDs throughout; source text is displayed directly; object and fact primary evidence must come from the assigned page | Foreign IDs, wrong table references and evidence/materialization tests |
| Muse reasoning consumes the entire output budget | Versioned `muse-atem/v2`: reasoning cap 2,048, final reserve 8,192, delimiter allowance 64 within the unchanged 16,384-token total | Actual Muse generation and token/channel telemetry |
| Unbounded whitespace and unrecorded compiler order | Versioned JSON/Muse grammar limits whitespace to 2 and pins the exact ordered schema string separately from canonical JSON identity | Request/replay tests and actual XGrammar 0.2.8 compilation |
| Full-paper distraction during a page assignment | Complete target-page windows plus the immediately preceding/following PDF pages; every page remains scheduled; classification sees its complete target group | All-corpus token and schema preflight; cross-page scope checks |
| Native/MinerU duplicate table views | Preserve separate locators and explicit duplicate-view diagnostics; each table has its own eligible evidence list and the inherited four-window citation limit | Actual V6 P01 p9 and P03 p4 inputs, each containing four table views; over-limit regression test |

The new protocol identifiers are `classification-anchors/v13r2` and `extraction-page-regions/v13r2`. Historical classifier v3, extraction V13 and grammar V1 behavior remain available for replay. New profile hashes identify changed runtime controls; unchanged human-readable model slot names do not imply unchanged experimental identity.

The model sees an abbreviated display contract while the decoder receives the complete bound schema. Repeated window/quote enumerations are removed only from the displayed contract. Identical navigation labels share a list of W IDs; complete source text remains directly attached to each W ID. Target-page quote options remain explicit. The source contains no dictionary that the model must decode to recover paper text.

An offline comparison of candidate 01 using the same disabled-navigation condition measured the following prompt character counts (not model token counts or speed measurements):

| Case | Original V13 | Repaired V13 | Reduction |
|---|---:|---:|---:|
| P01 p9 | 143,152 | 85,572 | 40.2% |
| P01 p10 | 138,054 | 105,445 | 23.6% |
| P03 p4 | 153,799 | 86,534 | 43.7% |
| P06 p8 | 61,602 | 59,722 | 3.1% |
| P08 p9 | 292,990 | 208,756 | 28.7% |

## Qualification and rollout

The fixed GPU load uses P01 p9/p10, P03 p4, P06 p8 and P08 p9 for each of the three models, seed 1729, one attempt per case. Qwen additionally runs the first existing classifier group overlapping P01 p11, P03 p4, P06 p8 and P08 p9. P08 p9 is a survey/hardware table used to stress context, not a declared positive dataset-field case. P01 p9, P03 p4 and P06 p8 must produce field candidates; an empty schema alone cannot qualify these cases.

These initial extraction probes use an explicitly disabled navigation index to isolate the output interface. Their reports do not claim production equivalence or fidelity. Production must build and verify its actual classifier index and recount each resulting request before dispatch.

Run qualification with:

```powershell
python scripts/deploy_v13_repair.py --candidate 2
python scripts/start_v13_repair_qualification.py --candidate 2
```

These commands have already been executed for candidate 02; they deliberately reject reuse of an existing candidate directory. Each candidate is immutable. The launcher waits for both GPU lease namespaces to be released, verifies assigned GPUs have no unowned processes, runs CPU preflight and independent capacity qualification, then runs the fixed output probes. The resource wait has a three-hour deadline and does not preempt another owner. It records raw answers, request identities, compilation evidence, progress, memory, channel-token counts, validation and replay. A returned invalid answer is retained and is not retried under the same condition. Failed qualifications block production freeze.

`freeze_v13_repair.py` requires passing reports, matching source/config/profile identities, all fixed probe record hashes and the actual model/image byte verification before bootstrapping a separate Scheduler V2 run. It does not start production by itself.

## Historical preservation

The original V13 run was cooperatively drained at group boundaries. All three GPU workers released their resources, all returned groups completed validation, and only the exact identified old controller/observer processes were stopped. The stopped run contains 90 returned classifier groups and 101 returned extraction groups. Of these, 74 classifier groups and 69 extraction groups were admitted under their original contracts. These are operational admission counts, not accuracy estimates.

The preservation baseline includes 1,804 stopped-run files, plus 1,002 source/input/record pins in the initial baseline. The local audit capture of 161 earlier V13 records replays successfully under the repaired checkout without changing their recorded statuses.

After staging candidate 01, and again after staging candidate 02 and its notification route, all **2,612 distinct baseline files** retained their recorded byte SHA-256 values. The later receipt is `preservation-check-after-candidate02.json`. This is a repair-time preservation check, not retroactive proof of historical identity.

Remote repair root:
`/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-repair-20260928-r2`

Local evidence root:
`data/experiments/v13_repair_2026_09_28_r2`

## Status

As of 2026-09-28 18:47 PDT, candidate 02 is queued on Mercury. Its **826 all-corpus CPU requests passed** token/context checks and actual XGrammar compilation (385 classifier requests and 147 extraction requests for each of three models). It is waiting for candidate 01's fixed probes to return their GPU leases. Full ten-paper production remains stopped. Candidate 01 passed all three GPU capacity tests; candidate 02 still requires its own capacity and output qualification.

Candidate 01's initial probe launcher stopped before generation because it incorrectly included the explicitly deferred frontier slot in local execution readiness. A separately pinned operational helper fixed that check, verifying every active profile and parameter while honoring deferred soft reference. The initial failure and helper identities are retained.

The resumed candidate 01 probes exposed a genuine schema defect: the new per-table evidence allowlist omitted the existing maximum of four references, so models emitted all eligible window IDs and the downstream validator rejected them. Candidate 02 restores that bound in the decoder schema; a regression test rejects five references. Candidate 01 answers are retained as rejected, including answers containing useful field candidates. The two candidate conditions are never pooled as interchangeable replicates.

Muse's first four extraction probes returned final JSON after exactly 2,048 reasoning tokens, with 4,050 / 1,284 / 5,508 / 4,444 final tokens. This shows the reserved final channel operating. Its P01 p9 output still contained no field candidates; P03 p4 and P06 p8 contained 18 and 7 candidates respectively. Qwen and Gemma each returned 32 field candidates on P01 p9. These answers remain contract-invalid because of the reference-limit defect. Candidate counts are neither accepted outputs nor verified dataset matches. Field omission and duplicate representation effects therefore remain under investigation.

Candidate 01's all-corpus input token totals were 81.5% lower for classification and 54.9–56.4% lower for extraction than the corresponding original V13 request construction. These descriptive input-size measurements exclude runtime effects and do not demonstrate better fidelity. Exact per-request measurements are in `all-corpus-qualification-metrics.json`; candidate 02 adds the citation bound and has separately frozen identities.

Offline checks passed: 91 core tests (4 local XGrammar-dependent skips), 42 operational tests, 15 downstream tests, 14 candidate-02 contract/packet tests, 52 notification/bridge tests and 8 probe/collection tests. Some suites overlap; these numbers must not be added as a unique-test count. The deployed container supplies the actual XGrammar checks omitted locally.

The candidate-02 notification observer is running. The local receiver received and authenticated its transport-test event, has no startup errors, and is armed. Completion/failure will trigger an independent read-only Codex diagnostic and a Telegram reply after this desktop turn is idle. The test event itself does not launch Codex, so a completed candidate-02 diagnostic is not yet claimed. Signed terminal receipts bind the original qualification summary and per-model report hashes; records and paper inputs are checked during collection. This route cannot activate production automatically. See [notification operations](v13_repair_probe_notifications.md).

Candidate-02 report directory:
`/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-repair-20260928-r2/candidate-02/audit`

Consult candidate-specific reports for executed GPU results. Staging or grammar compilation alone does not establish a qualified production deployment or improved fidelity.
