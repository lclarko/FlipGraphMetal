# Bounded signed two-auxiliary construction

Family identifier: `signed-two-aux-distinct-v1`.

## Implementation status

Milestone 1 supplies shared bounded representations, host relation preparation,
a streaming route enumerator, witness validation, and independent host tests.
The host closure used by these tests lives under `tests/workflow`. Production
evaluation is unchanged. There is no new GPU kernel, production CPU closure,
configuration option, search score, or campaign enablement in this milestone.

The inspected implementation base is
`226b57cd0c2933037a86c8a870372ba2d5a6e4cd`. The randomized pair reducer already
permits several non-target intermediates. This family extends the explicit
zero-/one-auxiliary constructor's systematic coverage.

## Directions, routes, and permitted rules

A signed binary addition or subtraction costs one operation. Global sign
changes and reuse cost nothing. Normalize a nonzero coefficient vector by making
its first nonzero coefficient positive; do not divide by a gcd. Intermediate
coefficients may be nonternary. Final production factors stay in the admitted
signed domain, with unchanged ordered factors, tensor, and rank.

For a map with `i` inputs and at most 23 output occurrences, let `D` contain the
ordered basis followed by first-seen distinct normalized nonzero target
directions. Preserve each output's direction and sign. Zero outputs have no
direction. Duplicates, negations, and input aliases add no required direction.
Write `n = |D|`, `m = n-i`, and `q` for the number of nonzero output occurrences,
counting multiplicity.

A candidate selects two creation routes:

```text
h1 = normalize(D[a] +/- D[b]),             a < b
h2 = normalize((D+h1)[c] +/- (D+h1)[d]),   c < d
```

Both helpers must be nonzero, `h1` must be outside `D`, and `h2` must be outside
`D+h1`. Thus the second helper can depend on the first. An auxiliary is a
constructed direction outside the basis and targets, not just a temporary wire.

The candidate's complete rule set comprises:

1. Every signed pair of distinct directions in `D` producing a target.
2. Every signed pair in `D+h1+h2` involving at least one helper and producing a
   target, including pairs using both helpers.
3. Exactly the selected creation rule for each helper.

Store the normalization sign in both gate coefficients. Each creation route's
operands must actually have been constructed. Mentioning a target in a helper
definition does not make that target available. Cyclic dependencies can leave a
candidate unreachable.

Retain different creation routes for equal helper vectors. Merging such routes
can lose reachability. For independent helpers only, keep the representative
whose first original-`D` route ID is less than its second original-`D` route ID.
Re-express the second route in `D` before this comparison; its enumeration ID in
`D+h1` differs because helper pairs are interleaved. Never apply this quotient to
dependent second-helper routes.

Forward same-direction doubling is excluded. There are no free scalar
multiplications or divisions, more than two helpers, multiple creation routes
for a helper within one candidate, reconstructed directions, basis changes,
scheme mutations, or F2 construction. Reverse transposition can emit `x+x`, which
costs one binary operation.

## Enumeration and deterministic budgets

Enumerate unordered pairs lexicographically by `(left,right)`, `left < right`,
then addition before subtraction. The second-route list enumerates `D+h1` in
the same order, including interleaved pairs with `h1`.

```text
R = n(n-1)
S = n(n+1)
raw_slot = first_route * S + second_route
raw_family_size = R * S
```

At `n=32`, there are 992 first routes, 1,056 second routes, and 1,047,552 raw
slots. Independent symmetry gives an upper bound of
`choose(992,2) + 992*64 = 555,024` retained pairs before other filtering.

`TwoAuxStream` visits an exact prefix of raw slots. Its positive budget is at
most 1,047,552 and is clipped to the map's raw family size. Invalid first-helper
blocks may be skipped analytically, including a block clipped at the budget.
Invalid second helpers and independent-swap duplicates still consume slots.
Filtering never replenishes a budget.

Classify filters in this order: invalid first helper, invalid second helper,
independent symmetry. At each returned task and at stream exhaustion:

```text
rawScanned = firstInvalid + secondInvalid + symmetryFiltered + prepared
```

No list of all pairs or witnesses is materialized. Production integration will
use a recommended 65,536-slot prefix per eligible stage and tiles of at most 128
prepared candidates, flushing partial tiles without rounding the raw budget up.

## Closure and witness validation

Only the basis is initially available. Repeatedly scan base target rules, helper
target rules, then both creation rules. Append a gate only if its operands are
available and its result is absent. Stop at complete targets or a fixed point.
Availability grows monotonically, so this computes reachability for the fixed
route pair. Complete enumeration covers the declared finite family; a prefix
establishes only partial coverage. Neither is a general optimality certificate.

Production closure will execute on Metal. Host preparation and witness
validation do not supply a production CPU fallback.

Validate every returned trace before accepting any result from its dispatch:

- Check status, gate count, indices, signs, availability-mask capacity, and
  bounded work counters before shifts or array accesses.
- Replay from the basis with wide coefficient arithmetic.
- Require available operands, distinct directions, and a new result.
- Require exact membership in the selected permitted rule set; helper gates
  must equal their selected creation gates.
- Check gate arithmetic, reconstructed availability, and status/completion
  agreement.

For an unsuccessful trace, scan the entire permitted rule set once:

```text
for rule in base_rules + helper_target_rules + selected_creation_rules:
    reject if both operands are available and the result is unavailable
```

This scan never appends a gate or iterates. If the replayed set is `S` and the
least reachable closure is `L`, valid replay gives `S subset L`, while containing
the basis and being closed gives `L subset S`. Hence `S=L`, so a missing target
proves failure for those routes.

A successful trace needs complete targets and valid construction, but need not
be a fixed point. Trim dead helper gates backwards, preserve topological order,
reconstruct output aliases/signs, and count every emitted gate. Record proposed
and live helper counts separately. The trimmed forward count is `m+k`, where
`k` is the number of live helpers.

Malformed output in any dispatched lane, including a tail after an earlier
success, fails the invocation. It cannot become no-improvement or exhaustion.
Milestone 1 tests the host validation seam; the actual GPU batch acceptance
boundary must be qualified in Milestone 2.

## Capacities and arithmetic

| Quantity | Capacity or bound |
|---|---:|
| Inputs / output occurrences | 9 / 23 |
| Basis and target directions | 32 |
| Directions with both helpers | 34 |
| Availability mask | 64 bits |
| Forward gates | 25 |
| Base target rules | 992 |
| Helper target rules | 130 |
| Total rules including creation | 1,124 |
| Productive plus final sweeps | 26 |
| GPU rule checks per candidate | 29,224 |
| Negative-validation rule checks per witness | 1,124 |

The coefficient admission bound is `M <= floor(INT32_MAX/4)`, with
`M = max(1, max(abs(D)))`. First helpers are bounded by `2M`, dependent second
helpers by `3M`, and helper-pair calculations by `5M`. Use `int64` for sums,
differences, normalization, and replay; range-check stored helpers before
converting to `int32`. Because every operand is an admitted `int32` value, these
wide pair operations cannot overflow `int64`. The kernel will consume rules
and masks without computing coefficient vectors.

The raw and filtered work bounds are not latency estimates. A 65,536-slot prefix
permits at most 1,915,224,064 GPU rule checks, plus separate host validation work.

Shared structs are separate from the old constructor types. Exact ABI and
padded-dispatch allocation checks belong to GPU integration. Account for
resident generation buffers, per-thread scratch, host preparation, verification,
serialization, pools, indexes, and transactions simultaneously. Old static
allocation assertions do not prove that the new phase fits.

## Later evaluator integration

Milestone 2 will add an optional `constructor` object under reduction settings:

```json
{"family":"signed-two-aux-distinct-v1","max_pair_slots":65536}
```

The option requires existing fixed-factor signed 3x3 rank-23 `combined`
evaluation. Omission preserves the existing evaluator. Keep its baseline and
zero-/one-helper passes. Escalate each stage only after smaller-family exhaustion
and only if its incumbent exceeds the applicable family floor. Do not screen on
heuristic total cost or on other stages' current costs.

After smaller-family exhaustion, both helpers in a successful witness must be
live, so U/V cost exactly `m+2`. Transpose the actual W-transpose DAG and verify
the resulting W coefficients and operation count. A trimmed forward DAG with
`g` gates, `q` nonzero singleton output references and `a` live inputs transposes
to `g+q-a` operations under the literal transposer. All nine inputs are essential
for production tensor-valid W maps; the eligible family cost is `m+2+q-9`.
These conditions justify first-success stopping in production. Generalized test
maps with inactive inputs require actual cost comparison unless a bound is met.

Replace a stage only after exact verification and only on strict improvement.
Incumbents win ties. Full family failure, a completed raw prefix, a proved
no-improvement bound, and infrastructure/numeric/witness failure are distinct.
Retain exact ordered-factor, tensor, count, work, coverage, stop-reason, timing,
and build provenance. Changed evaluator settings require fresh search histories.

## Calibration and acceptance gates

Milestone 1 requires vector-derived independent route and reachable-set tests,
including independent/dependent helpers, both-helper target production,
route-preserving duplicates, symmetry, cycles, aliases, zeros, nonternary
arithmetic, capacities, budgets, and negative fixed-point corruption cases.

The supplied Sun V witness uses helpers `e2+e5` and `e3-e5`, with 11 target
directions and 13 live gates. It is contained at raw slot 32,877, so a prefix of
32,878 includes it. Its full raw family has 159,600 slots and 75,342 valid route
pairs after the defined filters. Production must receive factors, never replay
the reference program.

Milestone 2 requires factors-only Sun V13 and complete Sun U13/V13/W30=56,
preserved CN122 U13/V14/W28=55, native and independent exact verification,
GPU/oracle agreement, all negative/batch-tail checks, and packaging checks.
Milestone 3 integrates the shared evaluator with native additive search and
qualifies history, transaction, and resident-memory boundaries.

Milestone 4 requires the separately approved measurement protocol. Keep Sun and
CN122 as calibration, Original and Laderman as skip controls, E4 as four eligible
schemes, and U8 as eight candidates frozen before eligibility screening. Acquire
and freeze the missing public identities before examining two-helper outcomes.
Report conditional E4 benefit separately from unconditional U8 workload benefit.
No correctness or calibration gate establishes broad effectiveness or attainment
of the research objective of at most 54 additions.

## Branches, review, and resource policy

Deliver each milestone as a separate PR targeting `main`. Use branches such as
`dev/two-aux-construction-m1`, created from freshly fetched `origin/main` after
the preceding milestone's checks, review, and approved merge. Preserve existing
work and reconcile relevant upstream changes. Smaller PRs must state dependencies
and remaining milestone requirements. Never commit or push directly to `main`.

Each PR identifies its scope, base and reviewed head commits, required exit
evidence, test results, and outstanding limitations. Request review and stop at
the milestone boundary. Merging requires explicit user approval; starting the
next milestone requires the preceding approved merge. PR approval does not
authorize a campaign or replace qualification and measurement gates.

New supervised attempts use an authorized 4-GiB sampled system wired-memory
cutoff (4,294,967,296 bytes), with 45-second children and the existing cleanup
procedure. Preserve the 1,946,157,056-byte prelaunch ceiling; the corresponding
reserve under the new cutoff is 2,348,810,240 bytes. Record and explicitly pass
the effective policy to all children. Preserve historical protocols and evidence
with their recorded limits. This cutoff is separate from allocation, record,
transaction, and read budgets. Wiring and testing the prospective policy is
required before Milestone 2 GPU qualification.
