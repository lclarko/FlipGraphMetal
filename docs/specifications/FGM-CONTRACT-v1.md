# FGM-CONTRACT-v1


This appendix is normative for the new interfaces and `controlled-v1`. A must retain its exact version and content digest with the reference and golden traces.

### N1. Selector encoding and order

Seeded selection ranks source presentations using SHA-256 of:

```text
ASCII "FGMSELECT\0"                       10 bytes, one terminal NUL
uint16 little-endian                     1
uint64 little-endian                     selection seed
LP32 UTF-8                              "presentation"
LP32 UTF-8                              collection namespace
LP32 UTF-8                              source presentation ID
```

`LP32` is an unsigned 32-bit little-endian byte length followed by exactly those bytes, without a terminator.

Strings are case-sensitive UTF-8, with no Unicode normalization. The namespace is the manifest’s declared namespace, not a filesystem path. Snapshot and source hashes remain bound by the selection receipt.

Sort by the 32 SHA-256 bytes in ascending unsigned lexicographic order, then by source-ID UTF-8 bytes ascending. Conflicting duplicate source IDs are errors.

Apply filters before selecting the first K records. Manifest enumeration order has no effect. Explicit-ID selection preserves its declared order and does not use seeded ranking.

### N2. Canonical scheme identity

For a signed term, form these four flattened coefficient tuples:

```text
(U, V, W)
(-U, -V, W)
(-U, V, -W)
(U, -V, -W)
```

Choose the **numeric lexicographically greatest** tuple using `-1 < 0 < 1`. Compare coefficients numerically, not as unsigned encoded bytes.

For F2, use the original tuple without sign normalization.

Sort complete resulting terms in **ascending numeric lexicographic order**. Preserve zero terms, duplicates and multiplicity. Do not cancel or combine terms.

Encode:

```text
ASCII "FGMSCHEME\0"                      10 bytes
uint16 little-endian                     version = 1
uint8                                   domain: 1 = ZT, 2 = F2
uint8                                   orientation = 1
uint32 little-endian                     n1
uint32 little-endian                     n2
uint32 little-endian                     n3
uint32 little-endian                     rank
coefficient bytes                       sorted terms, U then V then W
```

Orientation 1 is cyclic-W:

```text
U[i*n2 + k]
V[k*n3 + j]
W[j*n1 + i]
```

Each term contains `n1*n2`, `n2*n3` and `n3*n1` coefficients respectively. Encode signed coefficients as one byte: `-1 = ff`, `0 = 00`, `1 = 01`. F2 permits only `00` and `01`.

There is no padding, implicit terminator, per-term length, JSON content or provenance in this encoding. Check all widths, products and lengths for overflow.

The identifier is `fgm-scheme-v1:` plus lowercase SHA-256.

Explicitly convert source orientation before this encoding and retain that conversion in provenance. Identity computation does not change live term order or signs.

### N3. Ordered-factor encoding

Use the N2 field order and coefficient encoding, but:

```text
ASCII "FGMFACTORS\0"                     11 bytes
```

Do not normalize signs or sort terms. Encode the specified ordered factors in cyclic-W orientation.

Label the result `fgm-factors-v1:` plus lowercase SHA-256.

Original source formatting/orientation remains bound by source hashes and adapter records. Do not encode row-major W bytes while declaring orientation 1.

### N4. Canonical known-answer vectors

The table gives the SHA-256 component of the canonical ID.

| Input | SHA-256 |
|---|---|
| Signed scalar, one term `(1,1,1)` | `ba7df0b9cf2c12c6511161967cc2606abd4249607645d6d591ff606a26846765` |
| Same scalar over F2 | `efd4dd3be39f198b2b7145c01bcd7f4344184247fd8725d6d00ec5485fd9c0a6` |
| Signed scalar plus zero term `(0,1,1)` | `42f0021f81254bf76637ad6bbb8563d32ff33ecfb126218cabd59ac6a31026b7` |
| Signed scalar terms `(1,1,1)`, `(1,1,1)`, `(-1,1,1)` | `19d6ac0cb279bba477f23b0e39fbd9204e69af4015b0244ca5abf62f5367d468` |
| Signed schoolbook 2×3×4 | `0dd72c490cf1f6504bc79171b9b19b1ee4cfd931262bf22f46267cb4ce0ec5d6` |
| F2 schoolbook 2×3×4 | `1724d0275f23a619be93f7317a1bd8b3879fda070b2fc8bfc3b5e23ad81d34c5` |

For the rectangular vectors, dimensions are `(2,3,4)`, rank is 24, and every `(i,k,j)` contributes unit factors at `U[i*3+k]`, `V[k*4+j]` and `W[j*2+i]`.

Gauge/order variants preserve canonical identity; changed multiplicity does not.

Unknown identity versions fail. Future versions require explicit migration with retained old-to-new mappings.

### N5. Worker and host RNG

The new run seed is an explicit unsigned 32-bit integer. Zero is a deterministic seed in the new policy. Legacy seed-zero behavior remains unchanged.

Worker i starts with:

```text
state = uint32(seed) XOR uint32(0x9e3779b9 * uint32(i+1))
if state == 0:
    state = 1
```

Worker advancement is:

```text
x = state
x ^= uint32(x << 13)
x ^= x >> 17
x ^= uint32(x << 5)
state = x
return x
```

All worker arithmetic wraps modulo 2³². Right shift is unsigned.

GPU bounded choice consumes one word and returns `word % bound`. Bound must be positive. It consumes a word even for bound one unless a specified blocked-event rule prevents reaching the draw.

Inclusive interval selection is:

```text
minimum + word % (maximum - minimum + 1)
```

Use checked wider arithmetic for the span; never narrow a full-width span to zero.

Permutations initialize identity order and run descending Fisher–Yates from `n-1` through 1. Length zero or one consumes no words.

The host uses a separate `std::mt19937_64`, initialized directly from the zero-extended run seed. Do not use `seed_seq`, time or a derived seed. Host zero needs no repair.

For host bound b:

```text
threshold = 2^64 mod b
draw x
reject while x < threshold
return x % b
```

Allow at most 64 draws per selection; exhaustion is an error. Check bounds and weight sums for overflow.

Weighted selection draws from the total integer weight and chooses the first cumulative weight strictly greater than the sample. All-zero weights use uniform selection. A one-parent selection still consumes the prescribed draw.

Host and worker RNG streams continue across restarts and stages. A resumed execution initializes new recorded streams; it does not claim exact RNG continuation.

### N6. Reduction threshold and flip draws

The authoritative controlled reduction setting is unsigned 32-bit threshold q. When the decision is reached:

```text
word = next()
apply explicit reduction iff word <= q
```

Consume the draw for q=0 and q=2³²−1. Nonzero xorshift state makes q=0 disabled. Do not substitute legacy floating-point probability comparison.

Pinned arithmetic is the implementation at `9ea5bfc144b528184c32c178cae0b49bc60e8e3d`, including the declared branch order in `scheme_integer.h` and `scheme_z2.h`. This pins arithmetic without importing legacy controller or rejection-loop behavior.

Signed flip selection:

1. Concatenate candidates in factor-list order 0, 1, 2.
2. Fully shuffle candidate indices.
3. For each visited candidate, draw factor-order `%2`, then term-order `%2`.
4. Evaluate the pinned signed branches in order.
5. Apply the first admissible flip and its existing automatic reductions.

Deterministic orientation correction consumes no extra RNG. Do not substitute lazy selection.

F2 flip selection:

1. With no candidates, return unsuccessful without draws.
2. Draw candidate `%size`.
3. Draw factor-order `%2`.
4. Draw term-order `%2`.
5. Apply pinned arithmetic.

### N7. Expansion ceiling, proposals and draw consumption

Define the excursion anchor and effective expansion ceiling as:

```text
anchor =
    stage rank R             for rank reduction
    fixed collection rank T  for alternatives

ceiling = min(
    checked_add(anchor, excursion),
    checked_multiply(n1, n2, n3),
    representation_rank_capacity
)
```

Use checked arithmetic for the anchor sum, dimension product and rank increment. Overflow produces an explicit configuration or execution-eligibility error; do not wrap or silently clamp intermediate arithmetic. Such an error does not establish mathematical invalidity.

Before **every expansion primitive**, including each primitive of a recovery event, require:

```text
checked_add(current_rank, 1) <= ceiling
```

Reevaluate this condition against the current rank after any preceding primitive. An applied first recovery primitive can therefore make the second primitive rank-blocked.

An imported scheme above this ceiling is not mathematically invalid. Subject to the separate representation and candidate-capacity checks, admit it at its verified rank without eager reduction. Expansion remains blocked while `current_rank + 1 > ceiling`; other permitted operations may lower its rank.

A fully blocked event consumes no recovery-count, operator or proposal words.

An eligible scheduled event requests one primitive. An eligible recovery event first draws `1 + word % 2`.

Before each primitive, recheck the ceiling. Then draw one operator:

```text
word % 3:
0 = plus
1 = random split
2 = existing-factor split
```

Choose it once for that primitive. Allow at most the configured number of complete proposals, default 64.

Rejected proposals advance RNG but do not mutate live coefficients or candidates. Retry the entire proposal for the same operator. No existence pre-scan may bypass the specified draws in controlled helpers; legacy prechecks remain unchanged.

| Proposal | Draw and rejection order |
|---|---|
| Signed/F2 plus | Draw two term indices. Reject equal indices or any equal corresponding factor. Otherwise draw a three-factor permutation and variant `%3`, then evaluate pinned arithmetic. |
| Signed random split | Draw term index, factor permutation, then factor coefficients. Apply the pinned coefficient/orientation check. |
| F2 random split | Draw term index, factor permutation, low coefficient word and high word only if width exceeds 32. Mask to width; reject if equal to the selected factor. |
| Signed existing split | Draw two indices, then factor `%3`. Reject equal indices/factors or the pinned subtraction/orientation failure. Other factor positions are cyclic. |
| F2 existing split | Draw factor permutation, then two indices. Reject equal indices or equal factors in the selected position. |

Signed random factors consume a sign word then value word for each 16-entry chunk, use their low 16 bits, mask unused entries and clear signs on zero coefficients.

F2 random splitting does not add a nonzero-mask restriction. A zero mask is permitted when the pinned admissibility rule permits it.

Signed plus retains its pinned fallback branch behavior. If no arithmetic branch applies, report coefficient rejection without claiming success or rebuilding live candidates.

Exhaustion returns `proposal_exhausted`; there is no fallback scan. An applied first recovery primitive remains applied if the second rejects or is blocked.

A completed blocked/exhausted expansion event still reaches the specified interval redraw unless an earlier error, target or lifetime-stop rule terminates processing. Recovery with zero applied primitives requests restart; scheduled rejection continues.

### N8. Initialization, controlled-step ordering and outcomes

Starting parents are validated and checked in deterministic input order before worker initialization or RNG consumption. A satisfied parent target commits existing evidence without discovery credit. It does not initialize a worker or consume a first-countdown draw.

For each worker proceeding into execution:

1. Complete starting-parent verification and target preflight.
2. Initialize its RNG using N5.
3. Initialize its first expansion countdown using exactly one inclusive interval draw.
4. Enter the controlled-step loop; no flip attempt precedes that draw.

The first countdown is:

```text
span = checked_add(
    checked_subtract(maximum_interval, minimum_interval),
    1
)

word = next()
expansion_countdown = minimum_interval + word % span
```

Validate interval ordering and use checked wider arithmetic for the span and result. Consume the word even when `minimum_interval == maximum_interval`.

Ordinary batch boundaries preserve the countdown and RNG state. They do not repeat initialization or draw a replacement interval.

Completed expansion events and restart installation redraw the interval at the points specified below. A restart continues the existing worker RNG; it does not reseed or perform an additional first-countdown initialization.

Restart parents are checked after prior mandatory commitment and host selection, but before installation. Already-consumed host selection draws remain consumed. A restart parent that already satisfies the target commits existing evidence without installation or an installation-countdown draw.

For each active step:

1. Check terminal state and lifetime allowances. Return pending restart requests without another flip.
2. Charge one control step and flip attempt; decrement expansion countdown and increment stagnation.
3. Attempt the flip.
4. Observe errors, best state, mandatory capture, optional capture and rank target, in that order.
5. Stop if lifetime allowance is exhausted.
6. After successful flip, consume the reduction decision; apply at most one explicit reduction and observe its outcome.
7. After unsuccessful flip, perform recovery. Otherwise perform scheduled expansion if due. Recovery replaces a coincident scheduled event.
8. Observe each applied primitive before attempting another.
9. After a completed expansion event, redraw its interval.
10. Request restart after unsuccessful recovery or due stagnation; otherwise return at the batch boundary or continue.

An observation updates best state only on strict rank improvement and resets stagnation on that improvement. A flip plus its internal automatic reductions is one observable operation, with internal removal counters retained.

Capture follows applied observable operations. Mandatory storage retains the lowest encountered rank in the batch, with the earliest equal-rank encounter. It is independent of optional quota.

Optional encounters follow mode eligibility: collection rank T for alternatives and lower-than-stage ranks for rank reduction. Retain them in encounter order without GPU identity deduplication; count quota drops. Capture consumes no walk RNG. Host identity deduplication occurs after verification.

Initialization is not a new discovery. Restart cannot clear uncommitted mandatory evidence.

A target reached on the last allowed attempt is captured before termination, with no later reduction, expansion or RNG draw. GPU target flags remain provisional until the whole dispatch succeeds and evidence verifies and commits.

Parent installation consumes control credit. Stage advancement consumes a credit from the lowest-ID worker with credit remaining; installations are charged separately. Neither replenishes lifetime allowances.

Restart resets current/best to its parent, resets stagnation and samples a new interval. RNG streams and lifetime counters continue.

Distinguish at least these outcomes in traces: applied, unsuccessful flip, tuple rejection, coefficient rejection, proposal exhaustion, rank blocked, capacity error, restart requested, target pending and budget exhausted.

### N9. Golden-trace requirements

Golden traces bind specification version/digest, fixture bytes, domain, configuration, arithmetic reference and initial states.

Record:

- Draw sequence and pre/post worker and host RNG states.
- Ordered coefficients and candidate lists.
- Current/best rank and state.
- Timers and remaining budgets.
- Attempted/applied operations and rejection outcomes.
- Reduction attempts and removed terms.
- Mandatory/optional captures and drops.
- Restart, target, overflow and termination flags.
- Deterministic host admission and commitment results.

Exclude nondeterministic timestamps from equality comparisons, while retaining them in execution receipts.

Required traces cover zero repair, bound-one draws, empty candidates, q endpoints, rejection/exhaustion, blocked events, one/two recovery primitives, F2 widths 33/64, target/budget collisions, zero/full optional quota, restart and mandatory-evidence survival.

Expansion-ceiling cases must additionally cover:

- Both anchors: stage rank R and fixed collection rank T.
- Each ceiling component being the minimum: anchor plus excursion, naive rank and representation capacity.
- A mathematically valid imported scheme above the expansion ceiling, admitted without eager reduction.
- `current_rank == ceiling - 1`, allowing a primitive.
- `current_rank == ceiling` and `current_rank > ceiling`, blocking a primitive.
- A first recovery primitive reaching the ceiling, followed by a blocked second primitive with no second operator/proposal draws.
- Checked-arithmetic failures in anchor addition, dimension multiplication and rank increment.

Countdown and initialization cases must additionally cover:

- The first interval draw occurring after worker RNG initialization and before the first flip.
- Unequal interval endpoints and equal endpoints, including consumption of the bound-one draw.
- An already-satisfied starting target with no legal flips and optional quota zero: existing evidence commits, with no worker initialization or countdown draw.
- Ordinary batch boundaries preserving the countdown without an extra initialization draw.
- Completed expansion events consuming their prescribed interval draw.
- Restart installation consuming its prescribed interval draw from the continuing RNG, without reseeding or duplicate initialization.
- A restart parent already satisfying the target: prior host selection draws remain consumed, but no installation-countdown draw occurs.

Expansion-ceiling cases must additionally cover:

- Both anchors: stage rank R and fixed collection rank T.
- Each ceiling component being the minimum: anchor plus excursion, naive rank and representation capacity.
- A mathematically valid imported scheme above the expansion ceiling, admitted without eager reduction.
- `current_rank == ceiling - 1`, allowing a primitive.
- `current_rank == ceiling` and `current_rank > ceiling`, blocking a primitive.
- A first recovery primitive reaching the ceiling, followed by a blocked second primitive with no second operator/proposal draws.
- Checked-arithmetic failures in anchor addition, dimension multiplication and rank increment.

Countdown and initialization cases must additionally cover:

- The first interval draw occurring after worker RNG initialization and before the first flip.
- Unequal interval endpoints and equal endpoints, including consumption of the bound-one draw.
- An already-satisfied starting target with no legal flips and optional quota zero: existing evidence commits, with no worker initialization or countdown draw.
- Ordinary batch boundaries preserving the countdown without an extra initialization draw.
- Completed expansion events consuming their prescribed interval draw.
- Restart installation consuming its prescribed interval draw from the continuing RNG, without reseeding or duplicate initialization.
- A restart parent already satisfying the target: prior host selection draws remain consumed, but no installation-countdown draw occurs.

A freezes the reference and traces. B validates general execution against them. E validates packed execution against the same contract and traces. Optimized code is never its own sole oracle.
