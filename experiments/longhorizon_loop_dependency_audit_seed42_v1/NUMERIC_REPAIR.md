# Development-triggered cache-layout control

The original 2x2 protocol and all its outputs are retained. The locked manifest,
threshold rule, anchor definition and selected pairs do not change.

Development data showed a nonzero `U22_mask - U12`, although the masked source
deep KV is inaccessible in both conditions. Its mean was -0.00409 nats on
NESTFUL and -0.00791 nats on controlled chains. Both source-K2/consumer-K1 nulls
were exactly zero. The cache layout changes SDPA tensor dimensions between
source K1 and source K2; BF16 calculation is not invariant to those shapes.
This makes small unaligned effects insufficiently reliable.

The supplemental `dev_aligned`, `audit_aligned` and `rename_aligned` phases add
one all-zero **invalid** deep cache slot at the source position in K1 branches.
It has the same position ID, row ordering and shape as K2's masked entry.
It adds no readable state, performs no recurrent computation and contains no
future token information. All other slots, token IDs, depth decisions and the
native probability mixture are unchanged. Extra-iteration budgets still count
actual model loops, not dummy cache storage.

The complete original sample/pair plan is repeated, rather than selecting
positive effects or excluding examples with large noise. The same previously
frozen local-gain threshold is reused. Original and aligned analyses are saved
under distinct phase names; neither replaces the other. Aligned mask-equivalent
branches and development repeated branches provide a strict empirical null.
An aligned nonzero source-K2/consumer-K1 effect or persistent large masked-null
would invalidate the mechanism interpretation.

This is an engineering control, not a new allocation algorithm or model. It
cannot establish official serving-engine equivalence, semantic dependency
specificity, or free-generation gains by itself. Those remain separate gates.
