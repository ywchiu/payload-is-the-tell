# Erratum (11 November 2026)

This erratum concerns the established-brand control in the submitted report (§4.3, "Controls", and the abstract). The submitted PDF is left unchanged in this repository; the corrected numbers are below.

## What was wrong

1. **The mention count for 可不可 was inflated.**
   - The report says the brand was chosen as "the most discussed, 703 mentions". That number came from a substring count.
   - 632 of the 703 matches are the everyday phrase 「可不可以」 ("can you / may I"), not the brand.
   - The corrected count is **71 mentions** in the development window and 67 in the held-out window.
   - With the corrected count, 可不可 is not the most discussed chain: 50嵐/五十嵐 has 277 mentions. It is still the more-discussed of the two brands we tested; 迷客夏 has 46.
2. **The same substring problem affected PayloadConv's "unprompted" gate.**
   - The gate decides whether a thread's root post already names the endorsed brand, using a substring match on the root text.
   - Roots containing 「可不可以」 were therefore treated as naming 可不可, so swarm payload replies in those threads were wrongly excused as "prompted".

## Fix

`src/detect.py` now masks a short list of everyday words that contain a brand alias (`COLLIDE = ("可不可以", "都可以")`) before the root-text match. Setting `COLLIDE = ()` reproduces the submitted behaviour.

`src/erratum.py` recomputes every population-level detector (D4, D4a, D4b, D4c) and the frozen 0.5% alarm with both versions of the gate, in both windows, for all conditions (S0–S5, both established brands, and the Gemini-generated swarms). No LLM calls are involved; extractions are unchanged. The old-gate numbers reproduce the submitted results exactly.

## What changed

**Only the 可不可 condition (S4k) changes.** Every other number in the report is identical to three decimals, including:
- the alarm thresholds;
- the main table;
- the held-out results;
- the cross-family results;
- the 迷客夏 control.

PayloadConv on the 可不可 swarm (means over 3 seeds):

| | Submitted | Corrected |
|---|---|---|
| Development: AUROC / AP / recall@30 | 0.91 / 0.49 / 0.60 | **0.95 / 0.56 / 0.60** |
| Held-out: AUROC / AP / recall@30 | 0.84 / 0.68 / 0.67 | **0.91 / 0.81 / 0.80** |
| Frozen alarm, development: recall / precision | 81% / 0.67 | **90% / 0.69** |
| Frozen alarm, held-out: recall / precision | 62% / 0.90 | **80% / 0.92** |

**Corrected statements:**
- **Abstract:** "Promoting an established brand lowers AP to **0.56**–0.88" (submitted: 0.49–0.88).
- **§4.3:**
  - Development AP falls from 0.91 to **0.56–0.61** (submitted: 0.49–0.61).
  - Held-out AP falls from 0.93 to **0.81–0.88** (submitted: 0.68–0.88).
  - The frozen alarm flags **80–90%** of established-brand swarm accounts (submitted: 62–90%). Precision is unchanged at 0.67–0.93.
- **The explanation of the drop changes.**
  - The submitted report said 22–29% of 可不可 payload replies land in threads whose root already names the brand.
  - With the corrected match, this is **0–7%**. Most of the earlier figure came from roots containing 「可不可以」.
  - The remaining drop is therefore mainly due to weaker extraction of established-brand payloads: the extractor labels them as positive endorsements in 63–73% of messages, against 84–92% for the fictional brand. That figure is unchanged.

**Conclusions.** The qualitative conclusions do not change: PayloadConv "weakens but survives" when the swarm promotes an established brand. The correction makes the established-brand result stronger, not weaker.

## Also note

- **Blind audit (Appendix D).** It used a substring check for "the extracted name appears in the root text" when comparing with Gemini. That comparison covers organic positive endorsements, not the 可不可 payloads, and we have not re-run it.
- **Brand selection.** The brands were chosen by the inflated count. 可不可 would still have been selected as a well-known chain, but not as "the most discussed".

Found while preparing follow-up work, by checking entity counts against the claim extractor.
