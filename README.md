# The Payload Is the Tell

**Can platforms still catch coordinated fake accounts when those accounts are run by AI agents?**
Mostly not with today's signals, but there is one thing an AI swarm cannot hide: *what it is trying to sell.*

📄 Report: [PayloadIsTheTell_ApartSwarm2026.pdf](PayloadIsTheTell_ApartSwarm2026.pdf)
· Apart Research Swarm Alignment Hackathon, November 2026
· Track: Multi-Agent Oversight & Control

---

## Why: the problem

Fake-account campaigns ("coordinated inauthentic behaviour") are usually caught because human troll farms are lazy. They copy-paste the same text, post at the same moments, and keep showing up together.

AI agents don't have to be lazy. In 2026, Taiwanese Threads users found political attack comments that still contained the AI generator's own instructions ("Analyze the Persona & Constraints", "Platform: Threads"). Swarms like this are no longer hypothetical.

We asked a simple question: **if the fake accounts are LLM agents, which of the classic detection signals still work, and what should we watch instead?**

## Who, when, where

- **Who:** Yu-Wei Chiu, LargitData Inc. (Taipei, Taiwan).
- **When:** built during the Apart Research Swarm Alignment Hackathon, 6–9 November 2026.
- **Where:** real Taiwanese social media from October 2026: 33.7k messages from 25k accounts on Threads, Facebook pages, PTT and Mobile01.

## What we did

1. **Built a realistic sandbox.** We took real conversations and mixed in 30 AI-run accounts. Each account copies real users' platform and posting volume. The swarm secretly promotes a **fictional** bubble-tea brand. Everything stays offline; nothing was ever posted.
2. **Made the swarm smarter step by step (six levels):**
   - template bots;
   - a naive AI swarm;
   - agents with distinct personas;
   - a coordinated campaign with roles;
   - a swarm told how detectors work and asked to evade them;
   - an adaptive swarm that opens its own discussion threads.
3. **Tested the standard detectors** against each level:
   - copy-paste similarity;
   - synchronised timing;
   - topic clustering;
   - an AI monitor that reads one account at a time.
4. **Proposed a new detector, PayloadConv.**

## How: the method

The key observation is that an AI swarm can vary its wording, timing, personas and writing style freely. It cannot vary its goal: it has to keep pushing the same product, candidate or story.

PayloadConv:

1. **Reads every message and asks "who is being endorsed?"** An LLM extracts the brand, store, person or product each message recommends.
2. **Looks for unprompted endorsements.** It counts praise that appears in threads where nobody asked about that entity.
3. **Looks at the whole population, not single accounts.** It flags entities pushed by many different accounts:
   - in the same uniformly positive way;
   - unprompted;
   - spread across unrelated threads.

   It then scores each account by how much it takes part in such a push.

## Results

| | What happened |
|---|---|
| Classic detectors (copy-paste, timing, topic) | Once agents have personas, swarm accounts look **less** suspicious than real users (AUROC 0.14–0.48, at or below a coin flip). |
| AI monitor reading one account at a time | Gets weaker as the swarm gets smarter (0.86 → 0.74), and is never precise. |
| **PayloadConv** | Finds the swarm with high precision: average precision 0.69–0.99 when the swarm is just 1% of accounts. |
| Repeated on a later week, with every setting frozen | Still works (average precision 0.88–1.00). |
| Swarm written by a different AI model (Gemini) | Still works (average precision 0.71–0.98). |
| Swarm promotes a real, already-popular brand | Weaker but still far ahead of every other method (0.49–0.88 vs ≤ 0.07). |
| Fixed alarm (≤ 0.5% of real users flagged) | Catches 84–100% of non-adaptive swarm accounts. |
| **Adaptive swarm that opens its own threads** | **Defeats it** (back to chance). A provenance-aware variant recovers the ranking, but not yet the precision. |

We also found that **evasion leaves its own fingerprint.** When every agent is told to "add a small complaint to look neutral", most of them complain about the same thing: waiting (long queues, slow service).

## Benefits: why this matters

- **For platforms:** don't tune fake-account detection only on human troll farms. AI swarms can look *more* normal than real users on those signals.
- **For AI oversight:** watching agents one at a time misses the problem. The evidence shows up only at the population level, in the shared goal.
- **Practical:** the extraction step is cheap. One shared GPU processed 48k messages in under an hour.
- **For the arms race:** we show the next attacker move (owning the context by opening threads) and the new, smaller surface it creates for defenders.

## Conclusion

**The payload is the tell.** An AI swarm can disguise how it talks, but not what it wants. Monitoring *what is being pushed, by how many accounts, and how unprompted* catches swarms that slip past every classic signal.

It is not a silver bullet. A swarm that manufactures its own context can evade it, and political supporters are its hardest false positives. That is the next problem to solve.

## Limitations and responsible release

- **Ground truth:** the swarms are synthetic. Real accounts are assumed benign.
- **Models, not humans:** the claim extractor was audited by a second AI model (Gemini), not by human annotators.
- **Not released:**
  - raw social-media data;
  - account IDs;
  - the swarm's detector-evasion instructions (available to verified researchers on request).
- **Dual use:** the swarms promoted a fictional brand, and nothing was ever posted.

## Reproducing

See [REPRODUCE.md](REPRODUCE.md) for the pipeline and file overview.

## Disclosure

The data source is InfoMiner, a social-listening product of LargitData Inc., the author's company. The author has a commercial interest in coordination-detection products.
