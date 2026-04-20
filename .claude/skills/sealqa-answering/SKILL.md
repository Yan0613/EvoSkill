---
name: sealqa-answering
description: Answer SealQA short-answer questions using web search. Covers search strategy, answer extraction, temporal sensitivity, and NOT_ATTEMPTED cases.
---

## Goal
Return the shortest complete answer span inside `<FINAL_ANSWER>...</FINAL_ANSWER>` tags.
No explanation, no preamble, no punctuation beyond what the answer itself requires.

## Search Strategy

1. **First search**: use the key entity + constraint from the question.
   - Example: "Grammy all-time record most album of the year wins" not "Grammy awards history"
2. **If 0 results or unhelpful**: broaden or rephrase — drop one qualifier, try an alternate name.
3. **WebFetch** on the most promising URL (Wikipedia, official stats sites) to get exact numbers or lists.
4. **Cross-check** conflicting results with a second search before finalising.

## Answer Extraction Rules

- **Counts/quantities**: give the integer only (e.g. "11", not "11 players").
- **Dates**: use the format asked by the question; if no format given, use the most natural short form.
- **Names**: use the canonical name from the source (e.g. "UnionPay" not "China UnionPay").
- **Fractions / percentages**: match the precision in the question or source.
- **"No one" / "none" answers**: if confirmed by multiple sources that no entity meets the criterion, answer "No one" or "None".

## Temporal Sensitivity

SealQA questions often ask about the **current** state of a record, ranking, or count.
- Always check the **publication date** of sources — prefer sources dated 2024–2026.
- For "most recent" or "current" questions, verify the answer reflects 2025/2026 state.
- If sources conflict, use the most recent credible one.

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Returning a full sentence instead of a span | Strip to the answer entity only |
| Using training knowledge without web verification | Always search first |
| Stopping at the first search result | Check at least one source URL |
| Confusing "since YEAR" (inclusive) with "after YEAR" | Re-read the question constraint |
| Off-by-one counts | Count items in the listed source, do not estimate |

## NOT_ATTEMPTED vs INCORRECT

If after two searches you cannot find a credible answer:
- Return `<FINAL_ANSWER>I don't know</FINAL_ANSWER>` — the LLM judge scores NOT_ATTEMPTED at 0 but avoids a confidently wrong answer penalty.
- Do **not** guess from training knowledge.
