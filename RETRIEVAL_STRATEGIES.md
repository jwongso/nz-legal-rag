# How the Search Works - Four Different Strategies Explained

*Written for anyone curious about how this tool finds relevant Tenancy Tribunal decisions.
No technical background needed.*

---

## The Big Picture

When you type a question like *"My landlord won't fix the heating - what can I do?"*, the
tool needs to search through 32,000+ real Tenancy Tribunal decisions and find the ones most
likely to help answer your question. The challenge is that none of those decisions will
contain your exact sentence. The tool has to figure out which decisions are *about the same
thing*, even when the words are different.

There are four strategies for doing this. Think of them like four different research
assistants, each with their own style.

---

## Strategy 1 - Vector Search (the default)

### The idea

This strategy teaches the computer to understand *meaning*, not just words.

Every decision in the database has been converted into a list of around 768 numbers - a
kind of mathematical fingerprint of what the decision is about. Your question gets
converted into the same kind of fingerprint. Then the tool finds the decisions whose
fingerprints are closest to yours.

This is called a "vector" because it's literally a point in a very large mathematical
space. Two decisions about similar topics end up close together in that space, even if
they use completely different words.

### A simple example

Imagine you have 5 decisions stored like this (in reality each fingerprint is 768 numbers,
but let's simplify to 2 for the drawing):

```
Decision A - bond refund after carpet wear:         [0.8, 0.2]
Decision B - landlord refused to return bond:       [0.7, 0.3]
Decision C - tenant damaged walls and windows:      [0.3, 0.9]
Decision D - heating not working, tenant withheld rent: [0.1, 0.6]
Decision E - landlord entered without notice:       [0.2, 0.1]
```

Now you ask: *"My landlord is keeping my bond even though the carpet was just old."*
Your question fingerprint comes out as [0.75, 0.25].

The tool measures the distance from your fingerprint to each decision and picks the
closest ones. Decisions A and B are very close to [0.75, 0.25] - they win. Decision E
about entry rights is far away - it drops out.

The magic is that the fingerprint understands meaning. If you write "carpet was worn out"
or "floor covering deteriorated" or "rug was old when I moved in", you still get a similar
fingerprint because the *idea* is the same.

### The reranking bonus

After retrieving the top candidates, this strategy applies one more step: a legal
authority check. It gives a small boost to decisions from higher courts (like a District
Court appeal) over decisions from a basic Tenancy Tribunal hearing, because higher-court
decisions carry more legal weight.

Think of it like a librarian who pulls the relevant books first, then puts the ones
written by the most respected experts on top.

---

## Strategy 2 - Vector (No Rerank)

### The idea

Exactly the same as Strategy 1, but without the legal authority boost at the end.

The fingerprint matching works identically. The only difference is that the final ranking
is purely based on how similar the decision is to your question - court level does not
matter.

### When does it make a difference?

When you compare Strategy 1 and Strategy 2 side by side, you often see the same decisions
appear in both lists, because most Tenancy Tribunal decisions are at the same court level
(there is no hierarchy to rerank). The difference shows up more clearly when the corpus
includes Court of Appeal decisions alongside Tribunal decisions on the same topic.

### Example

Imagine the top 5 by similarity are:

```
1. Decision A - Tribunal (similarity: 0.92)
2. Decision B - District Court appeal (similarity: 0.89)
3. Decision C - Tribunal (similarity: 0.87)
4. Decision D - Tribunal (similarity: 0.85)
5. Decision E - Tribunal (similarity: 0.84)
```

- **No rerank (Strategy 2):** keeps this exact order.
- **With rerank (Strategy 1):** Decision B gets a small boost because it is an appeal
  decision, potentially moving it to position 1.

For most tenancy questions, the result is the same. For questions touching on points of
law that have been appealed, the reranked version surfaces the authoritative ruling first.

---

## Strategy 3 - MMR (Maximal Marginal Relevance)

### The idea

MMR solves a specific problem: what if all the top-ranked decisions are basically saying
the same thing?

Imagine you ask about bond disputes. The five most similar decisions might all be about
*exactly* the same type of case - say, dirty carpets at the end of a tenancy. They are
all highly relevant, but you would learn more from reading five *different* bond dispute
scenarios than five nearly identical ones.

MMR builds the list one decision at a time, using a rule:

> Each new decision I add must be relevant to the question AND as different as possible
> from the ones I have already picked.

### A simple example

You ask: *"Can my landlord claim for damage when I move out?"*

Round 1 - pick the most similar decision overall: **Decision A** (carpet damage, 0.92)

Round 2 - pick the next best that is *also* different from A:
- Decision B (also carpet damage, similar to A): scores lower after MMR penalty
- Decision C (wall damage, quite different from A): scores higher after MMR bonus
- **Winner: Decision C**

Round 3 - pick the next best that is different from both A and C:
- Decision D (bond dispute, different angle): scores well
- **Winner: Decision D**

And so on.

### The result

MMR tends to give you a broader view of the topic - different cases, different aspects,
different landlord claims. Strategy 1 gives you the most similar cases. MMR gives you
the most *useful variety*.

The tradeoff: MMR is slightly slower (an extra similarity calculation per round) and
sometimes picks a decision that is a little less relevant than the pure top-5 would have
been, because it is deliberately trading some relevance for diversity.

---

## Strategy 4 - BM25 (Keyword Search)

### The idea

BM25 is the oldest trick in the book, and still one of the best for certain questions.
It works like a very smart version of Ctrl+F.

Every decision in the database is indexed by the words it contains (using a PostgreSQL
database, not the vector database). When you search, the tool finds decisions that contain
your search words, and ranks them by:

1. How many of your words appear in the decision
2. How rare each word is across all decisions (rare words score higher)
3. How long the decision is (shorter decisions with your terms rank higher than huge
   decisions where your terms appear only once)

This formula is called BM25 (Best Match 25 - the 25th version of a research paper from
the 1990s that is still used everywhere today).

### A simple example

You ask: *"Section 45 Residential Tenancies Act damage"*

BM25 looks through all 32,000 decisions and finds every one that contains "section",
"45", "Residential", "Tenancies", "Act", or "damage".

- "damage" appears in 94,000 chunks - very common, low score contribution
- "section 45" appears in only a few hundred chunks - rare, high score contribution
- A decision that contains "section 45" AND "damage" in the same paragraph scores very
  highly
- A decision that mentions "damage" 50 times but never mentions "section 45" scores much
  lower

### Where BM25 wins

BM25 is best when you know the exact words you are looking for:

- **Statute references:** "section 42", "section 48(2)", "RTA 1986"
- **Legal terms of art:** "fair wear and tear", "exemplary damages", "unjustified termination"
- **Case numbers or dates:** specific citations

### Where BM25 loses

BM25 is blind to meaning. If a Tribunal decision says *"the floor covering showed
reasonable deterioration consistent with the duration of the tenancy"* and you search
for *"carpet wear"*, BM25 scores it zero because neither "carpet" nor "wear" appears
in that sentence. The vector strategies would still find it because they understand the
concept.

### The two-pass trick

Because natural-language questions contain many common words ("should", "my", "landlord",
"which", "act") that appear in almost every decision, BM25 first strips those out and
tries to match only the meaningful terms. If that finds results, great. If not, it relaxes
to a looser search using only the rarest terms in your question.

---

## Side-by-Side Summary

| | Vector | Vector (No Rerank) | MMR | BM25 |
|---|---|---|---|---|
| Understands meaning | Yes | Yes | Yes | No |
| Exact word matching | No | No | No | Yes |
| Result diversity | Low | Low | High | Varies |
| Good for section references | Average | Average | Average | Excellent |
| Good for open questions | Excellent | Excellent | Good | Poor |
| Prefers authoritative courts | Yes (slightly) | No | No | No |
| Speed | Fast | Fast | Fast | Fast* |

*BM25 can be slow if your question contains very common words - the two-pass design
keeps it fast for most queries.

---

## What Affects Quality - and Where We Could Do Better

This is where your suggestions would genuinely help. Below is an honest description of
what the system does well, where it struggles, and some open questions.

### What it does well

- **Finding thematically similar cases** even when the wording is completely different
- **Citing real decisions** - every source link goes to a real Tenancy Tribunal case
- **Staying on topic** - the system is restricted to residential tenancy matters only
- **Admitting uncertainty** - if it cannot find enough relevant decisions, it says so
  rather than making things up

### Where it sometimes struggles

**1. Statute names and section numbers**

The system knows that the governing law is the Residential Tenancies Act 1986. But if
the retrieved decisions do not explicitly mention a section number, the model sometimes
invents one ("section 42(6)" may be correct, or may be a plausible-sounding hallucination).

*Question for you:* Is there a way to cross-check generated section numbers against
the actual Act text in real time? Should we add the Act itself to the database?

**2. Rare or niche topics**

If only a handful of decisions in the database deal with your specific situation (say,
damage to a tree in the garden), the top-5 results might be about vaguely related topics
(general property damage) rather than the specific issue. The answer is then built from
imperfect context.

*Question for you:* Would it help to tell the user "I only found 2 decisions closely
related to your question - here are the closest ones, but you may want to also ask
Tenancy Services directly"?

**3. Decisions that contradict each other**

Tribunals are not perfectly consistent. Two decisions on similar facts can go different
ways. The current system picks the top 5 most relevant and synthesises an answer, but
it does not flag when those 5 decisions disagree with each other.

*Question for you:* Would a warning like "note: some of the retrieved decisions reached
different conclusions on this point" be helpful or confusing?

**4. Outdated decisions**

Tenancy law has changed over the years. A highly relevant 2019 decision might describe
rules that were amended in 2021. The system retrieves by relevance, not by recency.

*Question for you:* Should recent decisions always get a small ranking bonus? Or should
the answer explicitly note when a cited decision predates a known legislative change?

**5. Repair for the BM25 hallucination problem**

When BM25 retrieves decisions that do not mention the Act explicitly, the model sometimes
fills the gap with a made-up Act name. We have anchored "Residential Tenancies Act 1986"
in the system instructions, which helps, but the model can still invent section numbers
when none appear in the retrieved text.

*Question for you:* One idea is to always include one "anchor chunk" - a short passage
from the Act itself that names sections - alongside the retrieved decisions. Would that
feel right, or would it muddy the source list?

---

## Open Ideas - What Would You Change?

If you were designing this from scratch, what would you do differently? Some starting
points to think about:

- **Different sources:** Should we include the actual Residential Tenancies Act text as
  a searchable source, not just Tribunal decisions?
- **Better questions:** Could the tool automatically rephrase vague questions into clearer
  ones before searching? (e.g. "my landlord is mean" -> "landlord obligations and tenant
  rights under RTA 1986")
- **More context from the user:** Should the tool ask a clarifying follow-up question
  before searching? ("Are you currently in the tenancy, or has it ended?")
- **Confidence signalling:** Should the answer include a plain-English confidence level
  ("I found 8 directly relevant decisions" vs "I only found 2 loosely related decisions")?
- **Comparison across cases:** Instead of just summarising what the law says, could the
  tool show you two real cases - one where the landlord won and one where the tenant won
  - and explain what made the difference?
- **User feedback loop:** The tool collects thumbs-up / thumbs-down ratings. Should
  low-rated answers trigger automatic re-retrieval with a different strategy?

No idea is too simple or too ambitious. The goal of this document is to start a
conversation.
