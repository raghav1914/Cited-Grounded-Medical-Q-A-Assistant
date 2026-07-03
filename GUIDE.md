# Cited Medical Q&A Bot — What It Is & What It Does

A plain-language guide. No code — just what the tool is, why it's built the way
it is, and what each feature does.

---

## In one paragraph

This is a medical question-and-answer assistant that **never makes things up**.
You ask a health question — or upload your own lab report — and it gives you a
clear, everyday-language answer where **every fact is backed by a trusted source
you can click and read yourself**. If it can't find a reliable source for your
question, it simply says *"I don't have a reliable source for that"* instead of
guessing. It will not diagnose you or tell you what medicine to take, and if you
describe an emergency it stops and tells you to get real help immediately.

## Why it's different from a normal chatbot

A normal chatbot answers from memory, and sometimes that memory is confidently
wrong ("hallucination"). This tool is built so that **the AI is only allowed to
talk about a small library of trusted medical fact sheets and your own uploaded
report — nothing else.** Then a *second, independent* AI pass re-reads every
sentence and throws out anything the sources don't actually support.

The guiding rule of the whole project: **the AI (Claude) is used for only two
jobs — writing the answer and fact-checking the answer.** Everything else
(reading your report, pulling out lab numbers, searching the library, spotting
emergencies, deciding when to stay silent) is done by ordinary, predictable
computer code — not the AI. That's what makes the "it doesn't make things up"
promise trustworthy rather than just a hope.

## How a question flows through it

1. **Emergency check first.** Before anything else, plain rules scan your words
   for signs of a medical emergency (chest pain, trouble breathing, self-harm,
   etc.). If found, it immediately shows a "get help now" message and stops.
2. **Search the library.** It finds the most relevant passages from the trusted
   fact sheets (and from your uploaded report, if any).
3. **Decide whether it even can answer.** If nothing in the library is a good
   match, it stops and says it has no reliable source — *before* the AI is ever
   asked. It would rather stay silent than guess.
4. **Write the answer (AI job #1).** The AI writes a short, plain answer using
   *only* those passages, and tags each sentence with the source it came from.
5. **Fact-check the answer (AI job #2).** A separate AI pass re-checks each
   sentence against only the source it cited and deletes any sentence the source
   doesn't truly back up. If nothing survives, the tool abstains.
6. **Add context and safety.** It attaches a confidence rating, a disclaimer,
   and (for reports) flags any values outside the normal range.

---

## The features, one by one

### The chat interface
Ask questions in a familiar chat box. The conversation stays on screen so you
can follow up. **Follow-up questions understand context** — if you ask "what is
creatinine?" and then "is mine high?", it remembers the topic when searching.
(Important nuance: it only reuses earlier questions to *search better*; the
actual answer is still built only from freshly retrieved sources, so old chatter
can never leak in as an unsupported "fact".)

### Cited answers with clickable sources
Every factual sentence carries a highlighted tag like `S2`. Below the answer,
each source is listed and **expands when you click it** so you can read the exact
passage the claim came from — and follow a link to the original fact sheet
(MedlinePlus, WHO, etc.). Nothing is stated without a source you can verify.

### "I don't know" instead of guessing (abstention)
If your question falls outside the trusted library, the tool tells you it has no
reliable source rather than inventing an answer. This is a feature, not a
failure — it's the core of the trust promise.

### The faithfulness fact-check
A second, independent AI pass verifies each sentence against its own cited
source and removes anything unsupported. You can open a **"Filtered out"** panel
to see exactly what was removed and why — full transparency.

### Confidence indicator
Each answer shows a **High / Medium / Low** confidence badge, worked out by
plain rules (not the AI grading itself): how strong the source match was, and
whether any sentences were dropped during fact-checking. Click **"Why?"** to see
the reasons.

### Reading-level toggle
Switch between **Standard** and **Simple**. "Simple" rewrites the answer at about
a 6th-grade reading level with shorter sentences and everyday words. It only
changes the wording — never the facts or the citations.

### Upload your own lab report
Upload a lab report (plain text, PDF, or a photo/scan) and ask about *your*
numbers, e.g. *"what is my creatinine and is it high?"* The tool reads the file,
pulls out each lab value by rule, and treats your report as its own citable
source called **"Your report"** — so the answer can cite both your specific value
and the general medical context, and it's fact-checked the same way.
*(Photos and scanned PDFs need the free Tesseract/Poppler tools installed — see
`SETUP.md`. Plain text and normal PDFs work with no extra setup.)*

### Out-of-range flagging
For uploaded reports, simple arithmetic compares each value against the reference
range printed on the report and flags it **high / low / normal**. This is a
factual comparison, never a diagnosis — and when something is out of range, the
disclaimer gets stronger, reminding you only your provider can interpret it.

### Safety guardrails
- **Emergencies:** emergency-sounding questions short-circuit straight to a
  "seek immediate care" message (self-harm gets a crisis-line number).
- **No diagnosing or prescribing:** the assistant is instructed, and verified,
  to refuse to name your condition or recommend specific medicines or doses.
- **Always a disclaimer:** every answer closes with a reminder that this is
  general information, not medical advice.

### Feedback thumbs
A 👍 / 👎 on each answer. Your rating is saved (to `data/eval/feedback.jsonl`)
and can be folded back into the project's automated test set, so real usage helps
measure and improve quality over time.

---

## What it is **not**

- **Not a doctor.** It gives general information from fact sheets; it does not
  diagnose, treat, or prescribe. Always talk to a healthcare professional.
- **Not all-knowing.** It only knows its small trusted library plus whatever you
  upload. Outside that, it deliberately says "I don't know."
- **Not an emergency service.** In an emergency it will tell you to call for
  help — it can't help itself.

## How good is it? (measured, not claimed)

The project includes an automated test harness that runs 49 labeled questions
through the whole system. On that set it scored: **100% correct on when to answer
vs. stay silent, 100% of emergencies caught, 100% of facts carrying a citation,
and 0% hallucination** (measured by a separate AI judge). These are strong
numbers on a small, clean library — not a guarantee for every possible question,
but evidence the design works as intended.

## Trying it

Open a terminal in the project folder and run:

```
.venv\Scripts\streamlit.exe run app.py
```

It opens in your browser. Type a question, or upload a report from the sidebar
and ask about your values. There's also a command-line version — see `README.md`.
