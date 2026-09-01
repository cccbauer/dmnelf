# Wednesday talk — speaker narrative

Companion to `wednesday_presentation.pptx`. This walks through the whole arc, explains the
method in plain language (so you can field questions, not just read slides), and gives
facilitation notes for the two live/interactive parts (the opening exercise and the demo).

Slides marked **[PLACEHOLDER]** in the deck need your own material inserted — nothing there is
fabricated data. Everywhere else, the numbers are real, pulled directly from the analysis
results in this repo (cited below so you can trace each one back if asked).

---

## Part 1 — Experiential intro (slides 2–3)

**Goal:** the audience *feels* their own DMN activate before you ever say the word "DMN."

Facilitation:
- Ask the room to close their eyes or soften their gaze. Say you'll do this together, in silence,
  and that you'll tell them when it's over — don't narrate the phases as they happen.
- **30 seconds**: just breathing, nothing else.
- **2.5 minutes**: "Bring to mind something unresolved — a decision you're unsure about, an
  interaction that didn't go the way you wanted, something still nagging at you. Don't try to
  solve it. Just stay with it." Keep time silently (phone, not a visible countdown).
- Bring the room back gently. Don't ask "how was that" yet — ask **"what did you notice?"** and
  actually take 2–3 answers before you explain anything.

Expect: racing thoughts, replaying the scenario, self-judgment, losing the thread of the
instruction itself. That's the point — nobody chose to ruminate, it happened automatically. That
automatic quality *is* the DMN's default behavior, and it's why the network has that name.

Bridge line: *"What you just experienced — unprompted, self-referential, looping thought — has a
name in neuroscience: the Default Mode Network. It's called 'default' because, left with nothing
else to do, that's where the brain goes."*

## Part 2 — Why this matters (slides 4–6)

Plain-language version of the argument, in order:

1. DMN activity rises when we're not engaged in an external task — mind-wandering,
   self-referential thought, autobiographical memory. You just generated some on purpose.
2. When that pattern gets stuck — hard to disengage from, looping — it's rumination, and
   rumination is a cross-cutting feature of depression and anxiety, not a niche symptom.
   *(Insert your own citations/prevalence numbers on slide 4 — left as placeholder deliberately.)*
3. Mindfulness practice is associated with reduced DMN activity and altered DMN connectivity,
   both at rest and during tasks. In effect, meditators have trained themselves to do, at will,
   something like what just happened to this room involuntarily — in reverse.
4. The catch: that training normally takes months to years before it generalizes.
5. Neurofeedback is a shortcut hypothesis: if you can *see* your own DMN/CEN balance in real
   time, you might learn to shift it far faster than through practice alone — because you get
   the same object of information (your own state) but immediate, or when you cannot introspect on it directly otherwise.

**[PLACEHOLDER — slide 7]** is where your own lab's prior/current fMRI neurofeedback results go
— this is the evidence that the scanner-based version of this actually works, which is why
scaling it is worth the effort in the first place.

## Part 3 — The scaling problem → Meir-Hasson 2014 → our method (slides 8–11)

The argument: fMRI neurofeedback works but requires a scanner — expensive, immobile, not
something you can send someone home with. EEG is cheap and wearable, but EEG doesn't measure
BOLD directly; DMN/CEN are fMRI-defined networks. So the problem becomes: **can we decode
fMRI-defined network activity from scalp EEG well enough to feed back on it?**

**[PLACEHOLDER — slides 9–10]**: Meir-Hasson et al. (2014) is the paper that first showed this is
possible in principle — insert their own summary/figures here.

**Slide 11 — our method, in plain language** (this is the part you'll get asked about, so know it
cold):

> For one electrode, split the EEG signal into 10 frequency bands (found data-drivenly per
> subject, not fixed textbook bands like alpha/beta). For each band, look at its power over
> roughly the last 12 seconds. Feed all of that — 10 bands × ~11 time-lags each — into a
> regularized linear regression (ridge regression) that predicts the fMRI signal *right now*.
> Critically, the model is allowed to pick a **different best lag for every frequency band**
> instead of assuming one fixed delay (like the ~5-second hemodynamic lag everyone assumes by
> default) — this is the "sliding time-delay" part, and it's the main methodological difference
> from a naive approach.

Why this matters for the "is this legit" question people will ask: it's fit and evaluated with
double cross-validation (an outer loop holding out blocks of time, an inner loop picking the
regression's regularization strength) — so the numbers on the next few slides are not fit-and-
report-the-same-data results.

## Part 4 — Results (slides 12–15)

Say these numbers as **correlations between predicted and actual signal**, not accuracy percentages.

- **Slide 12 (within-subject, n=19)**: for each of 19 subjects, using their own single best
  electrode, the EFP model beats both a fixed-HRF baseline and the traditional theta/alpha-ratio
  approach, across every network target. Source: `results/full/fig_efp_vs_baselines.png`,
  generated by `scripts/paper_figures.py` / `scripts/build_ppt.py`.
- **Slide 13 (LOSO — leave-one-subject-out)**: train a single general model on 18 subjects,
  predict the 19th, who was never seen. This is the harder, more honest test than slide 12 (each
  subject there had their own fitted model). Source: `results/full/efp_group_loso.csv`.
- **Slide 14 (n=17→19)**: adding the 2 recovered subjects (dmnelf002/003 — recovered via
  reconstructing missing fMRI trigger markers) only helped; 3 targets crossed into significance
  that weren't before. Mention this briefly — it's a "our sample size is still growing and the
  signal is getting cleaner, not noisier" point, useful if asked about sample size.
- **Slide 15 — the headline slide**: the DMNELF-trained fingerprint, applied with **zero
  retraining**, to a **completely different study's** participants (rtBPD) — and it still works
  for CEN and DMN, replicated across that cohort's two separate recording sessions. This is the
  slide that answers "so what, does this generalize beyond your own dataset" — say so explicitly.
  *(PDA — the CEN−DMN contrast — also replicates significantly here at n=19, in this classic
  single-best-electrode research-cap method: nf1 r=+0.080 p=0.017, nf2 r=+0.145 p=0.007. Note
  this is a DIFFERENT, separate finding from mindwear's deployed portable 12-channel decoder,
  where PDA specifically was found to have no out-of-sample validity — don't conflate the two if
  asked. The slide deliberately leads with CEN/DMN since those are the targets that transfer
  most consistently across every version of this analysis, but PDA isn't a weak point here.)*

## Part 5 — Interpretability → mindwear → demo (slides 16–19)

- **Slide 16**: the model wasn't told anything about hemodynamics, yet the learned weights peak
  at physiologically plausible 4–7 second lags. This is a good "it learned something real, not
  noise" visual if anyone's skeptical about a black-box model.
- **Slide 17–18**: the pivot from science to product. mindwear runs the identical visual
  ball-feedback paradigm, driven by the same EFP approach, restricted to a 12-channel portable
  headset instead of the 31-channel research cap. **[PLACEHOLDER — slide 18]**: drop in a
  screenshot or architecture diagram if you have one.
- **Slide 19 — live demo**: run a short session on yourself first so the room sees the ball move
  before anyone volunteers. Suggested framing while it's running: *"This is noisier than what you
  saw from the scanner data — that's expected, and it's exactly what we just spent the whole talk
  characterizing. Watch the trend over the run, not any single moment."* That framing pre-empts
  the obvious "it's jittery" reaction rather than looking caught off guard by it.

## Part 6 — Close (slide 20)

One sentence each: replication succeeded (within-subject, LOSO, cross-cohort for CEN/DMN) → that
validated fingerprint now runs in a portable tool → the goal is taking DMN self-regulation
training outside the scanner, ultimately into everyday settings.

---

## If someone asks the harder questions

- **"Does the portable version work as well as the scanner-validated fingerprint?"** — Honest
  answer: no, not yet, and we've tested this directly. The portable 12-channel decoder, scored
  honestly on subjects it never trained on, shows CEN and DMN transferring weakly but reliably
  (r≈0.05–0.09, p<0.01) — real, but well below the research-cap numbers on these slides. We also
  tried pooling in more subjects, a wider montage, different regression methods, and per-subject
  calibration to close that gap; none of it worked yet. Don't overclaim the live demo's numbers
  as equivalent to slides 12–15 — they're a harder, more restricted test, and an open problem.
- **"What about PDA / the combined DMN-vs-CEN measure you feed back on?"** — In this classic
  research-cap method it replicates fine (see above). On the *portable* decoder specifically, we
  found PDA has no out-of-sample validity at all (near-zero, not significant) even though CEN and
  DMN individually still transfer — differencing the two cancels out the part that's
  cross-subject transferable. That's specific to the deployed tool, not this method broadly.
