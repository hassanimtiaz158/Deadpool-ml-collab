## What changed and why

<!-- What does this PR do, and which issue/phase does it belong to? -->

## Metrics (before -> after)

<!-- Paste the dvc exp show row or metrics.json diff. Never claim an improvement
     without the numbers that prove it. -->

| Metric | Before | After |
| --- | --- | --- |
| test accuracy | | |
| test top-5 accuracy | | |
| test macro F1 | | |

- commit SHA trained: `<sha>`
- data `.dvc` hash: `<md5>`

## Review checklist

- [ ] No data leakage (no target or future information in features)
- [ ] Splits are fixed; preprocessing fit on training data only
- [ ] No hardcoded paths; runs on a teammate's machine
- [ ] Seeds set for shuffling, initialisation and sampling
- [ ] Metric computed the way the team reports it
- [ ] `dvc push` done before `git push` (if data or models changed)
- [ ] Notebook restarted and run top to bottom (if notebooks changed)
- [ ] Style and naming (linter passes)
- [ ] Tests added/updated for new logic in `src/`

## How you verified this

<!-- The exact commands you ran, and what you saw. Reviewers should be able to
     reproduce this without asking you a question. -->

```
git clone ...
dvc pull
dvc repro
```

## Notes / follow-ups

<!-- Anything deliberately left out, and the ticket for it. -->