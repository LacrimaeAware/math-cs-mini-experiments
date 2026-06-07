# Writing conventions

These apply to all writing in this repository: README, docs, notes, code comments,
and commit messages.

## Tone

- Plain and factual, like a technical report. State what a script does and what it
  computes. Do not sell or impress.
- No promotional or emotional words.
- No first person (I, we, our) and no second person addressing the reader (you,
  your). Use impersonal constructions and name the method or script.
- Do not use "honest" or "honestly" as a qualifier. Use "limitations" or "caveats".
- Do not use em dashes or en dashes. Use periods, commas, parentheses, or colons.
  Use hyphens for ranges (1-4) and for joined names (Hardy-Littlewood).
- Do not reference private or parallel context an outside reader cannot follow (the
  git-ignored `private/` folder, other repositories, prior chats).
- Documents are records, not a conversation.

## Math notation

- Plain ASCII where practical: `p_n#`, `P/A^2`, `q^2 - p^2`, `sum 1/log N`,
  `phi(N)`, `C +/- P^x`, `6k +/- 1`.
- The notes under `number_theory/notes/` use LaTeX (`$...$`), which renders on
  GitHub.

## Structure

- The README is the front page: a factual description, a table of scripts and what
  each computes, a layout tree, and how to run.
- Notes are split into short topic files with an index, not one large file.

## Commits

- State what changed and why, plainly. Do not add a Co-Authored-By trailer.
