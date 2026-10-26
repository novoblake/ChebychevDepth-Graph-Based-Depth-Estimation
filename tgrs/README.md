# TGRS manuscript workspace

Drafting area for the journal paper built on ChebDepth (IEEE Transactions on Geoscience and Remote Sensing). Everything for the paper lives here so it can be worked on separately from the code in the repo root.

```
tgrs/
├── README.md              this file
├── .gitignore             LaTeX build artifacts
├── paper/
│   ├── main.tex           IEEEtran journal skeleton; \input's the sections below
│   ├── references.bib     seed references (verify before submission)
│   └── sections/
│       ├── abstract.tex
│       ├── introduction.tex
│       ├── related_work.tex
│       ├── method.tex       drafted from chebfeatdepth.py
│       ├── experiments.tex  placeholders only, no results yet
│       └── conclusion.tex
├── figures/               figure sources and exports
├── experiments/           configs, logs, result tables for the paper
└── notes/
    └── todo.md            open questions and task list
```

## Building

```bash
cd tgrs/paper
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs a TeX distribution with the `IEEEtran` class (TeX Live: `texlive-publishers`). Build output is gitignored.

## Conventions

- Unfinished text uses `\todo{...}` (renders red). Search for `\todo` to see what is left.
- No numbers go into the paper unless they come from a run recorded under `experiments/`.
- The method section mirrors `../chebfeatdepth.py`. If the code changes, update `sections/method.tex`.

## Status

Skeleton only. See `notes/todo.md` for the first open question, which is the remote-sensing framing.
