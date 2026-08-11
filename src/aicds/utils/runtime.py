"""Environment setup and duration formatting shared by both arms.

Both of these were defined three times over -- once in ``models/bert_models.py``,
once in ``models/baseline_sent2vec.py``, and once in the orphaned
``evaluation/bert_eval.py``. The two live copies of ``format_time`` were
byte-identical; the two live copies of ``ensure_nltk_data`` differed only in the
glyphs they print, and that difference was a latent Windows crash (see below).

That third set of copies is gone: ``evaluation/bert_eval.py`` was deleted
outright rather than salvaged (owner decision 2026-08-10; see
``docs/plans/TODO.txt``, "DECISIONS 2026-08-10" item 2). Be precise about what
that decision turned on, because the file did hold two things worth naming: a
raw ``AutoModel`` + manual mean-pooling path distinct from sentence-transformers
pooling, and the repo's only GPU-aware line. What never existed in any branch is
the *destination* -- no ``encoders/`` dir or ``hf_automodel*`` file has ever been
added (``git log --all --diff-filter=A`` is empty for those paths), so the
salvage was never performed and there was nothing to rewire toward. The owner's
other two reasons are why it was not worth building one now: the GPU-aware line
is worthless here, since encoding is 0.17-0.45% of wall-clock and a GPU buys the
encoder arms nothing (``docs/findings/08-runtime-and-cost.md``), and the
mean-pooling path can be rebuilt via the P38 encoder registry if it is ever
wanted. The file also had zero callers.

Neither function can move a number. ``ensure_nltk_data`` only writes to the
console, and ``format_time`` feeds the wall-clock trailer that the golden
comparator truncates before diffing -- which also means the golden does NOT
cover this module. Changes here have to be reasoned about directly.
"""

import nltk

# stopwords feeds tokenization; punkt_tab is the modern tokenizer table and punkt
# the fallback for older NLTK versions. All three are required.
REQUIRED_NLTK_DATA = {
    "stopwords": "corpora/stopwords",
    "punkt_tab": "tokenizers/punkt_tab",
    "punkt": "tokenizers/punkt",
}


def ensure_nltk_data():
    """Download any missing NLTK corpora, so a first run needs no manual setup.

    Deliberately ASCII-only in its output. The former baseline copy printed
    "✓" and "✗", which raise UnicodeEncodeError on a cp1252 Windows
    console -- so on Windows the baseline crashed precisely when it had work to
    do (a corpus was missing) and stayed silent when it did not.
    """
    missing = []
    for name, path in REQUIRED_NLTK_DATA.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    if missing:
        print("[INFO] First-time setup: Downloading required NLTK data packages...")
        print(f"[INFO] Missing packages: {', '.join(missing)}")
        for name in missing:
            print(f"[INFO] Downloading '{name}'... ", end='', flush=True)
            try:
                nltk.download(name, quiet=True)
                print("OK")
            except Exception as e:
                print(f"ERROR (Error: {e})")
        print("[SUCCESS] NLTK data download complete!")
        print("")


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{seconds:.2f} seconds ({minutes:.2f} minutes)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{seconds:.2f} seconds ({hours:.2f} hours, {minutes:.2f} minutes)"
