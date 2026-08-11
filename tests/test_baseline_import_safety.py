"""Importing the baseline arm must do NOTHING.

Until this commit, ``aicds.models.baseline_sent2vec`` ran its entire 10-fold
pipeline at import time: no ``run_analysis``, no ``__main__`` guard.  Importing
it read the dataset, loaded the 21 GB BioSentVec binary, created two output
directory trees in the CURRENT WORKING DIRECTORY, and -- on any machine without
sent2vec, which is every Windows machine -- called ``sys.exit(1)`` from inside
an import.

Three consequences are worth pinning, because each one silently returns if the
restructuring is ever undone:

1. **No side effects at import.**  A module that mkdirs on import cannot be
   imported by a test, a parser, or a ``--help`` handler.  The directories are
   named by timestamp, so every accidental import also littered the cwd with a
   directory nobody would recognise later.
2. **A real entry point.**  ``run_analysis(encoder=None, config=LEGACY)``
   mirrors ``bert_models.run_analysis`` minus ``model_id``.  The ``encoder``
   seam is what lets a stub replace the 21 GB model; the ``config`` default is
   what keeps the legacy numbers reachable without argument.
3. **The config is a PARAMETER, not an import-time environment read.**  The
   module used to resolve ``from_env()`` at import, so ``$AICDS_PIPELINE`` was
   the only hook that arrived early enough -- and a bogus value raised from
   inside the import statement.  Resolution now belongs to the caller
   (``scripts/run_baseline.py``), so importing the module is inert whatever the
   environment says.

These tests import the module for real.  That is the point: on this repo's
Windows checkout the import would previously have exited the interpreter.
"""

import importlib
import inspect
import os
import sys


MODULE = "aicds.models.baseline_sent2vec"


def _fresh_import():
    """Import the module with an EMPTY module cache for it.

    Without the pop these tests would be vacuous: another test module (or an
    earlier test here) leaves the import cached, ``import_module`` returns the
    cached object without executing anything, and "the import created no
    directories" becomes trivially true.  The side effects under test happen
    exactly once per interpreter unless the cache is cleared.
    """
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def test_import_creates_no_directories(tmp_path, monkeypatch):
    """Import from an empty cwd and assert the cwd is still empty.

    The old module built ``Prediction Output_<timestamp>/`` and ``Prediction
    Symptom Details_<timestamp>/`` off ``os.getcwd()`` during import.  Checking
    for *any* new entry rather than those two names keeps the test honest if the
    output naming changes (it is scheduled to, onto the underscore spelling).
    """
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))

    _fresh_import()

    after = set(os.listdir(tmp_path))
    assert after == before, (
        "importing %s created %s in the cwd -- the pipeline must not run at "
        "import time" % (MODULE, sorted(after - before))
    )


def test_run_analysis_signature():
    """``run_analysis(encoder=None, config=LEGACY)`` -- both defaults matter.

    ``encoder=None`` keeps the untouched real path the default, and
    ``config=LEGACY`` is the invariant ``aicds.config`` exists to protect: if
    the default ever stops being LEGACY, the golden's byte-exact reference stops
    describing what a no-argument run does.
    """
    from aicds.config import LEGACY

    module = _fresh_import()

    assert callable(module.run_analysis)

    sig = inspect.signature(module.run_analysis)
    assert list(sig.parameters) == ["encoder", "config"]
    assert sig.parameters["encoder"].default is None
    assert sig.parameters["config"].default is LEGACY


def test_import_does_not_read_the_pipeline_env_var(tmp_path, monkeypatch):
    """A bogus $AICDS_PIPELINE must not make the import raise.

    ``from_env`` raises ``ValueError`` on an unknown name -- correctly, since a
    typo'd pipeline should never silently fall back to legacy.  While that call
    sat at module scope, the exception came out of the *import*, which meant a
    typo in one environment variable broke every consumer of the module,
    including ones that only wanted to read a function signature.  The check is
    on the *raise*, not on the value, because the module now keeps no config
    attribute at all.
    """
    monkeypatch.setenv("AICDS_PIPELINE", "definitely-not-a-pipeline")
    monkeypatch.chdir(tmp_path)

    module = _fresh_import()  # must not raise

    assert callable(module.run_analysis)
    # And the env var is still nobody's business at module scope.
    assert not hasattr(module, "PIPELINE_CONFIG")


def test_module_exposes_a_main_guard():
    """``python -m aicds.models.baseline_sent2vec`` still runs the pipeline.

    Read from source rather than executed, for the obvious reason.  Without the
    guard the module is importable but no longer runnable, which would trade one
    defect for another.
    """
    module = _fresh_import()
    source = inspect.getsource(module)

    assert 'if __name__ == "__main__":' in source
    guard = source.split('if __name__ == "__main__":')[1]
    assert "run_analysis(" in guard
    # And it must pass the ENVIRONMENT's config, not accept run_analysis'
    # LEGACY default.  This entry point is the one with no ``--pipeline`` flag
    # to fall back on, so a bare ``run_analysis()`` here silently drops
    # AICDS_PIPELINE -- which the import-time version of this module honoured.
    assert "from_env()" in guard
