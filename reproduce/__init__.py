"""Reproduction of the paper's evaluation from the simulator in ``src/``.

The pipeline has three layers:

* :mod:`reproduce.scenarios` reads the systems under study from
  ``scenarios/``; :mod:`reproduce.protocol` holds the measurement protocol
  and the four policies, and composes the two into a runnable config.
* :mod:`reproduce.runner` executes those configs, caching by configuration
  text and spreading independent runs across processes.
* :mod:`reproduce.figures` turns the results into the paper's figures, with
  :mod:`reproduce.style` holding what they share.

Run it with ``python -m reproduce``.
"""
