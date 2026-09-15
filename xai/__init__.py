"""Anatomically anchored attribution analysis for the Retinexis ISBI study.

See README.md for the experiment plan. Run order:
    xai.check_frame -> xai.run_attribution -> xai.analyze -> xai.run_ablation -> xai.figures
"""
__all__ = ["frames", "modelio", "folddata", "attribution", "anatomy", "metrics"]
