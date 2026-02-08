"""
base — abstract base class and preprocessing contract for all classifiers.

Every model in this package inherits from BaseClassifier so the base class
enforces a consistent interface so that training scripts never need to
know which architecture they are working with. This is a new introduction
from last project as it annoyed me that we had two training scripts for trasnfer
learning and from-scratch, this will allow for unified training pipeline this time.

Key purposes:
    - Define the forward / backbone / head / get_features contract
    - Own the linear-probe lock that prevents model.train() from
      reverting the backbone to train mode (the most common silent
      failure in linear probing experiments)
    - Provide freeze / unfreeze helpers that operate on the explicit
      backbone and head properties rather than fragile name-matching
    - Carry a PreprocessingConfig so each model declares its own
      input size, normalisation, and crop strategy — preventing
      silent pipeline mismatches between models

PreprocessingConfig is a frozen dataclass. Training code reads it from
the model and builds transforms accordingly; it never hard-codes
normalisation values.
"""
