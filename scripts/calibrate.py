import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# CONFIGURATION — update these paths before running
# =============================================================================

# parent folder which contains all the folds and val_predictions.csv
# then script will recursively search inside for all the val_predictions.csv
RUN_DIR = Path("YOUR/RUN/DIR/HERE")

# Column names in val_predictions.csv
# Change these if your CSV uses different column headers
COL_LABELS = "y_true"   # the ground truth column (0 or 1)
COL_PROBS  = "y_prob"   # the predicted probability column (0.0 to 1.0)

# =============================================================================

# treats temperature as a learnable parameter
# allows the optimiser to be able to tune it
class TemperatureScaler(nn.Module):
    """A tiny 1-parameter model that divides logits by Temperature (T)."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


# the original logits is not saved, only the probablities are saved
# so we run the signmoid in reverse to get back original numbers
def get_logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid to recover raw logits from 0-1 probabilities."""
    probs = torch.clamp(probs, 1e-7, 1 - 1e-7) # avoid dividing by zero crash
    return torch.log(probs / (1.0 - probs))



def calibrate_fold(predictions_csv_path: Path) -> float:
    """Finds the optimal Temperature parameter for a single fold."""

    # load the validation predictions
    df = pd.read_csv(predictions_csv_path)

    # extract labels and probablities as pytorch tensors
    labels = torch.tensor(df[COL_LABELS].values, dtype=torch.float32)
    probs = torch.tensor(df[COL_PROBS].values, dtype=torch.float32)
    logits = get_logits_from_probs(probs)


    scaler = TemperatureScaler() # one-parameter "model" starting at T=1
    criterion = nn.BCEWithLogitsLoss() # loss function
    optimiser = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    # training loop
    def eval_loss():
        optimiser.zero_grad()

        scaled_logits = scaler(logits)

        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimiser.step(eval_loss)

    optimal_t = scaler.temperature.item()
    return optimal_t



def main():
    parser = argparse.ArgumentParser(description="Calibrate model temperature")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the base run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print(f"Starting Temperature Calibration for runs in: {run_dir}")


    prediction_files = list(run_dir.rglob("val_predictions.csv"))

    if not prediction_files:
        print("No val_predictions.csv files found. Did Phase 2 finish?")
        return

    # Dictionary to store the T values for each fold
    calibration_results = {}

    for csv_path in prediction_files:
        # need a comment here
        fold_name = csv_path.parent.parent.name

        print(f"  Calibrating {fold_name}...")
        optimal_t = calibrate_fold(csv_path)

        print(f"    Optimal Temperature (T): {optimal_t:.4f}")
        calibration_results[fold_name] = optimal_t
        
        calibrated_probs = torch.sigmoid(logits / optimal_t)
        # compute ECE on calibrated_probs vs labels

        # Save T locally in that fold's prediction folder
        t_path = csv_path.parent / "temperature.json"
        with open(t_path, "w") as f:
            json.dump({"temperature": optimal_t}, f, indent=4)

    summary_path = run_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(calibration_results, f, indent=4)

    print(f"\nCalibration complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
