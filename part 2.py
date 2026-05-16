"""
main.py — Full CNN Pipeline Runner
====================================
Runs all 5 tasks end-to-end:
    Task 1 → Problem Identification
    Task 2 → Dataset Exploration
    Task 3 → Image Preprocessing
    Task 4 → CNN Model Creation
    Task 5 → Model Training & Evaluation

SETUP:
    1. Install dependencies:  pip install -r requirements.txt
    2. Set DATA_DIR below to your dataset root folder
    3. Run:  python main.py
"""

import os
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR    = "data"    # ← UPDATE THIS to your dataset path
NUM_CLASSES = 6         # ← UPDATE THIS to your number of classes
EPOCHS      = 30
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from task1_problem_identification import PROBLEM_TYPE, JUSTIFICATION
from task2_dataset_exploration    import run_exploration
from task3_preprocessing          import run_preprocessing
from task4_model                  import run_model_creation
from task5_training_evaluation    import run_training_evaluation


def main():
    print("\n" + "█" * 60)
    print("  PART 2: CNN COMPUTER VISION — FULL PIPELINE")
    print("█" * 60)

    # ── TASK 1 ──────────────────────────────────────────────────────
    print(f"\n[1/5] Problem Type: {PROBLEM_TYPE}")
    print(JUSTIFICATION)

    # ── TASK 2 ──────────────────────────────────────────────────────
    run_exploration(data_dir=DATA_DIR)

    # ── TASK 3 ──────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_names = run_preprocessing(data_dir=DATA_DIR)

    if train_ds is None:
        print("\n  ⚠  Dataset not found — skipping Tasks 4 & 5.")
        print("     Update DATA_DIR in main.py and re-run.")
        return

    num_classes = len(class_names)
    print(f"\n  Detected {num_classes} classes: {class_names}")

    # ── TASK 4 ──────────────────────────────────────────────────────
    model = run_model_creation(num_classes=num_classes)

    # ── TASK 5 ──────────────────────────────────────────────────────
    run_training_evaluation(
        model, train_ds, val_ds, test_ds, class_names, epochs=EPOCHS
    )

    print("\n" + "█" * 60)
    print("  ALL TASKS COMPLETE  —  outputs saved to outputs/ folder")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
