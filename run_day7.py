import sys, os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"

print("=" * 60)
print("  DAY 7 - FINAL PROJECT SUBMISSION")
print("  Claysys AI Hackathon 2026")
print("=" * 60)

final_df = pd.read_csv(REPORTS_DIR / "day6_final_results.csv", index_col=0)
print("  [OK] Final results loaded.\n")
print(final_df[["MAE", "RMSE", "MAPE", "R2"]].head(5).to_string())

champion = final_df.index[0]
improvement = (final_df.loc["Naive Seasonal", "RMSE"] - final_df.loc[champion, "RMSE"]) / final_df.loc["Naive Seasonal", "RMSE"] * 100

with open(REPORTS_DIR / "final_submission_summary.txt", "w", encoding="utf-8") as f:
    f.write("Energy Consumption Forecasting - Final Report\n")
    f.write("===========================================\n\n")
    f.write("Overall Leaderboard (Top 5):\n")
    f.write(final_df[["MAE", "RMSE", "MAPE", "R2"]].head(5).to_string())
    f.write(f"\n\nWinning Model: {champion}\n")
    f.write(f"Improvement   : {improvement:.2f}%\n")
    
print("\n" + "=" * 60)
print("  PROJECT IS READY FOR SUBMISSION!")
print("=" * 60)

