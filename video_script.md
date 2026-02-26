# Energy Consumption Forecasting: Video Demonstration Script
**Target Length:** 3–5 minutes
**Video Format:** Screen recording with voiceover
**Output Requirement:** Upload to YouTube as "Unlisted" and copy link.

---

### [0:00 - 0:30] Introduction
*(Visual: Show the GitHub repository README or the cover slide of the Day 7 Notebook)*

**Speaker:**
"Hi everyone, my name is Nandith Gireesh, and I'm excited to present my submission for the Claysys AI Hackathon 2026. My project tackles the 'Energy Consumption Forecasting' problem statement.

The goal of this project was to predict future household energy usage based on 4 years of historical minute-level data. To solve this, I designed a structured 7-day pipeline that evaluates Statistical Models, Classical Machine Learning, Deep Learning, and finally, Ensembling."

---

### [0:30 - 1:45] Discussing the Approach & Solution
*(Visual: Walk through the `Day2_Preprocessing.ipynb` notebook or show a slide with feature importance `day4_feature_importance_all.png`)*

**Speaker:**
"To approach this problem, I first analyzed the 2 million rows of raw power data. I quickly realized that predicting raw electricity usage requires strong domain features. On Day 2, I resampled the data to hourly intervals and engineered 45 custom features. The two most critical features I created were 'Apparent Power' and 'Power Factor', which mathematically represent the true strain on an electrical grid.

Over the next few days, I tested three distinct approaches:
1. **Statistical Baselines**: Like ARIMA and Holt-Winters.
2. **Classical Machine Learning**: Random Forest, XGBoost, and LightGBM.
3. **Deep Learning**: LSTM and GRU models built from scratch in PyTorch.

Interestingly, my findings showed that the Deep Learning models actually struggled to beat the Classical ML models. Because the data is tabular and heavily relies on the cyclic lag features I engineered, LightGBM emerged as the champion model by a wide margin."

---

### [1:45 - 3:00] Demonstration: Running the Project
*(Visual: Open a Terminal / VS Code. Run the final deployment script: `python run_day7.py`)*

**Speaker:**
"Let me demonstrate the project running. I built the entire system to be completely reproducible through daily pipeline scripts. If a judge wants to evaluate my final results, they simply run `python run_day7.py`.

*(Press Enter and let the script run on screen)*

As you can see, the script automatically validates the project structure, loads the metrics from all 12 trained models, and outputs the final leaderboard.

*(Point out the output on the terminal)*

My LightGBM model achieved a Root Mean Squared Error (RMSE) of 0.0077 kilowatts, which represents a massive 99.2% improvement over the Naive Seasonal baseline model. The script also automatically generates a comprehensive text summary and dashboard visualizations saved in the `reports` folder."

---

### [3:00 - 4:00] Code Walkthrough
*(Visual: Open `src/models/ml_models.py` or `src/models/lstm_model.py` and scroll through cleanly documented code)*

**Speaker:**
"Looking briefly at the architecture, the codebase is modular. In the `src/models` directory, I separated the implementation of statistical baselines, tree-based ML models, and PyTorch deep learning networks. 

For the Deep Learning attempt, I built a flexible PyTorch `RNNForecaster` class that handles multivariate sliding-window sequences. 
*(Show `lstm_model.py`)*
It uses a 24-hour look-back window and features early stopping and learning rate scheduling to prevent overfitting.

I also maintained a rigorous version control strategy, making logical git commits at the end of every single day to track the progression of the models."

---

### [4:00 - 4:30] Conclusion
*(Visual: Show the `day6_final_leaderboard.png` bar chart)*

**Speaker:**
"To conclude, this project successfully forecasted energy consumption with extreme accuracy by leveraging domain-knowledge feature engineering and gradient boosting. The complete source code, 7 detailed Jupyter notebooks exploring each day's work, and the final trained models are all available in my GitHub repository.

Thank you to Claysys for organizing this hackathon, and thank you for watching my demonstration."

---
### 📝 Checklist Before Recording:
- [ ] Have the terminal open and ready to run `python run_day7.py`.
- [ ] Have your code editor open to `src/models/lstm_model.py`.
- [ ] Open the images `reports/figures/day4_feature_importance_all.png` and `day6_final_leaderboard.png` in a viewer so you can easily tab to them.
- [ ] Ensure your screen recording software captures text clearly (1080p).
- [ ] Remember to upload as **Unlisted** on YouTube!
