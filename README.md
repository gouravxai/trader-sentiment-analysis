This project analyzes the correlation between market sentiment (Bitcoin Fear & Greed Index) and actual trader performance on the Hyperliquid platform. The goal was to determine if sentiment scores act as a leading indicator for profitability and trading behavior.

Core Logic
Data Fusion: Integrated daily Bitcoin Fear & Greed Index logs with raw trade-level data from Hyperliquid, aligned by precise timestamps.

Trader Segmentation: Used K-Means Clustering to categorize traders into three distinct groups based on frequency, win rate, and volume:

Group A (Consistent): Steady performers with controlled drawdowns.

Group B (High-Frequency): Aggressive scalpers with high sensitivity to market shifts.

Group C (Passive): Traders who only enter during extreme market conditions.

Predictive Modeling: Implemented a Random Forest classifier to predict trade outcomes by using a combination of sentiment levels, trader history, and volatility metrics as input features.

Key Findings
Performance Variance: Significant shifts in PnL and win rates were observed when the sentiment moved from "Fear" to "Greed."

Overtrading Patterns: Data showed a direct spike in trade frequency during "Extreme Fear" cycles, often resulting in higher net losses for high-frequency accounts.

Resilience Mapping: The "Consistent" cluster showed the lowest correlation between sentiment swings and performance drops, indicating higher emotional/systemic discipline.

Technical Stack
Language: Python

Analysis: Pandas, NumPy

Machine Learning: Scikit-Learn (K-Means, Random Forest)

Visualization: Matplotlib, Seaborn
