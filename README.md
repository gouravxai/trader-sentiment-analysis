Data Science Intern Project @ Primetrade.ai
Created by: Gourav Sharma

 The Big Idea
Have you ever wondered if the "mood" of the market actually changes how people trade? When everyone is terrified (Fear), do traders play it safe? When everyone is dreaming of Lambos (Greed), do they get reckless?

I wanted to find out. This project looks at Hyperliquid (a crypto trading platform) to see if Bitcoin’s Fear & Greed Index actually predicts how traders perform and behave.

The goal: Figure out if market sentiment is just noise, or if it’s a secret signal for winning.

How I Built This
1. The Ingredients (The Data)
I used two main sources:

Daily Bitcoin Sentiment: A log of whether the market was feeling Fearful or Greedy.

Hyperliquid Trading History: The raw data of what traders actually did.

2. The Process
Clean & Connect: I synced up the timestamps so I could see exactly what was happening on the platform during specific market moods.

The Deep Dive: I looked for patterns. Did win rates drop when fear was high? Did people trade more often when they were greedy?

Grouping Traders: I used an AI model (K-Means) to sort traders into three distinct "personalities":

The Pros: Consistent winners who keep their cool.

The Hustlers: High-frequency traders who are always active.

The Lurkers: Passive traders who only jump in occasionally.

The Crystal Ball: I built a Random Forest model to see if we could actually predict if a trade would be profitable based on the market mood and the trader's history.

 What did I find? (The "Aha!" Moments)
Sentiment Matters: Traders don't just feel differently during Fear vs. Greed—they perform differently. PnL (profit) and win rates shift noticeably depending on the market's mood.

Fear makes us frantic: Trade frequency changes when the market gets moody. Some traders overtrade when they're scared, which usually leads to losing money.

Different strokes for different folks: "Consistent Winners" handle fear much better than "High-Frequency Traders."

 My Advice for Traders
Size Down in Fear: When the market index hits "Fear," it’s probably time to lower your position sizes. The risk-to-reward ratio gets much uglier during these times.

Cool Your Jets: If you’re a high-frequency trader, watch out for "overtrading" during extreme sentiment swings. It’s a fast track to a big drawdown.

 Want to see the data yourself?
Step 1: Get the code
Bash
git clone https://github.com/gouravxai/trader-sentiment-analysis
cd trader-sentiment-analysis
Step 2: Get the tools
You'll need Python and a few standard libraries:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn
Step 3: Run the magic
Make sure you have your CSV files (fear_greed_index.csv and historical_data.csv) in the folder, then run:

Bash
python analysis.py
📊 What’s in the results?
The script will spit out a bunch of charts (look for the .png files!) showing everything from how PnL shifts during Greed, to which trader archetypes are the most resilient.

The MVP file: feature_importance.png — this shows you what actually matters most when trying to predict a winning trade.
