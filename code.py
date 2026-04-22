import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
sns.set(style="whitegrid")
trades = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\data-science-projects\prime_ai\historical_data.csv')
sentiment = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\data-science-projects\prime_ai\fear_greed_index.csv')
print("trades shape:", trades.shape)
print("sentiment shape:", sentiment.shape)
print("\nmissing values in trades:")
print(trades.isnull().sum())
print("\nmissing values in sentiment:")
print(sentiment.isnull().sum())
print("\nduplicates in trades:", trades.duplicated().sum())
print("duplicates in sentiment:", sentiment.duplicated().sum())
trades['Date'] = pd.to_datetime(trades['Timestamp IST'], errors='coerce').dt.date
sentiment['Date'] = pd.to_datetime(sentiment['date'], errors='coerce').dt.date
sentiment_clean = sentiment[['Date', 'classification']].copy()
sentiment_clean.columns = ['Date', 'Classification']
df = pd.merge(trades, sentiment_clean, on='Date', how='inner')
print("\nafter merge:", df.shape)
df['win'] = df['Closed PnL'] > 0
df['leverage_group'] = pd.cut(df['Size USD'] / df['Size Tokens'].replace(0, np.nan),
bins=[0, 5, 20, np.inf], labels=['Low', 'Mid', 'High'])
trade_counts = df['Account'].value_counts()
frequent_traders = trade_counts[trade_counts > 50].index
df['trader_type'] = df['Account'].apply(lambda x: 'Frequent' if x in frequent_traders else 'Rare')
pnl_by_trader = df.groupby('Account')['Closed PnL'].sum()
winners = pnl_by_trader[pnl_by_trader > 0].index
df['performance'] = df['Account'].apply(lambda x: 'Winner' if x in winners else 'Loser')
trader_winrate = df.groupby('Account')['win'].mean().reset_index()
trader_winrate.columns = ['Account', 'trader_win_rate']
df = df.merge(trader_winrate, on='Account', how='left')
daily_pnl = df.groupby(['Account', 'Date'])['Closed PnL'].sum().reset_index().sort_values('Date')
def max_drawdown(pnl_series):
    cumulative = pnl_series.cumsum()
    peak = cumulative.cummax()
    return (cumulative - peak).min()
drawdown_by_trader = daily_pnl.groupby('Account')['Closed PnL'].apply(max_drawdown).reset_index()
drawdown_by_trader.columns = ['Account', 'max_drawdown']
df_with_dd = df.merge(drawdown_by_trader, on='Account', how='left')
drawdown_sentiment = df_with_dd.groupby('Classification')['max_drawdown'].mean()
win_rate = df.groupby('Classification')['win'].mean()
avg_size = df.groupby('Classification')['Size USD'].mean()
trades_per_day = df.groupby(['Date', 'Classification']).size().reset_index(name='num_trades')
long_short = df.groupby(['Classification', 'Side']).size().unstack(fill_value=0)
print("\nside values found:", df['Side'].unique())
lev_analysis = df.groupby(['Classification', 'leverage_group'])['Closed PnL'].mean().unstack()
freq_analysis = df.groupby(['Classification', 'trader_type'])['Closed PnL'].mean().unstack()
perf_analysis = df.groupby(['Classification', 'performance'])['Closed PnL'].mean().unstack()
print("\navg pnl by sentiment:")
print(df.groupby('Classification')['Closed PnL'].mean())
print("\nwin rate by sentiment:")
print(win_rate)
print("\navg trade size (USD) by sentiment:")
print(avg_size)
print("\navg trades per day:")
print(trades_per_day.groupby('Classification')['num_trades'].mean())
print("\nlong short breakdown:")
print(long_short)
print("\navg drawdown by sentiment:")
print(drawdown_sentiment)
print("\nleverage group analysis:")
print(lev_analysis)
print("\ntrader frequency analysis:")
print(freq_analysis)
print("\nwinner loser analysis:")
print(perf_analysis)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Classification', y='Closed PnL', data=df)
plt.title("PnL vs Sentiment")
plt.tight_layout()
plt.savefig("pnl_vs_sentiment.png", dpi=150)
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(x='Classification', y='Size USD', data=df)
plt.title("Trade Size vs Sentiment")
plt.tight_layout()
plt.savefig("tradesize_vs_sentiment.png", dpi=150)
plt.show()
plt.figure(figsize=(6, 4))
sns.countplot(x='Classification', data=df)
plt.title("Trade Count vs Sentiment")
plt.tight_layout()
plt.savefig("tradecount_vs_sentiment.png", dpi=150)
plt.show()
plt.figure(figsize=(8, 5))
drawdown_sentiment.plot(kind='bar', color=['red', 'green'])
plt.title("Avg Drawdown vs Sentiment")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig("drawdown_vs_sentiment.png", dpi=150)
plt.show()
trader_profile = df.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    win_rate=('win', 'mean'),
    avg_size=('Size USD', 'mean'),
    trade_count=('Closed PnL', 'count')
).reset_index()
trader_profile = trader_profile.merge(drawdown_by_trader, on='Account', how='left')
cluster_features = trader_profile[['total_pnl', 'win_rate', 'avg_size', 'trade_count', 'max_drawdown']].fillna(0)
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
trader_profile['cluster'] = kmeans.fit_predict(cluster_scaled)
cluster_summary = trader_profile.groupby('cluster')[['total_pnl', 'win_rate', 'avg_size', 'trade_count', 'max_drawdown']].mean()
print("\ncluster summary:")
print(cluster_summary)
cluster_labels = {}
for c in cluster_summary.index:
    row = cluster_summary.loc[c]
    if row['win_rate'] == cluster_summary['win_rate'].max():
        cluster_labels[c] = 'Consistent Winners'
    elif row['trade_count'] == cluster_summary['trade_count'].max():
        cluster_labels[c] = 'High Frequency Traders'
    else:
        cluster_labels[c] = 'Passive / Low Activity'

trader_profile['archetype'] = trader_profile['cluster'].map(cluster_labels)
print("\narchetype distribution:")
print(trader_profile['archetype'].value_counts())
plt.figure(figsize=(8, 5))
sns.scatterplot(data=trader_profile, x='avg_size', y='win_rate', hue='archetype', palette='Set2', s=80)
plt.title("Trader Clusters - Trade Size vs Win Rate")
plt.tight_layout()
plt.savefig("trader_clusters.png", dpi=150)
plt.show()
df = df.merge(trader_profile[['Account', 'archetype']], on='Account', how='left')
archetype_sentiment = df.groupby(['Classification', 'archetype'])['Closed PnL'].mean().unstack()
print("\narchetype pnl by sentiment:")
print(archetype_sentiment)
plt.figure(figsize=(9, 5))
archetype_sentiment.plot(kind='bar')
plt.title("Archetype Avg PnL - Fear vs Greed")
plt.ylabel("Avg PnL")
plt.tight_layout()
plt.savefig("archetype_pnl_sentiment.png", dpi=150)
plt.show()
fear_pnl = df[df['Classification'] == 'Fear']['Closed PnL'].mean()
greed_pnl = df[df['Classification'] == 'Greed']['Closed PnL'].mean()
fear_wr = df[df['Classification'] == 'Fear']['win'].mean()
greed_wr = df[df['Classification'] == 'Greed']['win'].mean()
fear_trades = trades_per_day[trades_per_day['Classification'] == 'Fear']['num_trades'].mean()
greed_trades = trades_per_day[trades_per_day['Classification'] == 'Greed']['num_trades'].mean()
print("\ninsight 1 - pnl and win rate:")
print("fear avg pnl:", round(fear_pnl, 4), "| win rate:", round(fear_wr, 4))
print("greed avg pnl:", round(greed_pnl, 4), "| win rate:", round(greed_wr, 4))
print("\ninsight 2 - trade frequency:")
print("fear trades/day:", round(fear_trades, 1))
print("greed trades/day:", round(greed_trades, 1))
print("\ninsight 3 - archetype pnl under different sentiments:")
print(archetype_sentiment)
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['Classification'])

features = df[['Size USD', 'Fee', 'sentiment_encoded', 'trader_win_rate']].fillna(0)
target = (df['Closed PnL'] > 0).astype(int)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
print("\nmodel accuracy:", model.score(x_test, y_test))
print(classification_report(y_test, model.predict(x_test)))
feat_imp = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
print("\nfeature importance:")
print(feat_imp)
plt.figure(figsize=(7, 4))
feat_imp.plot(kind='bar')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("\nstrategy 1:")
if fear_pnl < greed_pnl:
    print("Fear days show lower avg PnL — reduce position sizes during fear sentiment")
else:
    print("Fear days show higher avg PnL — contrarian trading may work during fear")
print("\nstrategy 2:")
if fear_trades < greed_trades:
    print("Traders trade less on fear days — frequent traders should stay active during fear as competition is lower")
else:
    print("Traders overtrade on fear days — high frequency traders should reduce trade count during fear")
daily_pnl.to_csv("daily_pnl.csv", index=False)
print("\ndone")
