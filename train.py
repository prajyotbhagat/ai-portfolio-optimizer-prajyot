import os
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import pickle
import json
from tqdm import tqdm
import urllib.request

def get_major_index_tickers():
    """Fetches tickers from major global indices to create a large, high-quality universe."""
    print("Fetching constituent lists for US, European, and Asian indices...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

    def fetch_table(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            html_content = response.read()
        return pd.read_html(html_content)
    
    sp500_tickers = fetch_table('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    nasdaq100_tickers = fetch_table('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()

    nifty50_tickers = []
    try:
        nifty50_table = fetch_table('https://en.wikipedia.org/wiki/NIFTY_50')
        nifty50_tickers = (nifty50_table[1]['Symbol'] + '.NS').tolist()
        print(f"Successfully fetched {len(nifty50_tickers)} Nifty 50 tickers.")
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 50 tickers. Error: {e}")

    ftse100_tickers = []
    try:
        ftse100_table = fetch_table('https://en.wikipedia.org/wiki/FTSE_100_Index')
        ftse100_tickers = (ftse100_table[3]['Ticker'] + '.L').tolist()
        print(f"Successfully fetched {len(ftse100_tickers)} FTSE 100 tickers.")
    except Exception as e:
        print(f"Warning: Could not fetch FTSE 100 tickers. Error: {e}")

    core_etfs = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "GLD", "DBC", "INDA"]
    all_tickers = core_etfs + sp500_tickers + nasdaq100_tickers + nifty50_tickers + ftse100_tickers
    unique_tickers = sorted(list(set(all_tickers)))
    print(f"Created a global universe of {len(unique_tickers)} unique assets.")
    return unique_tickers

TICKERS = get_major_index_tickers()
START = "2014-01-01"
PRED_HORIZON = 21
MODEL_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "device": "gpu", "random_state": 42, "verbose": -1}

def download_prices(tickers, start):
    df = yf.download(tickers, start=start, progress=True, threads=True)["Close"]
    df = df.dropna(axis=1, thresh=int(len(df) * 0.7)).fillna(method="ffill").dropna()
    return df

def get_and_save_fundamental_data(tickers):
    fund_data = {}
    for ticker in tqdm(tickers, desc="Fetching Fundamentals"):
        try:
            info = yf.Ticker(ticker).info
            fund_data[ticker] = {'longName': info.get('longName', ticker), 'sector': info.get('sector', 'N/A'), 'currency': info.get('currency', 'USD'), 'marketCap': info.get('marketCap'), 'trailingPE': info.get('trailingPE'), 'returnOnEquity': info.get('returnOnEquity')}
        except: continue
    # Save directly to the project folder
    with open("fundamental_data.json", "w") as f:
        json.dump(fund_data, f)

def make_features_with_macro(prices, vix_series, fund_df):
    ret = prices.pct_change()
    feat_frames = []
    for w in [21, 63, 126, 252]:
        mom = prices.pct_change(w).shift(1); vol = ret.rolling(w).std().shift(1)
        mom.columns = pd.MultiIndex.from_product([mom.columns, [f"mom_{w}"]]); vol.columns = pd.MultiIndex.from_product([vol.columns, [f"vol_{w}"]])
        feat_frames.extend([mom, vol])
    
    fund_features = fund_df[['trailingPE', 'returnOnEquity', 'marketCap']].reindex(prices.columns)
    for col in fund_features.columns:
        ff = pd.DataFrame(np.tile(fund_features[col].values, (len(prices), 1)), index=prices.index, columns=prices.columns)
        ff.columns = pd.MultiIndex.from_product([ff.columns, [col]]); feat_frames.append(ff)

    df_macro = pd.DataFrame(index=prices.index)
    df_macro["vix_level"] = vix_series.reindex(prices.index).fillna(method="ffill")
    df_macro["vix_mom_21"] = df_macro["vix_level"].pct_change(21).shift(1)
    for col in df_macro.columns:
        mat = pd.DataFrame(np.tile(df_macro[col].values.reshape(-1,1), (1, prices.shape[1])), index=df_macro.index, columns=prices.columns)
        mat.columns = pd.MultiIndex.from_product([mat.columns, [col]]); feat_frames.append(mat)
        
    return pd.concat(feat_frames, axis=1).sort_index(axis=1)

def stack_features(feat_df):
    col_names = list(feat_df.columns.levels[1])
    rows = []
    for date in tqdm(feat_df.index, desc="Stacking Features"):
        day = feat_df.loc[date]; tickers = sorted({c[0] for c in day.index})
        for t in tickers:
            try: rows.append((date, t, *day[t].values))
            except: rows.append((date, t, *([np.nan]*len(col_names))))
    return pd.DataFrame(rows, columns=["date","ticker"] + col_names).set_index(["date","ticker"])

def run_training_and_save():
    print("Starting training process...")
    prices_all = download_prices(TICKERS + ["^VIX"], START)
    vix = prices_all["^VIX"]; prices = prices_all.drop(columns=["^VIX"], errors="ignore")
    get_and_save_fundamental_data(prices.columns.tolist())
    # Load directly from the project folder
    with open("fundamental_data.json", "r") as f:
        fund_df = pd.DataFrame.from_dict(json.load(f), orient='index')

    feat = make_features_with_macro(prices, vix, fund_df)
    target = prices.pct_change(PRED_HORIZON).shift(-PRED_HORIZON)
    common_idx = feat.index.intersection(target.index)
    feat_sub, target_sub = feat.loc[common_idx], target.loc[common_idx]
    
    X = stack_features(feat_sub)
    y = target_sub.stack().rename("fwd_ret"); y.index.names = ['date', 'ticker']
    data_full = X.join(y, how="inner").dropna()
    feature_cols = [c for c in data_full.columns if c != "fwd_ret"]

    print(f"Training final model on {len(data_full)} data points...")
    final_model = lgb.LGBMRegressor(**MODEL_PARAMS).fit(data_full[feature_cols], data_full["fwd_ret"])
    
    print("Saving final model and artifacts...")
    # --- MODIFIED LINE ---
    # Save model in a version-agnostic text format instead of pickle
    final_model.save_model("latest_model.txt")
    
    with open("feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print("Training complete!")

if __name__ == "__main__":
    run_training_and_save()
