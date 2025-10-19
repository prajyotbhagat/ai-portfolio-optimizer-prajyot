import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb  # <-- Import LightGBM
import json
import os
from scipy.optimize import minimize
import plotly.express as px

st.set_page_config(page_title="AI Sector-Based Optimizer", layout="wide")
st.title("Sectoral AI Portfolio Optimizer")
st.markdown("Select the company sectors you want to invest in, and the AI will build the optimal risk-return portfolio for you.")

@st.cache_resource
def load_artifacts():
    try:
        # Load the model using LightGBM's built-in, version-safe function
        model = lgb.Booster(model_file="latest_model.txt")
        
        with open("feature_cols.json", "r") as f:
            feature_cols = json.load(f)
        with open("fundamental_data.json", "r") as f:
            fund_data = json.load(f)
        return model, feature_cols, pd.DataFrame.from_dict(fund_data, orient='index')
    except (FileNotFoundError, lgb.basic.LightGBMError):
        # Updated error handling for both file not found and model loading issues
        st.error("Model files missing or corrupt. Please run `train.py` first.", icon="🚨")
        return None, None, None

def make_features_live(prices, vix_series, fund_df):
    ret = prices.pct_change()
    feat_frames = []
    for w in [21, 63, 126, 252]:
        mom = prices.pct_change(w)
        vol = ret.rolling(w).std()
        mom.columns = pd.MultiIndex.from_product([mom.columns, [f"mom_{w}"]])
        vol.columns = pd.MultiIndex.from_product([vol.columns, [f"vol_{w}"]])
        feat_frames.extend([mom, vol])
    
    fund_features = fund_df[['trailingPE', 'returnOnEquity', 'marketCap']].reindex(prices.columns)
    for col in fund_features.columns:
        ff = pd.DataFrame(np.tile(fund_features[col].values, (len(prices), 1)), index=prices.index, columns=prices.columns)
        ff.columns = pd.MultiIndex.from_product([ff.columns, [col]])
        feat_frames.append(ff)
        
    df_macro = pd.DataFrame(index=prices.index)
    df_macro["vix_level"] = vix_series.reindex(prices.index).fillna(method="ffill")
    df_macro["vix_mom_21"] = df_macro["vix_level"].pct_change(21)
    for col in df_macro.columns:
        mat = pd.DataFrame(np.tile(df_macro[col].values.reshape(-1, 1), (1, prices.shape[1])), index=df_macro.index, columns=prices.columns)
        mat.columns = pd.MultiIndex.from_product([mat.columns, [col]])
        feat_frames.append(mat)
        
    return pd.concat(feat_frames, axis=1).sort_index(axis=1).iloc[-1]

@st.cache_data(ttl=3600)
def find_ai_optimal_portfolio(_model, feature_cols, fund_df, tickers):
    if not tickers:
        return None, None, None, None

    all_tickers = tickers + ["^VIX"]
    prices = yf.download(all_tickers, period="5y", progress=False)['Close'].dropna(axis=1, thresh=500).fillna(method="ffill")
    if prices.empty or "^VIX" not in prices.columns:
        return None, None, None, None
    
    valid_tickers = [t for t in tickers if t in prices.columns]
    vix = prices["^VIX"]
    prices = prices[valid_tickers]

    latest_features = make_features_live(prices, vix, fund_df)
    pred_df = pd.DataFrame([latest_features.get(t) for t in prices.columns], index=prices.columns, columns=feature_cols).fillna(0.0)
    mu_annualized = pd.Series(_model.predict(pred_df), index=pred_df.index) * 12

    S = prices.pct_change().cov() * 252

    def neg_sharpe(weights):
        p_return = np.sum(mu_annualized * weights)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        return -p_return / p_volatility if p_volatility > 0 else 0

    num_assets = len(valid_tickers)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    res = minimize(neg_sharpe, np.array([1./num_assets]*num_assets), method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = pd.Series(res.x, index=valid_tickers, name="Weight").round(4)
    expected_return = np.sum(mu_annualized * optimal_weights)
    expected_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(S, optimal_weights)))
    sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0

    return optimal_weights[optimal_weights > 0.001], expected_return, expected_volatility, sharpe_ratio

# --- Streamlit UI ---
st.sidebar.header("Your Portfolio")
model, feature_cols, fund_df = load_artifacts()

if fund_df is not None:
    all_sectors = sorted(fund_df['sector'].dropna().unique())
    selected_sectors = st.sidebar.multiselect(
        "1. Select Sectors to Analyze",
        all_sectors,
        default=["Technology", "Healthcare", "Financial Services", "Consumer Cyclical"]
    )

    if st.sidebar.button("✅ Find Optimal Portfolio"):
        if not selected_sectors:
            st.error("Please select at least one sector.")
        else:
            filtered_df = fund_df[fund_df['sector'].isin(selected_sectors)]
            tickers_to_analyze = filtered_df.index.tolist()
            
            with st.spinner(f"Analyzing {len(tickers_to_analyze)} assets in the selected sectors..."):
                try:
                    weights, p_return, p_risk, p_sharpe = find_ai_optimal_portfolio(model, feature_cols, fund_df, tickers_to_analyze)
                    if weights is None or weights.empty:
                        st.error("Could not fetch valid data or build a portfolio. This may be a temporary network issue or no assets met the criteria.")
                    else:
                        st.session_state.weights = weights
                        st.session_state.p_return = p_return
                        st.session_state.p_risk = p_risk
                        st.session_state.p_sharpe = p_sharpe
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if 'weights' in st.session_state and st.session_state.weights is not None and not st.session_state.weights.empty:
    st.success("AI-Optimal portfolio generated successfully!")
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Predicted Annual Return", f"{st.session_state.p_return:.2%}")
    col2.metric("⚠️ Predicted Annual Risk (Volatility)", f"{st.session_state.p_risk:.2%}")
    col3.metric("⚖️ Sharpe Ratio", f"{st.session_state.p_sharpe:.2f}",
                help="Measures risk-adjusted return. A higher value is better. > 1 is good, > 2 is very good.")

    st.markdown("---")
    
    display_df = st.session_state.weights.to_frame()
    display_df = display_df.join(fund_df)

    st.subheader("💡 AI-Generated Recommendations")
    with st.container(border=True):
        if st.session_state.p_risk > 0.30:
            st.warning(f"**High Risk Portfolio:** This portfolio has a predicted annual volatility of **{st.session_state.p_risk:.1%}**. It has significant growth potential but is susceptible to large market swings.")
        elif st.session_state.p_risk > 0.20:
            st.info(f"**Moderate Risk Portfolio:** With a predicted volatility of **{st.session_state.p_risk:.1%}**, this portfolio offers a balance between growth and stability.")
        else:
            st.success(f"**Low Risk Portfolio:** This portfolio is focused on stability, with a predicted volatility of **{st.session_state.p_risk:.1%}**.")
        
        top_3_holdings = display_df.nlargest(min(3, len(display_df)), 'Weight')
        top_sectors = display_df.groupby('sector')['Weight'].sum().nlargest(min(2, display_df['sector'].nunique()))

        recommendations = []
        if len(top_sectors) >= 2:
            recommendations.append(f"- The AI has identified the **{top_sectors.index[0]}** and **{top_sectors.index[1]}** sectors as having the best risk-return trade-off based on current market data.")
        elif len(top_sectors) == 1:
            recommendations.append(f"- The AI has identified the **{top_sectors.index[0]}** sector as having the best risk-return trade-off based on current market data.")

        if len(top_3_holdings) >= 2:
            recommendations.append(f"- The portfolio's performance is primarily driven by its significant allocations to **{top_3_holdings.iloc[0]['longName']} ({top_3_holdings.index[0]})** and **{top_3_holdings.iloc[1]['longName']} ({top_3_holdings.index[1]})**.")
        elif len(top_3_holdings) == 1:
             recommendations.append(f"- The portfolio's performance is primarily driven by its allocation to **{top_3_holdings.iloc[0]['longName']} ({top_3_holdings.index[0]})**.")

        recommendations.append(f"- Given the Sharpe Ratio of **{st.session_state.p_sharpe:.2f}**, the model indicates this is a highly efficient portfolio for the selected sectors.")
        recommendations.append("- **Recommendation:** This portfolio is optimized based on the AI's forward-looking predictions. Consider re-running this analysis periodically to adapt to new market conditions.")
        
        st.markdown("\n".join(recommendations))

    st.subheader("💵 Recommended Allocation & Fundamentals")
    cols_to_display = ['longName', 'sector', 'currency', 'marketCap', 'trailingPE', 'returnOnEquity', 'Weight']
    cols_that_exist = [col for col in cols_to_display if col in display_df.columns]
    formatter = {"Weight": "{:.2%}", "marketCap": lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A", "trailingPE": "{:.2f}", "returnOnEquity": "{:.2%}"}
    st.dataframe(display_df[cols_that_exist].style.format(formatter), use_container_width=True)

    st.subheader("📊 Allocation Visualized")
    fig = px.pie(st.session_state.weights, values='Weight', names=st.session_state.weights.index, title='Optimal Portfolio Weights')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
