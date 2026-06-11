# 🧠 AI Sector-Based Portfolio Optimizer

### 📊 Intelligent AI-Powered Portfolio Construction using Machine Learning & Quantitative Finance

This project builds an **AI-driven portfolio optimizer** that predicts short-term stock returns and constructs **risk-efficient portfolios** based on user-selected sectors.  
It combines **fundamental**, **technical**, and **macroeconomic features** using a **LightGBM model**, and provides an interactive **Streamlit web app** for real-time decision support.

---

## 🚀 Features

- 🌍 **Global stock universe**: Automatically fetches tickers from S&P 500, NASDAQ-100, NIFTY 50, and FTSE 100.  
- 💹 **AI model (LightGBM)**: Predicts 21-day forward returns using multi-scale momentum, volatility, and valuation data.  
- 📈 **Optimization engine**: Constructs maximum Sharpe ratio portfolios using `scipy.optimize`.  
- 🧩 **Sector-based selection**: Users can choose preferred sectors to build customized portfolios.  
- 🧠 **Macro-aware**: Integrates VIX-based volatility indicators for market risk context.  
- 🖥️ **Streamlit dashboard**: User-friendly web interface with visual insights, tables, and pie charts.

---

## 🏗️ Project Architecture

```
├── train_model.py                # Fetch data, create features, and train the LightGBM model
├── app.py                        # Streamlit front-end for portfolio optimization
├── fundamental_data.json         # Cached fundamental data (market cap, ROE, etc.)
├── feature_cols.json             # Model feature metadata
├── latest_model.txt              # Trained LightGBM model (saved booster)
└── README.md                     # This file
```

---

## 🎯 Model Performance

The LightGBM regression model forecasts 21-day forward returns using technical indicators, fundamental data, and macroeconomic features. Evaluated on recent market data across key tech tickers and major indices, the model demonstrates the following performance:

- **RMSE:** 0.0853
- **MAE:** 0.0650
- **R-squared:** 0.2419

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-portfolio-optimizer-prajyot.git
cd ai-portfolio-optimizer-prajyot
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate     # Linux / Mac
.venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model
Run the training script to download market data, create features, and train the LightGBM model:
```bash
python train_model.py
```

This will generate:
- `latest_model.txt`
- `feature_cols.json`
- `fundamental_data.json`

### 5️⃣ Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 🧮 Methodology Overview

1. **Data Collection**  
   Uses `yfinance` to download 10+ years of global equity and VIX index data.

2. **Feature Engineering**  
   - Momentum (21, 63, 126, 252 days)  
   - Volatility (rolling standard deviation)  
   - Fundamentals (P/E, ROE, Market Cap)  
   - Macroeconomic features (VIX level and momentum)

3. **Model Training**  
   - Model: `LightGBMRegressor`  
   - Target: 21-day forward return  
   - Objective: Predict relative asset performance

4. **Portfolio Optimization**  
   - Uses expected returns (AI predictions) and covariance matrix  
   - Maximizes Sharpe ratio with full investment constraint

---

## 📊 Streamlit Dashboard Highlights

- **Sector selection sidebar**: Choose sectors (e.g., Technology, Healthcare)  
- **AI portfolio generation**: Compute optimal weights and risk-return metrics  
- **Visual outputs**:
  - Portfolio pie chart  
  - Fundamentals and weights table  
  - Risk category explanation  
  - Natural-language investment summary  

---

## 📈 Example Output Metrics

| Metric | Description | Example |
|--------|--------------|---------|
| 📈 Predicted Annual Return | Model-forecasted expected return | 13.5% |
| ⚠️ Annual Risk (Volatility) | Standard deviation of returns | 21.2% |
| ⚖️ Sharpe Ratio | Risk-adjusted performance | 0.74 |

---

## 🧠 Future Enhancements

- Integrate macro indicators like inflation, interest rates, and oil prices.  
- Add backtesting and live rebalancing modules.  
- Incorporate reinforcement learning for adaptive portfolio updates.  
- Expand to crypto and ETF universes.

---

## 🛠️ Tech Stack

| Category | Tools Used |
|-----------|-------------|
| Data | Yahoo Finance (`yfinance`), Wikipedia tables |
| ML Model | LightGBM (GPU acceleration) |
| Optimization | SciPy (SLSQP solver) |
| App Framework | Streamlit |
| Visualization | Plotly Express |
| Language | Python 3.10+ |

---

## 📁 Deliverables

- ✅ Trained model (`latest_model.txt`)  
- ✅ Fundamental dataset (`fundamental_data.json`)  
- ✅ Feature metadata (`feature_cols.json`)  
- ✅ Streamlit app for deployment  
- ✅ README & report  

---

## 👨‍💻 Author

**Prajyot Bhagat**  
AI Researcher & Machine Learning Engineer  
📧 Contact: [prajyotbhagat1989@.com]  
🌐 GitHub: [github.com/prajyotbhagat](https://github.com/prajyotbhagat)
