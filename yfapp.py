#---------------------Good Version - Use this

# --- Import Libraries ---
import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import time
from streamlit.components.v1 import html


if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
# --- Streamlit Setup ---
st.set_page_config(page_title="Warren's Fair Value Estimator", layout="wide")

def set_ticker_from_dropdown():
    st.session_state.selected_ticker = st.session_state.holding_select

# --- Sidebar ---
st.sidebar.title("📂 Famous Investors")

investor_holdings = {
    "Warren Buffett (Berkshire)": ["","AAPL", "KO", "AXP", "HPQ", "UNH"],
    "Ray Dalio (Bridgewater)": ["", "JNJ", "KO", "UNH", "PEP", "BABA", "CVS"],
    "Cathie Wood (ARK)": ["","TSLA", "ZM", "PATH", "NVDA"],
    "Michael Burry (Scion)": ["","GOOG", "META", "BABA", "CVS", "WMT", "T", "HCA"],
}

investor = st.sidebar.selectbox("Select Investor", list(investor_holdings.keys()), key="investor_select")
st.sidebar.selectbox(
    "Select Holding", 
    investor_holdings[investor], 
    key="holding_select", 
    on_change=set_ticker_from_dropdown
)

# --- Main App Title ---
st.title("📈 Warren's Fair Value Estimator")

# --- Input Section ---
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown("""<div style="padding-top:20px;"></div>""", unsafe_allow_html=True)

    # Add beautiful button-style link
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <a href="https://msm-dcf.streamlit.app/" target="_blank" style="
                text-decoration: none;
                background-color: #4CAF50;
                color: white;
                padding: 12px 22px;
                border-radius: 6px;
                display: inline-block;
                font-weight: 600;
                font-size: 15px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.15);
                transition: background-color 0.3s ease;
            " onmouseover="this.style.backgroundColor='#45a049'" onmouseout="this.style.backgroundColor='#4CAF50'">
                👉 Want a Precise Analysis for Indian Stocks?
            </a>
        </div>
    """, unsafe_allow_html=True)

    ticker = st.text_input("Enter Ticker : (E.g. AAPL, TSLA, AMZN, TRENT)", 
                       value=st.session_state.selected_ticker).strip().upper()
    st.session_state.selected_ticker = ticker  # Update session ticker if user types manually

    years = st.selectbox("Time Period:", options=[5, 10], index=0)
    discount = st.number_input("Discount Rate (%)", value=5.0, min_value=0.0, max_value=25.0) / 100
    terminal = st.number_input("Terminal Growth Rate (%)", value=2.0, min_value=0.0, max_value=10.0) / 100
    mos = st.number_input("Margin of Safety (%)", min_value=0, max_value=100, value=40) / 100

# --- YFinance Data Fetching ---
@st.cache_data(show_spinner=False)
def get_yf_data(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get total shares and market cap from info
        total_shares = info.get('sharesOutstanding')
        current_price = info.get('currentPrice')
        market_cap = total_shares * current_price if total_shares and current_price else None

        # !---------------------------------------------------------------Profit Data----------------------------------------------------------------------------------------!
        # Get profit (Net Income) for CAGR
        is_df = stock.income_stmt.T
        is_df.index = pd.to_datetime(is_df.index)

        # Filter out None and NaN values before calculating
        # This makes the code more robust against missing data points
        filtered_profit_list = [v for v in is_df['Net Income From Continuing Operation Net Minority Interest'].sort_index().tail(years).tolist() if v is not None and not np.isnan(v)]

        profit_cagr = 0
        if len(filtered_profit_list) >= 2 and all(v != 0 for v in filtered_profit_list):
            profit_cagr = ((filtered_profit_list[-1] / filtered_profit_list[0]) ** (1 / (len(filtered_profit_list) - 1))) - 1
        
        # !---------------------------------------------------------------Cash Flow Data----------------------------------------------------------------------------------------!
        # Get financial data for FCF calculation
        cf = stock.cashflow.T
        cf.index = pd.to_datetime(cf.index)

        # Handle potential KeyError for Capital Expenditures
        capex_col = ['Net PPE Purchase And Sale']
        ocf_col = ['Operating Cash Flow']

        # Calculate FCF
        fcf_df = pd.DataFrame()
        fcf_df['OperatingCashFlow'] = cf[ocf_col].sort_index().iloc[1:]
        fcf_df['Net PPE Purchase And Sale'] = cf[capex_col].sort_index().iloc[1:]
        # print(fcf_df['Net PPE Purchase And Sale'])
        print(fcf_df['OperatingCashFlow'])


        if 'Operating Cash Flow' not in cf.columns or capex_col is None:
            st.warning("⚠️ Required financial data (Operating Cash Flow or Capital Expenditures) not found.")
            print("Missing required columns in cash flow data.")
            return None, None, None, None, None, None, None


        # print(fcf_df['OperatingCashFlow'])
        # print(fcf_df['Net PPE Purchase And Sale'])
        # Check for NaNs or None in required columns
        if fcf_df['OperatingCashFlow'].isnull().any():
            if 'Free Cash Flow' in cf.columns:
                fcf_df['FreeCashFlow'] = cf['Free Cash Flow'].sort_index()
                print("Yes")
                # st.toast("Using Free Cash Flow", icon="🧮")

            else:
                st.warning("⚠️ Free Cash Flow data not available as fallback.")
                return None, None, None, None, None, None, None
        else:
            fcf_df['FreeCashFlow'] = fcf_df['OperatingCashFlow'] - fcf_df['Net PPE Purchase And Sale']
            print("No")
            # st.toast("Using CF and Net PPE", icon="🧮")

        fcf_list = fcf_df['FreeCashFlow'].tail(5).tolist()
        cfo_list = fcf_df['OperatingCashFlow'].tail(5).tolist()
        capex_list = fcf_df['Net PPE Purchase And Sale'].tail(5).tolist()


        return total_shares, current_price, market_cap, fcf_list, cfo_list, capex_list, profit_cagr
        
    except Exception as e:
        st.error(f"❌ Failed to retrieve data for {ticker}. Please check the ticker symbol. Error: {e}")
        return None, None, None, None, None, None, None

total_shares, current_price, market_cap, fcf_list, cfo_list, capex_list, profit_cagr = get_yf_data(ticker, years)

# --- CAGR Function ---
def adjusted_cagr(values):
    for i in range(len(values) - 1):
        if values[i] is not None and values[i] > 0:
            return ((values[-1] / values[i]) ** (1 / (len(values) - 1 - i))) - 1
    return 0

fcf_cagr = adjusted_cagr(fcf_list)
FCF = fcf_list[-1] if fcf_list else 0

# --- Max Growth Cap by Market Cap Category ---
def get_max_growth_cap(mcap):
    if mcap is None: return 0.0
    if mcap >= 1_000_000_000_000: # 1 Trillion USD
        return 0.15
    elif mcap >= 100_000_000_000: # 100 Billion USD
        return 0.25
    elif mcap >= 10_000_000_000: # 10 Billion USD
        return 0.40
    elif mcap >= 2_000_000_000: # 2 Billion USD
        return 0.60
    elif mcap >= 100_000_000: # 100 Million USD
        return 1.00
    return 0.0

# --- Multi-Stage DCF Model ---
def multi_stage_dcf(fcf, init_growth, terminal_growth, wacc, years, max_cap):
    init_growth = min(init_growth, max_cap)
    
    # Linear decay of growth rate
    growth_step = (init_growth - terminal_growth) / (years - 1) if years > 1 else 0
    growths = [init_growth - i * growth_step for i in range(years)]
    
    fcfs = []
    current_fcf = fcf
    for i in range(years):
        current_fcf = current_fcf * (1 + growths[i])
        fcfs.append(current_fcf)
        
    dcf = sum([fcfs[i] / ((1 + wacc) ** (i + 1)) for i in range(years)])
    
    if wacc > terminal_growth:
        terminal_value = (fcfs[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        dcf += terminal_value / ((1 + wacc) ** years)
    else:
        dcf = np.nan # Cannot calculate if WACC <= terminal growth
    
    return dcf, fcfs, init_growth

# --- DCF Calculation and Display ---
if all([total_shares, current_price, market_cap]) and fcf_list:
    max_cap = get_max_growth_cap(market_cap)

    dcf_profit, fcfs_profit, init_growth_profit = multi_stage_dcf(FCF, profit_cagr, terminal, discount, years, max_cap)
    dcf_fcf, fcfs_fcf, init_growth_fcf = multi_stage_dcf(FCF, fcf_cagr, terminal, discount, years, max_cap)
    
    price_profit = dcf_profit / total_shares if dcf_profit else 0
    price_fcf = dcf_fcf / total_shares if dcf_fcf else 0
    low_fair, high_fair = sorted([price_fcf, price_profit])

    with col1:
        col10, col11 = st.columns(2)
        with col10:
            st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🗠 Share Price</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>💰 Market Cap.</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🫙 Category</h4>", unsafe_allow_html=True)

        with col11:
            st.markdown(f"<p style='font-size:24px; text-align:center; margin:7px 0;'>${current_price:.2f}</p>", unsafe_allow_html=True)
            
            mcap_display = ""
            mcap_category = ""
            if market_cap >= 1_000_000_000_000:
                mcap_display = f"${market_cap/1e12:.1f}T"
                mcap_category = "Mega-Cap"
            elif market_cap >= 100_000_000_000:
                mcap_display = f"${market_cap/1e9:.1f}B"
                mcap_category = "Large-Cap"
            elif market_cap >= 10_000_000_000:
                mcap_display = f"${market_cap/1e9:.1f}B"
                mcap_category = "Mid-Cap"
            elif market_cap >= 2_000_000_000:
                 mcap_display = f"${market_cap/1e9:.1f}B"
                 mcap_category = "Small-Cap"
            else:
                mcap_display = f"${market_cap:.0f}M"
                mcap_category = "Micro-Cap"

            st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>{mcap_display}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>{mcap_category}</p>", unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Projected Free Cash Flows")
        fig, ax1 = plt.subplots()
        x = list(range(1, years + 1))

        # Filter out NaN/inf values for plotting
        fcf_vals = [v for v in fcfs_fcf if np.isfinite(v)]
        profit_vals = [v for v in fcfs_profit if np.isfinite(v)]

        # Check if lists have data before plotting
        line1, line2 = [], []
        if fcf_vals:
            line1 = ax1.plot(x[:len(fcf_vals)], fcf_vals, label="FCF CAGR", marker="o", color="blue")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Free Cash Flow", color="blue")
            ax1.tick_params(axis='y', labelcolor="blue")
            ax1.set_ylim([min(fcf_vals) * 0.9, max(fcf_vals) * 1.1])
        
        if profit_vals:
            ax2 = ax1.twinx()
            line2 = ax2.plot(x[:len(profit_vals)], profit_vals, label="Profit Growth Rate", marker="o", color="green")
            ax2.set_ylabel("Profit", color="green")
            ax2.tick_params(axis='y', labelcolor="green")
            ax2.set_ylim([min(profit_vals) * 0.9, max(profit_vals) * 1.1])
        
        # Combine legends
        lines_all = line1 + line2
        labels_all = [l.get_label() for l in lines_all]
        if lines_all:
            ax1.legend(lines_all, labels_all, loc='upper left')

        # Add text labels if data exists
        if fcf_vals and profit_vals:
            diff = abs(profit_vals[-1] - fcf_vals[-1])
            threshold = 0.05 * max(profit_vals[-1], fcf_vals[-1])
            profit_offset = 0.02 * profit_vals[-1] if diff < threshold else 0
            fcf_offset = -0.02 * fcf_vals[-1] if diff < threshold else 0

            ax2.text(years, profit_vals[-1] + profit_offset, f"{profit_cagr * 100:.1f}%", color="green", fontsize=8, fontweight="bold", ha="right", va="bottom")
            ax1.text(years, fcf_vals[-1] + fcf_offset, f"{fcf_cagr * 100:.1f}%", color="blue", fontsize=8, fontweight="bold", ha="right", va="top")

        fig.suptitle(f"Projected FCF Over Time - {ticker.upper()}")
        fig.tight_layout()
        st.pyplot(fig)

    safe_profit = price_profit * (1 - mos)
    safe_fcf = price_fcf * (1 - mos)
    low_safe, high_safe = sorted([safe_fcf, safe_profit])

    # --- Summary ---
    st.markdown("<hr style='border: 1px solid #666; margin-top:40px;'>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🎯 Estimated Fair Price Range for {ticker}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center; margin-top:5px;'>🛡️ Safe Price Range (with {int(mos*100)}% Margin of Safety)</h4>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<p style='font-size:24px; text-align:center; margin:5px 0;'>${low_fair:.2f} - ${high_fair:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:24px; text-align:center; margin:5px 0; margin-top:18px;'>${low_safe:.2f} - ${high_safe:.2f}</p>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #666; margin-top:40px;'>", unsafe_allow_html=True)
    
    # ------------------------------------------------------------- Price Chart ------------------------------------------------------------------
    st.subheader(f"📈 Interactive Price Chart for {ticker}")

    col201, col202 = st.columns([2, 2])
    with col201:
        period = st.radio("Select Chart Period:", options=["1 Year", "3 Years", "5 Years", "10 Years"], horizontal=True)
        selected_period = {"1 Year": "1y", "3 Years": "3y", "5 Years": "5y", "10 Years" : "10y"}[period]
    with col202:
        ema_selection = st.multiselect("📊 Add EMAs to Chart", options=["EMA50", "EMA200", "EMA300"], default=[])

    st.markdown("<hr style='border: 1.5px dashed rgba(102, 102, 102, 0.5); margin-top:10px;'>", unsafe_allow_html=True)

    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period=selected_period)

        if not hist.empty:
            if "EMA50" in ema_selection:
                hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            if "EMA200" in ema_selection:
                hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()
            if "EMA300" in ema_selection:
                hist["EMA300"] = hist["Close"].ewm(span=300, adjust=False).mean()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00BFFF')
            ))

            for ema in ema_selection:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist[ema],
                    mode='lines',
                    name=ema,
                    line=dict(width=1.5, dash='dot')
                ))
            
            # Highlight zones only if fair and safe prices are valid numbers
            if np.isfinite(low_fair) and np.isfinite(high_fair):
                fig.add_shape(type="rect", xref="paper", yref="y",
                                x0=0, x1=1, y0=low_fair, y1=high_fair,
                                fillcolor="rgba(0, 255, 0, 0.2)", line_width=0, layer="below")
            if np.isfinite(low_safe) and np.isfinite(high_safe):
                fig.add_shape(type="rect", xref="paper", yref="y",
                                x0=0, x1=1, y0=low_safe, y1=high_safe,
                                fillcolor="rgba(234, 239, 44, 0.3)", line_width=0, layer="below")
                            
            if low_safe > 0: # Avoid negative values for the undervaluation zone
                fig.add_shape(type="rect", xref="paper", yref="y",
                                x0=0, x1=1, y0=low_safe * 0.9, y1=low_safe,
                                fillcolor="rgba(165, 42, 42, 0.3)", line_width=0, layer="below")

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color="rgba(0, 255, 0, 0.8)", symbol="square"),
                name="Fair Value Zone"
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color="rgba(234, 239, 44, 0.7)", symbol="square"),
                name="Safe Value Zone"
            ))
            if low_safe > 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=15, color="rgba(165, 42, 42, 0.5)", symbol="square"),
                    name="Undervalued Zone"
                ))

            fig.update_layout(
                title=f"{ticker} Stock Price ({period})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified",
                dragmode="zoom",
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(255,255,255,0)',
                    borderwidth=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No historical price data found for this ticker.")
    except Exception as e:
        st.error(f"❌ Error loading interactive price chart: {e}")
        st.error(f"Reason: {e}")

    st.markdown("<hr style='border: 1px solid #666; margin-top:10px;'>", unsafe_allow_html=True)
    
    with st.expander("🔍 Show All Variables (Debug Info)"):
        st.write("**Fetched/Scraped Variables:**")
        st.write({
            "ticker": ticker,
            "years": years,
            "discount": discount,
            "terminal": terminal,
            "mos": mos,
            "total_shares": total_shares,
            "profit_cagr": profit_cagr,
            "current_price": current_price,
            "market_cap": market_cap,
            "fcf_list": fcf_list,
        })
        st.write("**Calculated Variables:**")
        st.write({
            "fcf_cagr": fcf_cagr,
            "FCF (latest)": FCF,
            "max_cap": max_cap,
            "dcf_profit": dcf_profit,
            "dcf_fcf": dcf_fcf,
            "price_profit": price_profit,
            "price_fcf": price_fcf,
            "low_fair": low_fair,
            "high_fair": high_fair,
            "safe_profit": safe_profit,
            "safe_fcf": safe_fcf,
            "low_safe": low_safe,
            "high_safe": high_safe,
        })
        st.write("**Projected FCFs:**")
        st.write({
            "fcfs_profit": fcfs_profit,
            "fcfs_fcf": fcfs_fcf,
        })

    st.subheader("📄 Cash Flow Summary Table")

    with st.expander("🔽 View Cash Flow Summary (Historical & Projected)", expanded=False):
        latest_year = pd.Timestamp.now().year
        hist_years = [str(latest_year - i - 1) for i in reversed(range(len(fcf_list)))]
        proj_years = [f"Year {i}" for i in range(1, years + 1)]

        hist_data = {
            "Year": hist_years,
            "Operating Cash Flow": cfo_list,
            "Capital Expenditures": capex_list,
            "Free Cash Flow": fcf_list
        }
        hist_df = pd.DataFrame(hist_data)
        
        proj_data = {
            "Year": proj_years,
            "Projected FCF": fcfs_fcf,
            "Projected Profit (Net Income)": fcfs_profit
        }
        proj_df = pd.DataFrame(proj_data)

        hist_df["Type"] = "Historical"
        proj_df["Type"] = "Projected"

        proj_df = proj_df.rename(columns={
            "Projected FCF": "Free Cash Flow",
            "Projected Profit (Net Income)": "Operating Cash Flow"
        })
        proj_df["Capital Expenditures"] = None

        combined_df = pd.concat([hist_df, proj_df], ignore_index=True)

        combined_df = combined_df[["Year", "Type", "Operating Cash Flow", "Capital Expenditures", "Free Cash Flow"]]

        def highlight_projected(row):
            if row["Type"] == "Projected":
                return ["background-color: rgba(225, 0, 0, 0.05)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            combined_df.style
                .apply(highlight_projected, axis=1)
                .format({
                    "Operating Cash Flow": "{:.2f}",
                    "Capital Expenditures": "{:.2f}",
                    "Free Cash Flow": "{:.2f}"
                }),
            use_container_width=True
        )

else:
    st.warning("⚠️ Could not fetch all required data (e.g., total shares, free cash flow). Please check the ticker symbol and try a different one.")