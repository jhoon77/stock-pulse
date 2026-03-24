#!/usr/bin/env python3
"""
Stock Pulse - 간단한 주식 백테스트 도구
이동평균 크로스오버 & RSI 전략 백테스트
"""

import json
import urllib.request
import urllib.parse
import time
import argparse
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate


# ─── Mock 데이터 생성 (네트워크 불가 시) ──────────────────────────────────────

def generate_mock_data(symbol: str, days: int = 252) -> pd.DataFrame:
    """실제와 유사한 주가 데이터를 시뮬레이션합니다 (GBM 모델)."""
    np.random.seed(hash(symbol) % 2**31)
    base_prices = {
        "AAPL": 170, "MSFT": 380, "GOOGL": 140, "AMZN": 180,
        "TSLA": 240, "NVDA": 800, "META": 500, "NFLX": 600,
        "SPY": 450, "QQQ": 380,
    }
    start_price = base_prices.get(symbol, 100)

    # Geometric Brownian Motion
    mu = 0.0003       # 일일 기대수익률
    sigma = 0.018     # 일일 변동성
    returns = np.random.normal(mu, sigma, days)
    prices = start_price * np.cumprod(1 + returns)

    dates = pd.bdate_range(end=datetime.now(), periods=days)
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        "close": prices,
        "volume": np.random.randint(10_000_000, 100_000_000, days),
    })
    return df


# ─── Yahoo Finance 데이터 수집 ───────────────────────────────────────────────

def fetch_yahoo_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Yahoo Finance에서 주가 데이터를 가져옵니다."""
    period_map = {
        "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
    }
    days = period_map.get(period, 365)
    end = int(time.time())
    start = end - days * 86400

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={end}&interval=1d"
    )

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        # CORS 프록시 경유 재시도
        proxy_url = f"https://corsproxy.io/?{urllib.parse.quote(url, safe='')}"
        req2 = urllib.request.Request(proxy_url, headers=headers)
        with urllib.request.urlopen(req2, timeout=15) as resp:
            data = json.loads(resp.read().decode())

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s").normalize(),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    })
    df.dropna(subset=["close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── 기술적 지표 계산 ─────────────────────────────────────────────────────────

def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def calc_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = calc_ema(series, 12)
    ema26 = calc_ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    return macd_line, signal_line


# ─── 매매 전략 ────────────────────────────────────────────────────────────────

def strategy_sma_crossover(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """이동평균 크로스오버 전략: 단기 MA가 장기 MA를 상향 돌파하면 매수, 하향 돌파하면 매도"""
    df = df.copy()
    df["sma_short"] = calc_sma(df["close"], short)
    df["sma_long"] = calc_sma(df["close"], long)
    df["signal"] = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
    df.loc[df["sma_short"] <= df["sma_long"], "signal"] = -1
    df["position"] = df["signal"].diff()
    return df


def strategy_rsi(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """RSI 전략: RSI가 과매도 구간 진입 시 매수, 과매수 구간 진입 시 매도"""
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"], period)
    df["signal"] = 0
    df.loc[df["rsi"] < oversold, "signal"] = 1
    df.loc[df["rsi"] > overbought, "signal"] = -1
    df["position"] = df["signal"].diff()
    return df


def strategy_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD 전략: MACD가 시그널선을 상향 돌파하면 매수, 하향 돌파하면 매도"""
    df = df.copy()
    df["macd"], df["macd_signal"] = calc_macd(df["close"])
    df["signal"] = 0
    df.loc[df["macd"] > df["macd_signal"], "signal"] = 1
    df.loc[df["macd"] <= df["macd_signal"], "signal"] = -1
    df["position"] = df["signal"].diff()
    return df


STRATEGIES = {
    "sma": ("SMA 크로스오버 (20/50)", strategy_sma_crossover),
    "rsi": ("RSI (14, 30/70)", strategy_rsi),
    "macd": ("MACD (12/26/9)", strategy_macd),
}


# ─── 백테스트 엔진 ────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """백테스트를 실행하고 결과를 반환합니다."""
    capital = initial_capital
    shares = 0
    trades = []
    entry_price = 0.0

    for i, row in df.iterrows():
        if pd.isna(row.get("position")):
            continue

        # 매수 신호
        if row["position"] > 0 and shares == 0:
            shares = int(capital // row["close"])
            if shares > 0:
                entry_price = row["close"]
                cost = shares * entry_price
                capital -= cost
                trades.append({
                    "type": "BUY",
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "price": round(entry_price, 2),
                    "shares": shares,
                    "capital": round(capital, 2),
                })

        # 매도 신호
        elif row["position"] < 0 and shares > 0:
            sell_price = row["close"]
            revenue = shares * sell_price
            pnl = (sell_price - entry_price) * shares
            capital += revenue
            trades.append({
                "type": "SELL",
                "date": row["date"].strftime("%Y-%m-%d"),
                "price": round(sell_price, 2),
                "shares": shares,
                "pnl": round(pnl, 2),
                "capital": round(capital, 2),
            })
            shares = 0

    # 마지막에 보유 중이면 최종 가격으로 평가
    last_price = df["close"].iloc[-1]
    portfolio_value = capital + shares * last_price

    # 수익률 계산
    total_return = (portfolio_value - initial_capital) / initial_capital * 100
    buy_hold_return = (last_price - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    # 일별 수익률 (Buy & Hold 기준 벤치마크)
    daily_returns = df["close"].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # 최대 낙폭 (MDD)
    cummax = df["close"].cummax()
    drawdown = (df["close"] - cummax) / cummax
    mdd = drawdown.min() * 100

    winning_trades = [t for t in trades if t["type"] == "SELL" and t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t["type"] == "SELL" and t.get("pnl", 0) <= 0]
    total_sells = len(winning_trades) + len(losing_trades)

    return {
        "initial_capital": initial_capital,
        "final_value": round(portfolio_value, 2),
        "total_return": round(total_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "total_trades": len(trades),
        "winning": len(winning_trades),
        "losing": len(losing_trades),
        "win_rate": round(len(winning_trades) / total_sells * 100, 1) if total_sells > 0 else 0,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(mdd, 2),
        "still_holding": shares,
        "trades": trades,
    }


# ─── 출력 ─────────────────────────────────────────────────────────────────────

def print_results(symbol: str, strategy_name: str, period: str, result: dict):
    print("\n" + "=" * 64)
    print(f"  STOCK PULSE BACKTEST REPORT")
    print("=" * 64)
    print(f"  종목: {symbol}   |   전략: {strategy_name}   |   기간: {period}")
    print("-" * 64)

    summary = [
        ["초기 자본", f"${result['initial_capital']:,.2f}"],
        ["최종 자산", f"${result['final_value']:,.2f}"],
        ["전략 수익률", f"{result['total_return']:+.2f}%"],
        ["Buy & Hold 수익률", f"{result['buy_hold_return']:+.2f}%"],
        ["초과 수익률", f"{result['total_return'] - result['buy_hold_return']:+.2f}%"],
        ["", ""],
        ["총 거래 횟수", f"{result['total_trades']}"],
        ["승리 / 패배", f"{result['winning']} / {result['losing']}"],
        ["승률", f"{result['win_rate']}%"],
        ["", ""],
        ["샤프 비율 (B&H)", f"{result['sharpe_ratio']}"],
        ["최대 낙폭 (MDD)", f"{result['max_drawdown']}%"],
    ]

    if result["still_holding"] > 0:
        summary.append(["현재 보유", f"{result['still_holding']}주 (미매도)"])

    print(tabulate(summary, tablefmt="simple", colalign=("left", "right")))

    # 거래 내역
    if result["trades"]:
        print("\n" + "-" * 64)
        print("  거래 내역")
        print("-" * 64)
        trade_rows = []
        for t in result["trades"]:
            pnl_str = f"{t['pnl']:+.2f}" if "pnl" in t else "-"
            trade_rows.append([
                t["date"], t["type"], f"${t['price']:.2f}",
                t["shares"], pnl_str, f"${t['capital']:,.2f}"
            ])
        print(tabulate(
            trade_rows,
            headers=["날짜", "유형", "가격", "수량", "손익", "잔액"],
            tablefmt="simple",
            colalign=("left", "center", "right", "right", "right", "right"),
        ))

    print("\n" + "=" * 64)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stock Pulse 백테스트 도구")
    parser.add_argument("symbol", nargs="?", default="AAPL", help="종목 티커 (기본: AAPL)")
    parser.add_argument("-s", "--strategy", choices=STRATEGIES.keys(), default="sma",
                        help="전략 선택: sma, rsi, macd (기본: sma)")
    parser.add_argument("-p", "--period", default="1y",
                        choices=["3mo", "6mo", "1y", "2y", "5y"],
                        help="백테스트 기간 (기본: 1y)")
    parser.add_argument("-c", "--capital", type=float, default=10000,
                        help="초기 자본금 (기본: $10,000)")
    parser.add_argument("--all", action="store_true",
                        help="모든 전략으로 백테스트 실행")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print(f"\n  {symbol} 데이터를 가져오는 중...")

    period_days = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
    try:
        df = fetch_yahoo_data(symbol, args.period)
        data_source = "Yahoo Finance (LIVE)"
    except Exception:
        print("  Yahoo Finance 접속 불가 - 시뮬레이션 데이터 사용")
        df = generate_mock_data(symbol, period_days.get(args.period, 252))
        data_source = "시뮬레이션 (MOCK)"

    print(f"  {len(df)}일 데이터 로드 완료 ({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  데이터 소스: {data_source}")

    strategies_to_run = STRATEGIES.keys() if args.all else [args.strategy]

    for key in strategies_to_run:
        name, func = STRATEGIES[key]
        processed = func(df)
        result = run_backtest(processed, args.capital)
        print_results(symbol, name, args.period, result)

    if args.all:
        print("\n  TIP: 특정 전략만 실행하려면 -s 옵션 사용")
        print("  예: python3 backtest.py AAPL -s rsi -p 2y\n")


if __name__ == "__main__":
    main()
