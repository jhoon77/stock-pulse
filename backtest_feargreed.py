#!/usr/bin/env python3
"""
Stock Pulse - 공포탐욕지수 기반 QQQ 분할매매 백테스트

전략:
  - Extreme Fear (0~25) 진입 시 → 3일간 1/3씩 매수
  - Extreme Greed (75~100) 진입 시 → 3일간 1/3씩 매도
  - 초기자본: $100,000 / 기간: 2023~2025

데이터 소스:
  - F&G 2023: GitHub whit3rabbit/fear-greed-data (실제 CNN F&G 일별 데이터)
  - F&G 2024~2025: 주요 시장 이벤트 기반 복원 (2024-08 일본발 폭락, 2024-12 Fed,
    2025-03~04 관세 쇼크 F&G=3 등)
  - QQQ 가격: 실제 월말 종가 기준 일봉 보간
"""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


# ─── 실제 데이터 로드 ─────────────────────────────────────────────────────────

def load_fear_greed() -> pd.DataFrame:
    """CSV 파일에서 Fear & Greed Index 일별 데이터를 로드합니다."""
    fg_2023 = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_2023.csv"))
    fg_2024_2025 = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_2024_2025.csv"))
    fg = pd.concat([fg_2023, fg_2024_2025], ignore_index=True)
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg.sort_values("date").reset_index(drop=True)
    return fg


def load_qqq_prices() -> pd.DataFrame:
    """월말 종가 기준 QQQ 데이터를 로드하고 일봉으로 보간합니다."""
    monthly = pd.read_csv(os.path.join(DATA_DIR, "qqq_monthly.csv"))
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly = monthly.sort_values("date").reset_index(drop=True)

    # 영업일 기준 일별 데이터 생성
    all_dates = pd.bdate_range(
        start=monthly["date"].iloc[0],
        end=monthly["date"].iloc[-1],
    )

    daily = pd.DataFrame({"date": all_dates})
    daily = daily.merge(monthly, on="date", how="left")

    # 선형 보간
    daily["close"] = daily["close"].interpolate(method="linear")
    # 첫 행 NaN 처리
    daily["close"] = daily["close"].ffill().bfill()

    # 2023-01-03 이후만
    daily = daily[daily["date"] >= "2023-01-03"].reset_index(drop=True)

    return daily


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """QQQ 가격과 F&G 지수를 매칭하여 데이터셋을 구성합니다."""
    qqq = load_qqq_prices()
    fg_df = load_fear_greed()

    # 날짜 기준 병합
    merged = qqq.merge(fg_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    fg_series = merged["value"].astype(float)

    df = merged[["date", "close"]].copy()

    return df, fg_series


# ─── 백테스트 엔진 ────────────────────────────────────────────────────────────

def run_fear_greed_backtest(
    df: pd.DataFrame,
    fg: pd.Series,
    initial_capital: float = 100_000,
    extreme_fear: int = 25,
    extreme_greed: int = 75,
    split_days: int = 3,
):
    """공포탐욕지수 기반 분할매매 백테스트"""

    capital = initial_capital
    shares = 0
    trades = []

    # 분할매매 큐
    buy_queue = []   # 남은 매수 분할 금액
    sell_queue = []  # 남은 매도 분할 수량

    prev_fg_zone = "neutral"
    total_invested = 0.0

    daily_log = []

    for i in range(len(df)):
        date = df["date"].iloc[i]
        price = df["close"].iloc[i]
        fg_val = fg.iloc[i]

        # 공포탐욕 구간 판별
        if fg_val <= extreme_fear:
            fg_zone = "EXTREME FEAR"
        elif fg_val >= extreme_greed:
            fg_zone = "EXTREME GREED"
        else:
            fg_zone = "neutral"

        # ── Extreme Fear 진입 시: 3일 분할매수 예약 ──
        if fg_zone == "EXTREME FEAR" and prev_fg_zone != "EXTREME FEAR":
            if capital > 100 and not buy_queue:  # 이미 매수 진행 중이 아닐 때
                buy_amount_per_day = capital / split_days
                for d in range(split_days):
                    buy_queue.append(buy_amount_per_day)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "SIGNAL",
                    "action": f"극단적 공포 감지 (F&G={fg_val:.0f}) -> 3일 분할매수 시작",
                    "price": price,
                    "shares": 0,
                    "amount": 0,
                    "capital": capital,
                    "total_shares": shares,
                })

        # ── Extreme Greed 진입 시: 3일 분할매도 예약 ──
        if fg_zone == "EXTREME GREED" and prev_fg_zone != "EXTREME GREED":
            if shares > 0 and not sell_queue:  # 이미 매도 진행 중이 아닐 때
                sell_shares_per_day = shares // split_days
                remainder = shares - sell_shares_per_day * split_days
                for d in range(split_days):
                    s = sell_shares_per_day + (1 if d < remainder else 0)
                    sell_queue.append(s)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "SIGNAL",
                    "action": f"극단적 탐욕 감지 (F&G={fg_val:.0f}) -> 3일 분할매도 시작",
                    "price": price,
                    "shares": 0,
                    "amount": 0,
                    "capital": capital,
                    "total_shares": shares,
                })

        # ── 분할매수 실행 ──
        if buy_queue:
            amount = buy_queue.pop(0)
            amount = min(amount, capital)
            buy_shares = int(amount // price)
            if buy_shares > 0:
                cost = buy_shares * price
                capital -= cost
                shares += buy_shares
                total_invested += cost
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "BUY",
                    "action": f"분할매수 ({split_days - len(buy_queue)}/{split_days})",
                    "price": round(price, 2),
                    "shares": buy_shares,
                    "amount": round(cost, 2),
                    "capital": round(capital, 2),
                    "total_shares": shares,
                })

        # ── 분할매도 실행 ──
        elif sell_queue:
            sell_shares = sell_queue.pop(0)
            sell_shares = min(sell_shares, shares)
            if sell_shares > 0:
                revenue = sell_shares * price
                avg_cost = total_invested / shares if shares > 0 else 0
                pnl = (price - avg_cost) * sell_shares
                capital += revenue
                shares -= sell_shares
                if shares == 0:
                    total_invested = 0
                else:
                    total_invested -= avg_cost * sell_shares
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "SELL",
                    "action": f"분할매도 ({split_days - len(sell_queue)}/{split_days})",
                    "price": round(price, 2),
                    "shares": sell_shares,
                    "amount": round(revenue, 2),
                    "capital": round(capital, 2),
                    "total_shares": shares,
                    "pnl": round(pnl, 2),
                })

        prev_fg_zone = fg_zone

        # 일별 포트폴리오 가치
        daily_log.append({
            "date": date,
            "close": price,
            "fg": fg_val,
            "portfolio": capital + shares * price,
            "shares": shares,
        })

    daily_df = pd.DataFrame(daily_log)

    # 최종 결과
    final_value = capital + shares * df["close"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    buy_hold_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    # MDD
    portfolio_series = daily_df["portfolio"]
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax
    mdd = drawdown.min() * 100

    # 연환산 수익률 (CAGR)
    years = len(df) / 252
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 승률
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    winning = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losing = [t for t in sell_trades if t.get("pnl", 0) <= 0]

    # Extreme Fear/Greed 발생 횟수
    signals = [t for t in trades if t["type"] == "SIGNAL"]
    fear_signals = [s for s in signals if "공포" in s["action"]]
    greed_signals = [s for s in signals if "탐욕" in s["action"]]

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return": round(total_return, 2),
        "buy_hold_return": round(buy_hold_return, 2),
        "cagr": round(cagr, 2),
        "mdd": round(mdd, 2),
        "total_trades": len([t for t in trades if t["type"] in ("BUY", "SELL")]),
        "buy_count": len([t for t in trades if t["type"] == "BUY"]),
        "sell_count": len(sell_trades),
        "winning": len(winning),
        "losing": len(losing),
        "win_rate": round(len(winning) / len(sell_trades) * 100, 1) if sell_trades else 0,
        "fear_signals": len(fear_signals),
        "greed_signals": len(greed_signals),
        "still_holding": shares,
        "trades": trades,
        "daily_df": daily_df,
    }


# ─── 출력 ─────────────────────────────────────────────────────────────────────

def print_report(result: dict, data_source: str):
    print()
    print("=" * 72)
    print("  STOCK PULSE - 공포탐욕지수 분할매매 백테스트")
    print("=" * 72)
    print("  종목: QQQ (Invesco QQQ Trust)")
    print("  전략: Extreme Fear -> 3일 1/3 분할매수 | Extreme Greed -> 3일 1/3 분할매도")
    print(f"  기간: 2023-01-03 ~ 2025-12-31")
    print(f"  데이터: {data_source}")
    print("-" * 72)

    summary = [
        ["초기 자본", f"${result['initial_capital']:>14,.2f}"],
        ["최종 자산", f"${result['final_value']:>14,.2f}"],
        ["", ""],
        ["전략 수익률", f"{result['total_return']:>+13.2f}%"],
        ["Buy & Hold 수익률", f"{result['buy_hold_return']:>+13.2f}%"],
        ["초과 수익률 (Alpha)", f"{result['total_return'] - result['buy_hold_return']:>+13.2f}%"],
        ["연환산 수익률 (CAGR)", f"{result['cagr']:>+13.2f}%"],
        ["", ""],
        ["최대 낙폭 (MDD)", f"{result['mdd']:>13.2f}%"],
        ["", ""],
        ["극단적 공포 시그널", f"{result['fear_signals']:>13}회"],
        ["극단적 탐욕 시그널", f"{result['greed_signals']:>13}회"],
        ["총 매수 횟수", f"{result['buy_count']:>13}회"],
        ["총 매도 횟수", f"{result['sell_count']:>13}회"],
        ["매도 승률", f"{result['win_rate']:>12.1f}%"],
    ]

    if result["still_holding"] > 0:
        last_price = result["daily_df"]["close"].iloc[-1]
        holding_value = result["still_holding"] * last_price
        summary.append(["", ""])
        summary.append(["잔여 보유", f"{result['still_holding']:>11}주"])
        summary.append(["보유 평가액", f"${holding_value:>14,.2f}"])

    print(tabulate(summary, tablefmt="simple", colalign=("left", "right")))

    # 거래 내역
    print("\n" + "-" * 72)
    print("  거래 내역")
    print("-" * 72)

    trade_rows = []
    for t in result["trades"]:
        if t["type"] == "SIGNAL":
            trade_rows.append([
                t["date"], ">>", t["action"], "", "", "",
                f"{t['total_shares']}주"
            ])
        elif t["type"] == "BUY":
            trade_rows.append([
                t["date"], "BUY",
                t["action"],
                f"${t['price']:.2f}",
                f"{t['shares']}주",
                f"-${t['amount']:,.2f}",
                f"{t['total_shares']}주"
            ])
        elif t["type"] == "SELL":
            pnl = t.get("pnl", 0)
            pnl_marker = "+" if pnl >= 0 else ""
            trade_rows.append([
                t["date"], "SELL",
                t["action"],
                f"${t['price']:.2f}",
                f"{t['shares']}주",
                f"+${t['amount']:,.2f} (P&L {pnl_marker}${pnl:,.2f})",
                f"{t['total_shares']}주"
            ])

    print(tabulate(
        trade_rows,
        headers=["날짜", "유형", "상세", "가격", "수량", "금액", "보유"],
        tablefmt="simple",
        colalign=("left", "center", "left", "right", "right", "right", "right"),
    ))

    # 연도별 수익률
    print("\n" + "-" * 72)
    print("  연도별 성과")
    print("-" * 72)

    daily = result["daily_df"].copy()
    daily["year"] = daily["date"].dt.year
    yearly_rows = []
    for year in sorted(daily["year"].unique()):
        year_data = daily[daily["year"] == year]
        start_val = year_data["portfolio"].iloc[0]
        end_val = year_data["portfolio"].iloc[-1]
        yr_return = (end_val - start_val) / start_val * 100

        bh_start = year_data["close"].iloc[0]
        bh_end = year_data["close"].iloc[-1]
        bh_return = (bh_end - bh_start) / bh_start * 100

        yearly_rows.append([
            year,
            f"${start_val:>12,.2f}",
            f"${end_val:>12,.2f}",
            f"{yr_return:>+8.2f}%",
            f"{bh_return:>+8.2f}%",
            f"{yr_return - bh_return:>+8.2f}%",
        ])

    print(tabulate(
        yearly_rows,
        headers=["연도", "시작 자산", "종료 자산", "전략", "B&H", "Alpha"],
        tablefmt="simple",
        colalign=("center", "right", "right", "right", "right", "right"),
    ))

    # F&G 구간별 분포
    print("\n" + "-" * 72)
    print("  공포탐욕지수 분포")
    print("-" * 72)

    fg = daily["fg"]
    zones = [
        ("Extreme Fear (0~25)", (fg <= 25).sum()),
        ("Fear (25~45)", ((fg > 25) & (fg <= 45)).sum()),
        ("Neutral (45~55)", ((fg > 45) & (fg <= 55)).sum()),
        ("Greed (55~75)", ((fg > 55) & (fg <= 75)).sum()),
        ("Extreme Greed (75~100)", (fg > 75).sum()),
    ]
    total_days = len(fg)
    zone_rows = []
    for name, count in zones:
        pct = count / total_days * 100
        bar = "#" * int(pct / 2)
        zone_rows.append([name, f"{count}일", f"{pct:.1f}%", bar])

    print(tabulate(zone_rows, tablefmt="simple", colalign=("left", "right", "right", "left")))

    print("\n" + "=" * 72)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_source = (
        "F&G 2023: CNN 실제 데이터 (GitHub) | "
        "F&G 2024-25: 주요 이벤트 기반 복원 | "
        "QQQ: 실제 월말 종가 보간"
    )

    print("\n  실제 데이터 로드 중...")
    df, fg = build_dataset()

    print(f"  {len(df)}일 데이터 로드 완료", end="")
    print(f" ({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"  QQQ 시작가: ${df['close'].iloc[0]:.2f} -> 종가: ${df['close'].iloc[-1]:.2f}")

    fg_min_idx = fg.idxmin()
    fg_max_idx = fg.idxmax()
    print(f"  F&G 최저: {fg.min():.0f} ({df['date'].iloc[fg_min_idx].strftime('%Y-%m-%d')})")
    print(f"  F&G 최고: {fg.max():.0f} ({df['date'].iloc[fg_max_idx].strftime('%Y-%m-%d')})")

    result = run_fear_greed_backtest(df, fg, initial_capital=100_000)
    print_report(result, data_source)
