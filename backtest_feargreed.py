#!/usr/bin/env python3
"""
Stock Pulse - 공포탐욕지수 기반 QQQ 분할매매 백테스트

전략:
  - Extreme Fear (0~25) 진입 시 → 3일간 1/3씩 매수
  - Extreme Greed (75~100) 진입 시 → 3일간 1/3씩 매도
  - 초기자본: $100,000 / 기간: 2023~2025
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime


# ─── 데이터 생성 (2023~2025 QQQ + Fear & Greed Index) ────────────────────────

def generate_qqq_data() -> pd.DataFrame:
    """2023~2025 QQQ 실제 흐름을 반영한 시뮬레이션 데이터 생성.

    주요 시장 이벤트 반영:
    - 2023 초: 약세장 회복기 (QQQ ~270)
    - 2023 중~말: AI 랠리 (QQQ ~390)
    - 2024 초: 조정 후 재상승
    - 2024 중: 사상 최고치 (QQQ ~500+)
    - 2024 8월: VIX 급등 조정 (일본 금리)
    - 2024 말~2025: 변동성 속 횡보/상승
    """
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-03", end="2025-12-31")

    # 구간별 추세 설정 (일일 수익률 평균, 변동성)
    segments = [
        ("2023-01-03", "2023-03-15", -0.0002, 0.015),   # 초반 약세
        ("2023-03-16", "2023-06-30",  0.0015, 0.012),   # 회복 랠리
        ("2023-07-01", "2023-10-31",  0.0003, 0.013),   # 횡보
        ("2023-11-01", "2023-12-31",  0.0018, 0.010),   # 연말 랠리
        ("2024-01-01", "2024-03-31",  0.0012, 0.011),   # AI 모멘텀
        ("2024-04-01", "2024-06-30",  0.0008, 0.014),   # 조정 후 상승
        ("2024-07-01", "2024-07-31",  0.0015, 0.010),   # 사상 최고
        ("2024-08-01", "2024-08-15", -0.0030, 0.025),   # 8월 폭락
        ("2024-08-16", "2024-09-30",  0.0015, 0.015),   # 반등
        ("2024-10-01", "2024-12-31",  0.0010, 0.013),   # 연말 상승
        ("2025-01-01", "2025-03-31",  0.0005, 0.016),   # 변동성
        ("2025-04-01", "2025-06-30", -0.0005, 0.018),   # 조정
        ("2025-07-01", "2025-09-30",  0.0010, 0.014),   # 회복
        ("2025-10-01", "2025-12-31",  0.0008, 0.012),   # 연말
    ]

    start_price = 270.0  # 2023년 초 QQQ 가격
    prices = []
    price = start_price

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        mu, sigma = 0.0003, 0.015  # default
        for seg_start, seg_end, seg_mu, seg_sigma in segments:
            if seg_start <= date_str <= seg_end:
                mu, sigma = seg_mu, seg_sigma
                break
        ret = np.random.normal(mu, sigma)
        price *= (1 + ret)
        prices.append(price)

    prices = np.array(prices)
    df = pd.DataFrame({
        "date": dates[:len(prices)],
        "close": prices,
        "open": prices * (1 + np.random.uniform(-0.003, 0.003, len(prices))),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.008, len(prices)))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.008, len(prices)))),
    })
    return df


def generate_fear_greed(df: pd.DataFrame) -> pd.Series:
    """QQQ 가격 데이터 기반 시뮬레이션 공포탐욕지수 생성.

    실제 CNN Fear & Greed Index와 유사하게:
    - 급락 시 → Extreme Fear (0~25)
    - 급등 시 → Extreme Greed (75~100)
    - 이동평균 대비 위치, 변동성, 모멘텀 반영
    """
    close = df["close"]

    # 1) 20일 수익률 모멘텀 → 0~100 스케일
    mom_20 = close.pct_change(20).fillna(0)
    mom_score = (mom_20 - mom_20.min()) / (mom_20.max() - mom_20.min()) * 100

    # 2) 가격 vs 125일 이동평균
    sma125 = close.rolling(125).mean()
    dist = ((close - sma125) / sma125).fillna(0)
    dist_score = (dist - dist.min()) / (dist.max() - dist.min()) * 100

    # 3) 변동성 (높을수록 공포)
    vol = close.pct_change().rolling(20).std().fillna(0)
    vol_score = 100 - (vol - vol.min()) / (vol.max() - vol.min()) * 100

    # 4) 52주 고점 대비
    high_52 = close.rolling(252, min_periods=1).max()
    from_high = (close / high_52)
    high_score = (from_high - from_high.min()) / (from_high.max() - from_high.min()) * 100

    # 가중 합산
    raw = mom_score * 0.30 + dist_score * 0.25 + vol_score * 0.25 + high_score * 0.20

    # 스무딩 + 노이즈
    fg = raw.rolling(5, min_periods=1).mean()
    noise = np.random.normal(0, 3, len(fg))
    fg = np.clip(fg + noise, 0, 100)

    return pd.Series(fg, index=df.index)


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
    buy_queue = []   # (날짜, 금액) — 남은 매수 분할
    sell_queue = []  # (날짜, 비율) — 남은 매도 분할

    prev_fg_zone = "neutral"  # 이전 공포탐욕 구간
    total_invested = 0.0      # 총 매수 금액 (평단가 계산용)

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
            if capital > 100:  # 매수 가능한 자금이 있을 때만
                buy_amount_per_day = capital / split_days
                for d in range(split_days):
                    buy_queue.append(buy_amount_per_day)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "SIGNAL",
                    "action": f"극단적 공포 감지 (F&G={fg_val:.0f}) → 3일 분할매수 시작",
                    "price": price,
                    "shares": 0,
                    "amount": 0,
                    "capital": capital,
                    "total_shares": shares,
                })

        # ── Extreme Greed 진입 시: 3일 분할매도 예약 ──
        if fg_zone == "EXTREME GREED" and prev_fg_zone != "EXTREME GREED":
            if shares > 0:
                sell_shares_per_day = shares // split_days
                remainder = shares - sell_shares_per_day * split_days
                for d in range(split_days):
                    s = sell_shares_per_day + (1 if d < remainder else 0)
                    sell_queue.append(s)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "SIGNAL",
                    "action": f"극단적 탐욕 감지 (F&G={fg_val:.0f}) → 3일 분할매도 시작",
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
                    "price": price,
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
                    "price": price,
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

def print_report(result: dict):
    print()
    print("=" * 70)
    print("  STOCK PULSE - 공포탐욕지수 분할매매 백테스트")
    print("=" * 70)
    print("  종목: QQQ (Invesco QQQ Trust)")
    print("  전략: Extreme Fear → 3일 1/3 분할매수 | Extreme Greed → 3일 1/3 분할매도")
    print("  기간: 2023-01-03 ~ 2025-12-31  |  데이터: 시뮬레이션")
    print("-" * 70)

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
    print("\n" + "-" * 70)
    print("  거래 내역")
    print("-" * 70)

    trade_rows = []
    for t in result["trades"]:
        if t["type"] == "SIGNAL":
            trade_rows.append([
                t["date"], "📡", t["action"], "", "", "",
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
            pnl_str = f"${pnl:+,.2f}"
            trade_rows.append([
                t["date"], "SELL",
                t["action"],
                f"${t['price']:.2f}",
                f"{t['shares']}주",
                f"+${t['amount']:,.2f}",
                f"{t['total_shares']}주"
            ])

    print(tabulate(
        trade_rows,
        headers=["날짜", "유형", "상세", "가격", "수량", "금액", "보유"],
        tablefmt="simple",
        colalign=("left", "center", "left", "right", "right", "right", "right"),
    ))

    # 연도별 수익률
    print("\n" + "-" * 70)
    print("  연도별 성과")
    print("-" * 70)

    daily = result["daily_df"]
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
    print("\n" + "-" * 70)
    print("  공포탐욕지수 분포")
    print("-" * 70)

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

    print("\n" + "=" * 70)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  QQQ + 공포탐욕지수 데이터 생성 중...")
    df = generate_qqq_data()
    fg = generate_fear_greed(df)

    print(f"  {len(df)}일 데이터 로드 완료", end="")
    print(f" ({df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')})")

    result = run_fear_greed_backtest(df, fg, initial_capital=100_000)
    print_report(result)
