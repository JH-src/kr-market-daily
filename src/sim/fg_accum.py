"""공포 구간에서만 매집(안 팖) vs 매달 적립식.
- 매달 첫 거래일에 1씩 적립(현금, 이자 0).
- 공포 매집: 공포·탐욕 지수가 기준(30, 참고 20·25·35) 이하인 날 종가 확인 → 다음 거래일 종가에 쌓인 현금 전부 매수. 팔지 않음. 비용 0.1%.
- 적립식: 적립한 날 종가에 바로 매수.
- 무작위 비교: 매수 실행일 수를 같게 맞춘 무작위 날짜에 쌓인 현금 전부 매수 1만 번.
결과: 마지막 평가액 ÷ 총 적립액, 돈의 가중 연수익(IRR), 평균 매수 단가(적립식 대비), 현금 대기 비율·최장 대기."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fear_greed as F
from sim.fg_backtest import prices

ROOT = F.ROOT; COST = 0.001; START = '2012-01-01'; TH = (20, 25, 30, 35); NRAND = 10000
rng = np.random.default_rng(11)


def irr(flows, dates, final_value, end):
    t = np.array([(d - dates[0]).days / 365.25 for d in dates]); T = (end - dates[0]).days / 365.25
    f = lambda r: np.sum(-np.array(flows) * (1 + r) ** (T - t)) + final_value
    lo, hi = -0.9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return mid


def simulate(px, contrib_days, buy_exec):
    """buy_exec: 매수 실행일(그날 종가에 쌓인 현금 전부 매수) 불리언 배열."""
    p = px.values; cash = 0.0; units = 0.0; spent = 0.0; idle = 0; waits = []; w = 0
    for i in range(len(p)):
        if contrib_days[i]:
            cash += 1
        if buy_exec[i] and cash > 0:
            units += cash * (1 - COST) / p[i]; spent += cash; cash = 0
            waits.append(w); w = 0
        if cash > 0:
            idle += 1; w += 1
    return units * p[-1] + cash, units, spent, cash, idle / len(p), max(waits + [w]), len(waits)


def main():
    fg = pd.read_csv(os.path.join(ROOT, 'data', 'fg', 'fg_daily.csv'), index_col=0, parse_dates=True)
    P = prices(); L = ['# 공포 구간에서만 매집(팔지 않음) vs 매달 적립식', '',
                      f'기간 {START}~, 매달 첫 거래일 1씩 적립, 기준 이하인 날 확인 → 다음 거래일 종가에 쌓인 현금 전부 매수, 비용 {COST:.1%}, 현금 이자 0.', '']
    for nm in ('코스피', '코스닥', '나스닥'):
        px = P[nm]; px = px[px.index >= START]
        f = fg[nm].reindex(px.index).ffill(limit=3)
        per = px.index.to_period('M'); cd = np.r_[True, per[1:] != per[:-1]]
        dates = list(px.index[cd]); n = cd.sum()
        v0, u0, s0, c0, i0, w0, b0 = simulate(px, cd, cd)                  # 적립식
        dca_cost = s0 * (1 - COST) / u0
        rows = [{'방식': '매달 적립식', '평가액/적립액': v0 / n, '연수익(IRR)': irr([1] * n, dates, v0, px.index[-1]) * 100,
                 '평균 매수단가(적립식=100)': 100.0, '매수 횟수': b0, '현금 대기 비율': i0 * 100, '최장 대기(거래일)': w0, '남은 현금': c0, '무작위 대비 백분위': np.nan}]
        for th in TH:
            sig = (f <= th).values
            ex = np.r_[False, sig[:-1]]                                # 다음 거래일 종가 체결
            v, u, s, c, idle, wmax, nb = simulate(px, cd, ex)
            k = nb                                                     # 실제 매수 횟수와 같게
            # 무작위: 실행일 수를 같게 맞춘 무작위 날짜(벡터 계산)
            C = np.cumsum(cd.astype(float)); pv = px.values; rv = np.empty(NRAND)
            for j in range(NRAND):
                e = np.sort(rng.choice(len(px), k, replace=False))
                amt = np.diff(np.r_[0.0, C[e]])
                rv[j] = np.sum(amt * (1 - COST) / pv[e]) * pv[-1] + (C[-1] - C[e[-1]])
            avg_cost = s * (1 - COST) / u if u else np.nan
            rows.append({'방식': f'{th} 이하일 때만 매집', '평가액/적립액': v / n, '연수익(IRR)': irr([1] * n, dates, v, px.index[-1]) * 100,
                         '평균 매수단가(적립식=100)': avg_cost / dca_cost * 100, '매수 횟수': nb,
                         '현금 대기 비율': idle * 100, '최장 대기(거래일)': wmax, '남은 현금': c, '무작위 대비 백분위': (rv < v).mean() * 100})
        T = pd.DataFrame(rows)
        L += [f'## {nm}', '', T.round(2).to_markdown(index=False), '']
    open(os.path.join(ROOT, 'docs', 'sim', 'fg_accum.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
