"""켈리 비중으로 지수 보유 비중 조절 (2026-11-09 사전 등록, 결과 보고 기준 바꾸지 않음)

- 대상: 코스피, 코스닥 지수(가격지수, 배당 미반영). 매매 2012-01부터.
- 매달 마지막 거래일 종가까지의 자료만으로 비중을 정해 다음 거래일부터 한 달 적용.
- 켈리 비중 f = 일평균 수익률 / 일간 수익률 분산 (연속형 켈리, 무위험금리 0 가정)
  추정 기간 3가지(최근 1년, 최근 3년, 2010년부터 그 시점까지 전체) × 풀 켈리·하프 켈리.
- 비중 범위 0~2배(공매도 없음, 레버리지 최대 2배). 1배 넘는 부분은 연 4% 차입비용, 현금 이자 0.
- 매매비용: 비중 변화분의 0.1%.
- 비교: 단순 보유(1배).
python3 -m sim.kelly_index → docs/sim/kelly_index.md"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fear_greed as F

ROOT = F.ROOT
START = '2012-01-01'; COST = 0.001; BORROW = 0.04; CAP = 2.0
WINS = {'1년': 252, '3년': 756, '전체': None}


def stats(r):
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(252)
    return cagr, mdd, vol, eq


def main():
    L = ['# 켈리 비중으로 지수 보유 비중 조절', '', __doc__.split('\n\n')[1], '']
    curves = {}
    for nm, idx in (('코스피', ('idx_kospi', '코스피')), ('코스닥', ('idx_kosdaq', '코스닥'))):
        px = F.iseries(*idx)
        r = px.pct_change().dropna()
        d = r.index
        month_end = d[np.r_[d[1:].to_period('M') != d[:-1].to_period('M'), True]]
        rows = []
        trade = r[r.index >= START]
        bh = trade.copy()
        c, m, v, eq = stats(bh)
        rows.append({'방식': '단순 보유(1배)', '연수익': c * 100, '최대낙폭': m * 100, '연변동성': v * 100, '평균 비중': 100.0, '1배 초과 달 비율': 0.0, '0배(현금) 달 비율': 0.0})
        curves[(nm, '단순 보유')] = eq
        for wn, w in WINS.items():
            raw = {}
            for me in month_end:
                past = r.loc[:me]
                if w:
                    past = past.iloc[-w:]
                if len(past) < 200:
                    continue
                raw[me] = past.mean() / past.var()
            raw = pd.Series(raw)
            for kn, k in (('풀 켈리', 1.0), ('하프 켈리', 0.5)):
                f = (raw * k).clip(0, CAP)
                # 월말 결정 → 다음 거래일부터 적용
                wd = f.reindex(r.index).ffill().shift(1)
                wd = wd[wd.index >= START].fillna(0)
                tr = trade.reindex(wd.index)
                turn = wd.diff().abs().fillna(wd.iloc[0])
                pr = wd * tr - (wd - 1).clip(lower=0) * BORROW / 252 - turn * COST
                c, m, v, eq = stats(pr)
                fm = f[f.index >= pd.Timestamp(START) - pd.offsets.MonthEnd(1)]
                rows.append({'방식': f'{kn}·{wn}', '연수익': c * 100, '최대낙폭': m * 100, '연변동성': v * 100,
                             '평균 비중': wd.mean() * 100, '1배 초과 달 비율': (fm > 1).mean() * 100, '0배(현금) 달 비율': (fm == 0).mean() * 100})
                curves[(nm, f'{kn}·{wn}')] = eq
        T = pd.DataFrame(rows)
        yrs = trade.index.year
        L += [f'## {nm} ({trade.index[0].date()}~{trade.index[-1].date()})', '', T.round(1).to_markdown(index=False), '']
    pd.DataFrame(curves).to_csv(os.path.join(ROOT, 'docs', 'sim', 'kelly_index_curves.csv'), encoding='utf-8-sig')
    open(os.path.join(ROOT, 'docs', 'sim', 'kelly_index.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
