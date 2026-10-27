"""공포·탐욕 지수 기준선 매매 백테스트(2012~): 지수가 매수 기준 이하로 내려간 날 종가 확인 → 다음 거래일 종가에 지수 전액 매수,
매도 기준 이상이면 다음 거래일 종가에 전액 매도. 그 사이 보유 유지, 현금 이자 0, 매매마다 비용 0.1%.
기준선 매수 20·25·30·35 × 매도 80·75·70·65(16개) × 코스피·코스닥·나스닥.
운 판정: 같은 보유 구간 길이·쉬는 구간 길이를 무작위로 섞은 매매 1만 번과 비교(보유 비율·매매 횟수 동일).
걷기식: 해마다 그 전 해까지의 결과(2012~전년)로만 16개 중 하나를 골라 그해에 적용.
python3 -m sim.fg_backtest → docs/sim/fg_backtest.md"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fear_greed as F

ROOT = F.ROOT
BUY = (20, 25, 30, 35); SELL = (80, 75, 70, 65)
COST = 0.001; START = '2012-01-01'; NRAND = 10000
rng = np.random.default_rng(7)


def prices():
    q = F.fred('NASDAQCOM')
    return {'코스피': F.iseries('idx_kospi', '코스피'), '코스닥': F.iseries('idx_kosdaq', '코스닥'), '나스닥': q}


def holding(fg, b, s):
    tgt, p = [], 0
    for v in fg.values:
        if not np.isnan(v):
            if v <= b:
                p = 1
            elif v >= s:
                p = 0
        tgt.append(p)
    return pd.Series(tgt, index=fg.index).shift(2).fillna(0)    # t일 종가 판단 → t+1일 종가 체결 → t+2일 수익부터


def run(r, h):
    tr = h.diff().abs().fillna(h.iloc[0])
    return h * r - tr * COST


def stats(x, r):
    yrs = len(x) / 252
    eq = (1 + x).cumprod(); bh = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; bcagr = bh.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min(); bmdd = (bh / bh.cummax() - 1).min()
    return cagr, bcagr, mdd, bmdd


def trades(h, r):
    out, ent = [], None
    eq = (1 + r).cumprod()
    for d, v, pv in zip(h.index, h.values, h.shift(1).fillna(0).values):
        if v == 1 and pv == 0:
            ent = d
        if v == 0 and pv == 1 and ent is not None:
            out.append(eq.loc[:d].iloc[-2] / eq.loc[:ent].iloc[-2] - 1); ent = None
    if ent is not None:
        out.append(eq.iloc[-1] / eq.loc[:ent].iloc[-2] - 1)
    return out


def runs(h):
    v = h.values; segs = []; cur = v[0]; n = 0
    for x in v:
        if x == cur:
            n += 1
        else:
            segs.append((cur, n)); cur = x; n = 1
    segs.append((cur, n))
    return segs


def random_cagr(h, r):
    segs = runs(h)
    ins = [n for c, n in segs if c == 1]; outs = [n for c, n in segs if c == 0]
    first = segs[0][0]; rv = r.values; yrs = len(rv) / 252; res = np.empty(NRAND)
    for k in range(NRAND):
        a = rng.permutation(ins); b = rng.permutation(outs)
        seq = []; ia = ib = 0; c = first
        while ia < len(a) or ib < len(b):
            if c == 1 and ia < len(a):
                seq += [1] * a[ia]; ia += 1
            elif c == 0 and ib < len(b):
                seq += [0] * b[ib]; ib += 1
            c = 1 - c
        hh = np.array(seq[:len(rv)], float)
        tr = np.abs(np.diff(np.r_[0, hh]))
        x = hh * rv - tr * COST
        res[k] = np.prod(1 + x) ** (1 / yrs) - 1
    return res


def main():
    fg = pd.read_csv(os.path.join(ROOT, 'data', 'fg', 'fg_daily.csv'), index_col=0, parse_dates=True)
    P = prices()
    L = ['# 공포·탐욕 기준선 매매 백테스트', '',
         f'기간 {START}~, 다음 거래일 종가 체결, 비용 매매당 {COST:.1%}, 현금 이자 0, 지수(가격지수) 그대로 매매. 단순 보유 = 같은 기간 계속 보유.', '']
    allrows = []; wf = {}
    for nm in ('코스피', '코스닥', '나스닥'):
        px = P[nm]; f = fg[nm].reindex(px.index).ffill(limit=3)
        px = px[px.index >= START]; f = f.reindex(px.index)
        r = px.pct_change().fillna(0)
        daily = {}
        rows = []
        for b in BUY:
            for s in SELL:
                h = holding(f, b, s)
                x = run(r, h); daily[(b, s)] = x
                cagr, bcagr, mdd, bmdd = stats(x, r)
                tl = trades(h, r)
                rnd = random_cagr(h, r)
                yx = (1 + x).groupby(x.index.year).prod() - (1 + r).groupby(r.index.year).prod()
                rows.append({'시장': nm, '매수': b, '매도': s, '연수익': cagr * 100, '보유 연수익': bcagr * 100, '초과': (cagr - bcagr) * 100,
                             '최대낙폭': mdd * 100, '보유 최대낙폭': bmdd * 100, '투자비율': h.mean() * 100, '매매(왕복)': len(tl),
                             '승률': (np.mean([t > 0 for t in tl]) * 100 if tl else np.nan), '무작위 대비 백분위': (rnd < cagr).mean() * 100,
                             '보유 이긴 해': f'{int((yx > 0).sum())}/{len(yx)}'})
        T = pd.DataFrame(rows); allrows.append(T)
        # 걷기식: 해마다 이전 해까지 초과 연수익이 가장 큰 조합
        D = pd.DataFrame(daily); yrs = sorted(set(D.index.year)); pick = {}; wx = []
        for y in yrs:
            if y < 2015:
                continue
            past = D[D.index.year < y]; pr_ = r[r.index.year < y]
            ex = {k: (1 + past[k]).prod() ** (252 / len(past)) - (1 + pr_).prod() ** (252 / len(pr_)) for k in D.columns}
            k = max(ex, key=ex.get); pick[y] = k
            wx.append(D.loc[D.index.year == y, k])
        wx = pd.concat(wx); rr = r[r.index >= wx.index[0]]
        c1, c2, m1, m2 = stats(wx, rr)
        yx = (1 + wx).groupby(wx.index.year).prod() - (1 + rr).groupby(rr.index.year).prod()
        wf[nm] = (c1, c2, m1, m2, pick, yx)
    T = pd.concat(allrows)
    for nm in ('코스피', '코스닥', '나스닥'):
        t = T[T['시장'] == nm].drop(columns='시장')
        c1, c2, m1, m2, pick, yx = wf[nm]
        L += [f'## {nm}', '', t.round(1).to_markdown(index=False), '',
              f'걷기식(2015~, 해마다 이전 결과로 조합 선택): 연수익 {c1*100:.1f}% vs 보유 {c2*100:.1f}% (초과 {(c1-c2)*100:+.1f}%p), 최대낙폭 {m1*100:.1f}% vs {m2*100:.1f}%, 보유 이긴 해 {int((yx>0).sum())}/{len(yx)}',
              '고른 조합: ' + ', '.join(f'{y} {b}/{s}' for y, (b, s) in pick.items()), '']
    out = os.path.join(ROOT, 'docs', 'sim', 'fg_backtest.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    T.round(2).to_csv(os.path.join(ROOT, 'docs', 'sim', 'fg_backtest.csv'), index=False, encoding='utf-8-sig')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
