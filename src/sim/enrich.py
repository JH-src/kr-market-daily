"""패널에 밸류·질·가격 흐름 지표를 덧붙인다(모두 판단 시점까지 공시·거래된 자료만).
python3 -m sim.enrich  -> SIM_CACHE/panel_<분류>_x.pkl
"""
import os, pickle
import numpy as np
import pandas as pd
from . import data as Dm

WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')


def ttm_table(f, cols):
    """종목·분기 순서로 정렬된 표에서 연속 4개 분기 합(TTM)과 그 시점(가장 늦은 공시일)."""
    f = f.sort_values(['stock_code', 'ord']).copy()
    g = f.groupby('stock_code')
    for c in cols:
        f[c + '_ttm'] = g[c].transform(lambda x: x.rolling(4, min_periods=4).sum())
    f['contig'] = g['ord'].transform(lambda x: x.diff(3)) == 3
    f['avail4'] = g['avail'].transform(lambda x: x.rolling(4, min_periods=4).max()) if False else f['avail']
    for c in cols:
        f.loc[~f['contig'], c + '_ttm'] = np.nan
    return f


def asof_join(panel, tab, cols):
    """각 (종목, 판단일)에 대해 공시일 <= 판단일인 마지막 행의 값을 붙인다."""
    tab = tab.dropna(subset=['avail']).sort_values('avail')
    out = []
    for code, g in panel.groupby('code'):
        t = tab[tab['stock_code'] == code]
        if t.empty:
            out.append(g); continue
        m = pd.merge_asof(g.sort_values('date'), t[['avail'] + cols].sort_values('avail'), left_on='date', right_on='avail', direction='backward')
        stale = (m['date'] - m['avail']).dt.days > 200
        m.loc[stale, cols] = np.nan
        out.append(m.drop(columns=['avail']))
    return pd.concat(out, ignore_index=True)


def main():
    W = pickle.load(open(os.path.join(WORK, 'prices.pkl'), 'rb'))
    P, cap, R, val = W['adj'], W['cap'], W['ret'], W['val']
    fq = Dm.fin_quarters()
    fq = ttm_table(fq, ['rev', 'op'])
    fq['opm_chg'] = (fq['op'] / fq['rev'] - fq['op_py'] / fq['rev_py']) * 100
    fx = Dm.fin_ext()
    if len(fx):
        fx = ttm_table(fx, ['ni'])
    for level in ('coarse', 'ksic3', 'theme'):
        if not os.path.exists(os.path.join(WORK, f'panel_{level}.pkl')):
            continue
        p = pickle.load(open(os.path.join(WORK, f'panel_{level}.pkl'), 'rb'))
        dates = sorted(p['date'].unique())
        feats = []
        for t in dates:
            i = P.index.get_loc(t)
            codes = p.loc[p['date'] == t, 'code'].values
            Pi = P.iloc[:i + 1][codes]
            d = pd.DataFrame({'date': t, 'code': codes})
            d['cap'] = cap.iloc[i][codes].values
            d['mom12_1'] = ((Pi.iloc[-22] / Pi.iloc[-253] - 1) * 100).values if i >= 253 else np.nan
            d['hi52'] = ((Pi.iloc[-1] / Pi.iloc[-252:].max() - 1) * 100).values
            d['vol60'] = (R.iloc[i - 59:i + 1][codes].std() * np.sqrt(250) * 100).values
            d['val_surge'] = (val.iloc[i - 19:i + 1][codes].mean() / val.iloc[i - 119:i + 1][codes].mean()).values
            feats.append(d)
        F = pd.concat(feats, ignore_index=True)
        p = p.merge(F, on=['date', 'code'], how='left')
        p = asof_join(p, fq.rename(columns={}), ['rev_ttm', 'op_ttm', 'opm_chg'])
        if len(fx):
            p = asof_join(p, fx, ['ni_ttm', 'eq', 'debt'])
        else:
            p['ni_ttm'] = p['eq'] = p['debt'] = np.nan
        pos = lambda x: x.where(x > 0)
        p['psr'] = p['cap'] / pos(p['rev_ttm'])
        p['por'] = p['cap'] / pos(p['op_ttm'])
        p['per'] = p['cap'] / pos(p['ni_ttm'])
        p['pbr'] = p['cap'] / pos(p['eq'])
        p['roe'] = p['ni_ttm'] / pos(p['eq']) * 100
        p['debt_ratio'] = p['debt'] / pos(p['eq']) * 100
        for c in ('per', 'pbr', 'por', 'psr'):
            p[c + '_secpct'] = p.groupby(['date', 'sector'])[c].rank(pct=True)   # 섹터 안 상대 위치(낮을수록 쌈)
        pickle.dump(p, open(os.path.join(WORK, f'panel_{level}_x.pkl'), 'wb'))
        print(level, p.shape, p[['per', 'pbr', 'por', 'roe', 'mom12_1', 'hi52']].notna().mean().round(2).to_dict(), flush=True)


if __name__ == '__main__':
    main()
