"""월말 시점별 판단 재료 + 이후 성과(시장·섹터·종목 고유 몫)를 한 표(패널)로 만든다.
한 번 만들면 로직 하나 평가가 1초 이내 -> 밤새 수천 가지 로직을 돌릴 수 있다.
python3 -m sim.panel   -> SIM_CACHE/panel.pkl
"""
import gc, os, pickle, sys
import numpy as np
import pandas as pd
from . import engine as E

CAP_MAX, VAL_MAX = 1200, 1500


def build(level, start='2011-01-01'):
    C = E.Ctx(level)
    P, R = C.P, C.W['ret']
    cap, close = C.W['cap'], C.close
    dates = C.dates
    rows = []
    mret = None
    for t in C.month_ends:
        t = pd.Timestamp(t)
        if t < pd.Timestamp(start):
            continue
        i = dates.get_loc(t)
        if i < 260:
            continue
        c_i = cap.iloc[i]
        ok = c_i.notna() & close.iloc[i].notna()
        cr = c_i[ok].rank(ascending=False)
        vr = C.val20.iloc[i][ok].rank(ascending=False)
        uni = cr.index[(cr <= CAP_MAX) & (vr.reindex(cr.index) <= VAL_MAX)]
        st = E.sector_stage(C, i)
        reg = E.regime(C, i)
        # 베타: 직전 250거래일, 비교 기준은 시총 상위 700 동일 비중 일간 수익률
        u700 = cr.index[cr <= 700]
        win = R.iloc[i - 249:i + 1]
        m = win[u700].mean(axis=1)
        X = win[uni]
        mc = m - m.mean()
        beta = (X.sub(X.mean()).mul(mc, axis=0)).sum() / (mc ** 2).sum()
        beta = beta.where(X.notna().sum() >= 120)
        Pi = P.iloc[:i + 1]
        r1 = (Pi.iloc[-1] / Pi.iloc[-22] - 1) * 100
        r3 = (Pi.iloc[-1] / Pi.iloc[-64] - 1) * 100
        g60 = (Pi.iloc[-1] / Pi.iloc[-60:].mean() - 1) * 100
        fin = E.fin_at(C, t)
        ev = C.ev[(C.ev['dt'] <= t) & (C.ev['dt'] > t - pd.Timedelta(days=180)) & (C.ev['kind'] == 'rights')]
        last_r = ev.groupby('stock_code')['dt'].max()
        fw = {}
        for h in (63, 126):
            a, b = i + 1, i + 1 + h
            if b >= len(dates):
                fw[h] = None; continue
            ret = P.iloc[b] / P.iloc[a] - 1
            mk = ret[u700].mean()
            sec = {s: C.secidx[s].iloc[b] / C.secidx[s].iloc[a] - 1 for s in C.secidx.columns}
            fw[h] = (ret, mk, sec)
        rsi = C.rsi.iloc[i]
        for c in uni:
            s = C.sec.get(c)
            S = st.get(s, {})
            f = fin.get(c, {})
            d = {'date': t, 'code': c, 'regime': reg, 'sector': s, 'stage': S.get('stage'), 'sec_m1': S.get('m1'), 'sec_m3': S.get('m3'),
                 'r1': r1[c], 'r3': r3[c], 'g60': g60[c], 'rsi': rsi[c], 'caprank': cr[c], 'valrank': vr.get(c),
                 'rights_days': (t - last_r[c]).days if c in last_r.index else 999,
                 'ry': f.get('ry'), 'ryp': f.get('ryp'), 'op': f.get('op'), 'opp': f.get('opp'), 'beta': beta.get(c)}
            for h in (63, 126):
                if fw[h] is None:
                    d[f'ret{h}'] = np.nan; continue
                ret, mk, sec = fw[h]
                d[f'ret{h}'] = ret[c]; d[f'mkt{h}'] = mk; d[f'sec{h}'] = sec.get(s, np.nan)
            rows.append(d)
    df = pd.DataFrame(rows)
    del C; gc.collect()
    return df


if __name__ == '__main__':
    work = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
    out = {}
    for level in (sys.argv[1:] or ['coarse', 'ksic3']):
        out[level] = build(level)
        pickle.dump(out[level], open(os.path.join(work, f'panel_{level}.pkl'), 'wb'))
        print(level, out[level].shape, flush=True)
