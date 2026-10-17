"""패널에 요소 후보 추가: 자사주 취득 결정·공급계약 공시 빈도(90일), 소속 섹터 관련 수출 증가율(직전 3개월 합계 전년비, 한 달 늦게 반영).
python3 -m sim.enrich_extra  (enrich, enrich_flows 다음에)"""
import json, os, pickle
import numpy as np
import pandas as pd

WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export_growth():
    p = os.path.join(ROOT, 'data', 'customs', 'items_monthly.csv')
    mp = json.load(open(os.path.join(ROOT, 'config', 'export_map.json')))
    d = pd.read_csv(p, dtype={'yymm': str})
    pv = d.pivot(index='yymm', columns='item', values='exp_usd')
    pv.index = pd.to_datetime(pv.index + '01')
    s3 = pv.rolling(3, min_periods=3).sum()
    g = (s3 / s3.shift(12) - 1) * 100
    sec = {}
    for it, e in mp.items():
        if it.startswith('_'):
            continue
        for s in e.get('sectors', []):
            sec.setdefault(s, []).append(it)
    out = {}
    for s, its in sec.items():
        its = [i for i in its if i in pv.columns]
        a = s3[its].sum(axis=1, min_count=1)
        out[s] = (a / a.shift(12) - 1) * 100
    return pd.DataFrame(out)          # 행: 그 달(1일), 값: 그 달까지 3개월 합계 전년비


def main():
    ev = pd.read_csv(os.path.join(ROOT, 'data', 'dart', 'events_extra.csv'), dtype=str) if os.path.exists(os.path.join(ROOT, 'data', 'dart', 'events_extra.csv')) else pd.DataFrame(columns=['rcept_dt', 'stock_code', 'kind'])
    ev['dt'] = pd.to_datetime(ev['rcept_dt'], errors='coerce')
    EG = export_growth()
    for level in ('coarse', 'ksic3', 'theme'):
        pth = os.path.join(WORK, f'panel_{level}_x.pkl')
        if not os.path.exists(pth):
            continue
        p = pickle.load(open(pth, 'rb'))
        p = p.drop(columns=[c for c in ('buyback90', 'supply90', 'exp_g3') if c in p.columns])
        for kind, nm in (('buyback', 'buyback90'), ('supply', 'supply90')):
            e = ev[ev['kind'] == kind][['stock_code', 'dt']]
            cnt = []
            for t, g in p.groupby('date'):
                c = e[(e['dt'] <= t) & (e['dt'] > t - pd.Timedelta(days=90))]['stock_code'].value_counts()
                cnt.append(g['code'].map(c).fillna(0).rename(nm))
            p[nm] = pd.concat(cnt) if cnt else np.nan
            if not len(e):
                p[nm] = np.nan
        # 수출: 판단일이 속한 달의 전전달까지(발표 시차 반영) 3개월
        if level == 'theme':
            ref = (p['date'].dt.to_period('M') - 2).dt.to_timestamp()
            p['exp_g3'] = [EG[s].get(r, np.nan) if s in EG.columns else np.nan for s, r in zip(p['sector'], ref)]
        pickle.dump(p, open(pth, 'wb'))
        print(level, p[[c for c in ('buyback90', 'supply90', 'exp_g3') if c in p]].notna().mean().round(2).to_dict(), flush=True)


if __name__ == '__main__':
    main()
