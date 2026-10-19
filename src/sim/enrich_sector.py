"""섹터 지수(theme 분류, 시총 가중)의 월말 가격 축 지표. 엔진 v3의 1층(주도 섹터) 가격 축에 쓴다.
python3 -m sim.enrich_sector → SIM_CACHE/sector_feat.pkl (date, sector, sp_slope40w, sp_gap200, sp_ret120, sp_rs10w)"""
import os, pickle
import numpy as np
import pandas as pd
from . import engine as E

WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')


def main():
    C = E.Ctx('theme')
    L = C.secidx
    k = C.kospi.reindex(L.index).ffill()
    wk = L.resample('W-FRI').last()
    ma40w = wk.rolling(40, min_periods=40).mean()
    slope40w = (ma40w / ma40w.shift(4) - 1).reindex(L.index, method='ffill')
    ma200 = L.rolling(200, min_periods=200).mean()
    rs = L.div(k, axis=0)
    rsw = rs.resample('W-FRI').last()
    rs10w = (rsw / rsw.rolling(10, min_periods=10).mean() - 1).reindex(L.index, method='ffill')
    ma20, ma60 = L.rolling(20, min_periods=20).mean(), L.rolling(60, min_periods=60).mean()
    rs20, rs60 = rs.rolling(20, min_periods=20).mean(), rs.rolling(60, min_periods=60).mean()
    q = lambda b: b.astype(float).where(ma60.notna()).rolling(60, min_periods=40).mean()   # 최근 3개월 중 조건이 맞은 날 비율(점이 아닌 흐름)
    F = {'sp_slope40w': slope40w, 'sp_gap200': L / ma200 - 1, 'sp_ret120': L / L.shift(120) - 1, 'sp_rs10w': rs10w,
         'tp_above20': q(L > ma20), 'tp_above60': q(L > ma60), 'tp_align': q(ma20 > ma60), 'tp_rs60': q(rs > rs60), 'tp_rs20': q(rs > rs20),
         'tp_ma60_slope': ma60 / ma60.shift(20) - 1, 'tp_rs_ma60_slope': rs60 / rs60.shift(20) - 1}
    me = sorted(set([pd.Timestamp(t) for t in C.month_ends] + list(L.index[-10:])))   # 월말 + 최근 10거래일(오늘 판정용)
    rows = []
    for t in me:
        if t not in L.index:
            continue
        for s in L.columns:
            if pd.isna(L.at[t, s]):
                continue
            rows.append({'date': t, 'sector': s, **{n: v.at[t, s] for n, v in F.items()}})
    out = pd.DataFrame(rows)
    os.makedirs(WORK, exist_ok=True)
    pickle.dump(out, open(os.path.join(WORK, 'sector_feat.pkl'), 'wb'))
    print(out.shape, out['date'].max())


if __name__ == '__main__':
    main()
