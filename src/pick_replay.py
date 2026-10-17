"""지금 로직으로 과거 매주 금요일(마지막 거래일) 시점에 골랐다면 무엇이었는지 재현해 data/picks/replay.csv 로 저장.
리포트의 선정 종목 추적에서 실제 선정과 구분해 보여 준다."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import report as R

def main(start='2026-01-01', end=None):
    from sim import engine as SE
    R._CTX['theme'] = R._CTX.get('theme') or SE.Ctx('theme')
    C = R._CTX['theme']
    ds = C.dates[(C.dates >= start) & (C.dates <= (pd.Timestamp(end) if end else C.dates.max()))]
    fri = pd.Series(ds, index=ds).groupby(ds.to_period('W')).max()
    rows = []
    for t in fri:
        sp = R.sim_picks_v2(t)
        for x in sp['rows']:
            rows.append({'date': t.strftime('%Y%m%d'), 'code': x['code'], 'name': x['name'], 'sector': x['sector'], 'stage': x['stage'],
                         'type': x['type'], 'close': f"{x['close']:.0f}", 'reg': sp['reg']})
        print(t.date(), sp['reg'], len(sp['rows']), sp['n_cand'], flush=True)
    out = os.path.join(R.DATA, 'picks', 'replay.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

if __name__ == '__main__':
    main(*sys.argv[1:])
