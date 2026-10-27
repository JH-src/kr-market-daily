"""매달 100 적립: 50은 월초 주도 ETF 매수, 50은 모아 두었다가 공포·탐욕(코스피)이 30 이하에서 30 위로 올라온 날 확인 → 다음 거래일 종가에 그때의 주도 ETF 매수. 팔지 않음.
비교: 매달 100을 월초에 주도 ETF 매수 / 매달 100 코스피200 ETF / 50+50 방식을 코스피200 ETF로.
주도 ETF(매수일 전날까지 자료): 종가가 20일선·50일선 위이고 상장좌수가 20거래일 전보다 늘어난 섹터 ETF(sim/etf_trend.SEG) 중 좌수 증가율 상위 3개, 같은 비중. 없으면 코스피200 ETF.
가격은 ETF 종가 수익률을 이어 붙인 지수(하루 ±35% 넘는 변동은 분할 등으로 보고 제외), 분배금 미반영(모든 방식 동일). 비용 0.1%, 현금 이자 0. 2012-01~.
python3 -m sim.fg_split_buy → docs/sim/fg_split_buy.md"""
import os
import numpy as np
import pandas as pd
from . import data as Dm
from .etf_trend import SEG
from .fg_accum import irr

ROOT = Dm.ROOT; COST = 0.001; START = '2012-01-01'; K200 = '069500'; TH = 30


def main():
    codes = sorted({c for v in SEG.values() for c in v} | {K200})
    e = Dm._read('etf', ['BAS_DD', 'ISU_CD', 'TDD_CLSPRC', 'LIST_SHRS'])
    e = e[e['ISU_CD'].isin(codes)].drop_duplicates(['BAS_DD', 'ISU_CD'], keep='last')
    e['date'] = pd.to_datetime(e['BAS_DD'])
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
    px = e.assign(v=num(e['TDD_CLSPRC'])).pivot_table(index='date', columns='ISU_CD', values='v', aggfunc='last').sort_index()
    sh = e.assign(v=num(e['LIST_SHRS'])).pivot_table(index='date', columns='ISU_CD', values='v', aggfunc='last').reindex(px.index)
    r = px.pct_change(fill_method=None); r = r.where(r.abs() < 0.35)
    lv = (1 + r.fillna(0)).cumprod().where(px.notna())             # 상장 기간만 값
    sig_ok = (px > px.rolling(20, min_periods=20).mean()) & (px > px.rolling(50, min_periods=50).mean()) & (sh > sh.shift(20))
    growth = (sh / sh.shift(20) - 1).where(sig_ok)
    sec_codes = [c for c in codes if c != K200]
    fg = pd.read_csv(os.path.join(ROOT, 'data', 'fg', 'fg_daily.csv'), index_col=0, parse_dates=True)['코스피'].reindex(px.index).ffill(limit=3)
    d = px.index[px.index >= START]; pos = {x: i for i, x in enumerate(px.index)}
    per = d.to_period('M'); first = d[np.r_[True, per[1:] != per[:-1]]]
    below = fg <= TH
    sig = (~below) & below.shift(1, fill_value=False) & fg.notna()
    sig_days = [x for x in d if sig.loc[x]]
    exec_days = {px.index[pos[x] + 1] for x in sig_days if pos[x] + 1 < len(px.index)}

    def leaders(day):
        prev = px.index[pos[day] - 1]                                  # 매수일 전날까지 자료로 판단
        g = growth.loc[prev, sec_codes].dropna().sort_values(ascending=False)
        g = g[[lv.loc[day, c] == lv.loc[day, c] for c in g.index]]    # 매수일에 가격 있는 것만
        return list(g.index[:3]) or [K200]

    def run(split, leader=True):
        units = {}; cash = 0.0; flows = []; log = []; firsts = set(first)
        for day in d:
            if day in firsts:
                flows.append(day)
                now = 100 * (1 - split)
                cash += 100 * split
                tg = leaders(day) if leader else [K200]
                for c in tg:
                    units[c] = units.get(c, 0) + now / len(tg) * (1 - COST) / lv.loc[day, c]
            if split and day in exec_days and cash > 0:
                tg = leaders(day) if leader else [K200]
                for c in tg:
                    units[c] = units.get(c, 0) + cash / len(tg) * (1 - COST) / lv.loc[day, c]
                log.append((day.date(), round(cash), tg)); cash = 0.0
        last = d[-1]
        val = sum(u * (lv.loc[last, c] if pd.notna(lv.loc[last, c]) else lv[c].loc[:last].dropna().iloc[-1]) for c, u in units.items()) + cash
        n = len(flows)
        return val, val / (100 * n), irr([1] * n, flows, val / 100, last), cash, log

    rows = []
    for nm, split, leader in (('매달 100 주도 ETF(기본)', 0.0, True), ('50 월초 + 50 공포 회복 시 주도 ETF', 0.5, True),
                              ('매달 100 코스피200', 0.0, False), ('50 월초 + 50 공포 회복 시 코스피200', 0.5, False)):
        val, mult, ir, cash, log = run(split, leader)
        rows.append({'방식': nm, '평가액/적립액': mult, '연수익(IRR)': ir * 100, '남은 현금 비율': cash / val * 100, '공포 회복 매수': len(log)})
        if split and leader:
            buylog = log
    T = pd.DataFrame(rows)
    md = ['# 50 월초 + 50 공포 회복 시 매수(팔지 않음)', '', __doc__.split('\n')[0], '', T.round(2).to_markdown(index=False), '',
          f'공포 회복 신호(30 이하→30 위) {len(sig_days)}회. 주도 ETF 방식의 공포 회복 매수 기록:', '']
    nmap = e.drop_duplicates('ISU_CD').set_index('ISU_CD')
    md += [f'- {d_} 현금 {c_}: ' + ', '.join(tg) for d_, c_, tg in buylog]
    open(os.path.join(ROOT, 'docs', 'sim', 'fg_split_buy.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md[:12]))


if __name__ == '__main__':
    main()
