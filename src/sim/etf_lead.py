"""주도 섹터 판정 방법 전수 탐색(세분화 섹터 대표 ETF 기준).
매월 말 각 방법의 점수로 세부 섹터를 줄 세워 상위 K개를 '주도'로 고르고, 이후 3·6개월 성과를 같은 날 전체 세부 섹터 평균과 비교.
방법 후보: 기간별 수익률, 이동평균 이격(일·주), 이동평균 기울기, 신고가 근접, 위험조정 수익률, 거래대금 증가, ETF 자금 유입, 그리고 이들의 2개 조합.
걷기식: 해마다 그 이전 해들만 보고 가장 꾸준한 방법(연도별 성과 t값)을 골라 그해에 적용.
python3 -m sim.etf_lead → docs/sim/etf_lead.md, docs/sim/etf_lead_current.json"""
import json
import os
import itertools
import numpy as np
import pandas as pd
from sim import data as Dm
from sim.etf_trend import SEG

K = 5


def build():
    codes = sorted({c for v in SEG.values() for c in v})
    e = Dm._read('etf', ['BAS_DD', 'ISU_CD', 'TDD_CLSPRC', 'NAV', 'MKTCAP', 'ACC_TRDVAL', 'LIST_SHRS'])
    e = e[e['ISU_CD'].isin(codes)].copy()
    for c in ('TDD_CLSPRC', 'NAV', 'MKTCAP', 'ACC_TRDVAL', 'LIST_SHRS'):
        e[c] = pd.to_numeric(e[c].astype(str).str.replace(',', ''), errors='coerce')
    e['date'] = pd.to_datetime(e['BAS_DD'])
    piv = lambda v: e.pivot_table(index='date', columns='ISU_CD', values=v).sort_index()
    px, cap, tv, sh, nav = piv('TDD_CLSPRC'), piv('MKTCAP'), piv('ACC_TRDVAL'), piv('LIST_SHRS'), piv('NAV')
    cap, tv, sh, nav = [x.reindex(px.index) for x in (cap, tv, sh, nav)]
    r = px.pct_change(fill_method=None)
    r = r.where(r.abs() < 0.35)
    nav = nav.where(nav > 0, px)
    flow = sh.diff() * nav
    flow = flow.where(sh.pct_change(fill_method=None).abs() < 3)
    w = cap.shift(1)
    L, TV, FL, AUM = {}, {}, {}, {}
    for s, cs in SEG.items():
        cs = [c for c in cs if c in r.columns]
        rr, ww = r[cs], w[cs].where(r[cs].notna())
        ret = ((rr * ww).sum(axis=1) / ww.sum(axis=1)).where(ww.sum(axis=1) > 0)
        first = ret.first_valid_index()
        lv = (1 + ret.fillna(0)).cumprod()
        lv[lv.index < first] = np.nan
        L[s] = lv
        TV[s] = tv[cs].sum(axis=1, min_count=1).where(lv.notna())
        FL[s] = flow[cs].sum(axis=1, min_count=1).where(lv.notna())
        AUM[s] = cap[cs].sum(axis=1, min_count=1).where(lv.notna())
    return pd.DataFrame(L), pd.DataFrame(TV), pd.DataFrame(FL), pd.DataFrame(AUM)


def build_stock():
    """종목 기준 섹터 지수(theme 분류, 시가총액 가중). 2010년부터 섹터 수가 많아 검증 표본이 큼."""
    W = Dm.prices()
    sec = Dm.sectors('theme')
    r, cap, val, adj = W['ret'], W['cap'], W['val'], W['adj']
    w = cap.shift(1)
    cols = [c for c in r.columns if c in sec.index]
    g = sec.reindex(cols)
    L, TV, AUM, BR, HI = {}, {}, {}, {}, {}
    ma50 = adj.rolling(50, min_periods=40).mean()
    above = (adj > ma50).where(ma50.notna())
    hi = (adj >= adj.rolling(250, min_periods=120).max() * 0.999).where(adj.notna())
    hi20 = hi.astype(float).rolling(20, min_periods=1).max().where(adj.notna())
    for s_, cs in g.groupby(g).groups.items():
        cs = list(cs)
        if s_ in ('기타', None) or len(cs) < 5:
            continue
        rr, ww = r[cs], w[cs].where(r[cs].notna())
        n = rr.notna().sum(axis=1)
        ret = ((rr * ww).sum(axis=1) / ww.sum(axis=1)).where((ww.sum(axis=1) > 0) & (n >= 5))
        first = ret.first_valid_index()
        if first is None:
            continue
        lv = (1 + ret.fillna(0)).cumprod(); lv[lv.index < first] = np.nan
        L[s_] = lv
        TV[s_] = val[cs].sum(axis=1, min_count=1).where(lv.notna())
        AUM[s_] = cap[cs].sum(axis=1, min_count=1).where(lv.notna())
        BR[s_] = above[cs].mean(axis=1).where(lv.notna())
        HI[s_] = hi20[cs].mean(axis=1).where(lv.notna())
    return pd.DataFrame(L), pd.DataFrame(TV), None, pd.DataFrame(AUM), pd.DataFrame(BR), pd.DataFrame(HI)


def features(L, TV, FL, AUM, BR=None, HI=None):
    F = {}
    for n in (5, 10, 20, 40, 60, 120, 250):
        F[f'수익률 {n}일'] = L / L.shift(n) - 1
    F['수익률 6개월-1개월'] = L.shift(21) / L.shift(126) - 1
    F['수익률 12개월-1개월'] = L.shift(21) / L.shift(250) - 1
    for n in (10, 20, 40, 60, 120, 200):
        ma = L.rolling(n, min_periods=n).mean()
        F[f'{n}일선 이격'] = L / ma - 1
        F[f'{n}일선 기울기'] = ma / ma.shift(5) - 1
    wk = L.resample('W-FRI').last()
    for n in (5, 10, 20, 30, 40):
        ma = wk.rolling(n, min_periods=n).mean()
        F[f'{n}주선 이격'] = (wk / ma - 1).reindex(L.index, method='ffill')
        F[f'{n}주선 기울기'] = (ma / ma.shift(2) - 1).reindex(L.index, method='ffill')
    for n in (60, 120, 250):
        F[f'{n}일 고점 근접'] = L / L.rolling(n, min_periods=n // 2).max() - 1
    vol = L.pct_change(fill_method=None).rolling(60, min_periods=40).std()
    F['위험조정 3개월'] = (L / L.shift(63) - 1) / vol
    F['위험조정 6개월'] = (L / L.shift(126) - 1) / vol
    F['저변동(60일)'] = -vol
    tv = TV.rolling(5, min_periods=3).mean()
    F['거래대금 20일/120일'] = TV.rolling(20, min_periods=10).mean() / TV.rolling(120, min_periods=60).mean()
    F['거래대금 5일/60일'] = tv / TV.rolling(60, min_periods=30).mean()
    if FL is not None:
        a = AUM.rolling(20, min_periods=10).mean()
        F['자금유입 20일(순자산 대비)'] = FL.rolling(20, min_periods=10).sum() / a
        F['자금유입 60일(순자산 대비)'] = FL.rolling(60, min_periods=30).sum() / a
        F['순자산 3개월 증가'] = AUM / AUM.shift(63) - 1
    else:
        mk = AUM.sum(axis=1)
        sh = AUM.div(mk, axis=0)
        F['거래대금 비중 변화(20일/120일)'] = (TV.div(TV.sum(axis=1), axis=0)).rolling(20, min_periods=10).mean() / (TV.div(TV.sum(axis=1), axis=0)).rolling(120, min_periods=60).mean()
        F['거래대금 회전율(20일, 시총 대비)'] = TV.rolling(20, min_periods=10).mean() / AUM
    if BR is not None:
        F['50일선 위 종목 비율'] = BR
        F['50일선 위 비율 1개월 변화'] = BR - BR.shift(21)
        F['신고가 종목 비율(20일 내)'] = HI
    age = L.notna().cumsum()
    ok = (age >= 130) & L.notna()
    return {k: v.where(ok) for k, v in F.items()}, ok


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return list(s.groupby(idx.to_period('M')).max())


def score_table(F, L, ok, hz):
    """방법별·월별 상위 K 섹터의 초과 성과(%p) 표."""
    me = month_ends(L.index)
    fwd = (L.shift(-hz) / L - 1)
    me = [d for d in me if fwd.loc[d].notna().sum() >= 8]
    fw = fwd.loc[me]
    ex = fw.sub(fw.mean(axis=1), axis=0) * 100
    R = {}
    ranks = {k: v.loc[me].rank(axis=1, pct=True) for k, v in F.items()}
    names = list(F)
    combos = [(a,) for a in names] + list(itertools.combinations(names, 2))
    for c in combos:
        sc = sum(ranks[x] for x in c)
        sc = sc.where(ex.notna())
        top = sc.rank(axis=1, ascending=False, method='first') <= K
        top = top & sc.notna()
        n = top.sum(axis=1)
        v = (ex.where(top).sum(axis=1) / n).where(n >= 3)
        R[' + '.join(c)] = v
    # 추세 필터 변형: 상위 K 중 가격이 10주선·60일선 위인 것만(없으면 현금 대신 0으로 둠: 초과 0)
    return pd.DataFrame(R), ex


def yearly(R):
    return R.groupby(R.index.year).mean()


def walkforward(R, start=2014, min_years=3):
    Y = yearly(R)
    rows, pick = [], {}
    for y in Y.index:
        if y < start:
            continue
        past = Y.loc[Y.index < y].dropna(axis=1, thresh=min_years)
        if past.shape[1] == 0:
            continue
        t = past.mean() / (past.std(ddof=1) / np.sqrt(past.count()))
        pos = (past > 0).mean()
        cand = t[(pos >= 0.7)].sort_values(ascending=False)
        if cand.empty:
            cand = t.sort_values(ascending=False)
        best = cand.index[0]
        top5 = list(cand.index[:5])
        pick[y] = best
        rows.append({'year': y, '고른 방법': best, '그해 성과': Y.loc[y, best], '상위5 평균 성과': Y.loc[y, top5].mean(),
                     '과거 t값': round(t[best], 2)})
    return pd.DataFrame(rows)


def main(mode='etf'):
    if mode == 'stock':
        L, TV, FL, AUM, BR, HI = build_stock()
        F, ok = features(L, TV, FL, AUM, BR, HI)
        tag, title = '_stock', '종목 기준 섹터 지수(theme 분류)'
    else:
        L, TV, FL, AUM = build()
        F, ok = features(L, TV, FL, AUM)
        tag, title = '', '세분화 섹터 대표 ETF'
    out = {}
    md = [f'# 주도 섹터 판정 방법 전수 탐색({title})', '',
          f'세부 섹터 {L.shape[1]}개, 방법 {len(F)}개와 2개 조합(순위 합), 매월 말 상위 {K}개 섹터 → 이후 성과 - 전체 세부 섹터 평균(%p).',
          '걷기식: 해마다 이전 해들의 연도별 성과로 t값을 내고(플러스 해 70% 이상 우선) 1등 방법을 그해에 적용. 상위5 평균은 1등 우연을 줄이려고 같이 봄.', '']
    for hzn, hz in (('3개월', 63), ('6개월', 126)):
        R, ex = score_table(F, L, ok, hz)
        Y = yearly(R)
        full = pd.DataFrame({'평균': Y.mean(), '플러스 해': (Y > 0).sum().astype(str) + '/' + Y.count().astype(str),
                             't값': Y.mean() / (Y.std(ddof=1) / np.sqrt(Y.count())),
                             '2016~23': Y.loc[2016:2023].mean(), '2024~': Y.loc[2024:].mean()})
        full = full[Y.count() >= 8].sort_values('t값', ascending=False)
        single = full[~full.index.str.contains(r' \+ ')]
        wf = walkforward(R)
        out[hzn] = {'wf': wf, 'full': full}
        md += [f'## {hzn} 뒤', '', '### 걷기식 결과', '',
               f"평균 {wf['그해 성과'].mean():+.2f}%p, 플러스 해 {(wf['그해 성과'] > 0).sum()}/{len(wf)} · 상위5 평균 {wf['상위5 평균 성과'].mean():+.2f}%p, 플러스 해 {(wf['상위5 평균 성과'] > 0).sum()}/{len(wf)}", '',
               wf.round(2).to_markdown(index=False), '',
               '### 전 기간 단일 방법 순위(참고, 사후 결과)', '', single.round(2).head(25).to_markdown(), '',
               '### 전 기간 조합 상위 25(참고, 사후 결과)', '', full.round(2).head(25).to_markdown(), '']
    # 현재 판정: 6개월 걷기식이 고른 최근 방법 + 상위 방법들로 오늘 순위
    wf6 = out['6개월']['wf']; best = wf6.iloc[-1]['고른 방법']
    last = L.index.max()
    rk = {k: v.loc[last].rank(pct=True) for k, v in F.items()}
    sc = sum(rk[x] for x in best.split(' + '))
    cur = sc.dropna().sort_values(ascending=False)
    md += [f'## 현재 판정({last:%Y-%m-%d}) · 방법: {best}', '', cur.round(2).to_frame('점수').to_markdown(), '']
    open(f'../docs/sim/etf_lead{tag}.md', 'w').write('\n'.join(md))
    json.dump({'asof': f'{last:%Y-%m-%d}', 'method': best, 'top': list(cur.index[:K]), 'score': cur.round(3).to_dict()},
              open(f'../docs/sim/etf_lead{tag}_current.json', 'w'), ensure_ascii=False, indent=1)
    print('\n'.join(md[:60]))


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'etf')
