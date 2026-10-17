"""섹터 판정 검증: 어떤 기준으로 고른 '주도 섹터'가 실제로 이후 3·6개월 시장(후보군 동일가중)을 이겼나.
섹터(theme 분류, 후보 5종목 이상)마다 매월 말 기준값을 만들고, 섹터 간 순위 상관과 상위 3개 섹터의 초과수익을 연도별로 본다.
결과 docs/sim/sector_test.md.  python3 -m sim.sector_test"""
import os
import numpy as np
import pandas as pd
from sim import search as S
from sim.factors import ROOT


def build():
    d = S.panel('theme').copy()
    d = d[d['sector'].notna() & (d['sector'] != '기타')]
    d['op_g'] = np.where(d['opp'] > 0, d['op'] / d['opp'] - 1, np.nan)
    d['near_hi'] = (d['hi52'] > -10).astype(float) if 'hi52' in d else np.nan
    d['above0_3m'] = (d['r3'] > 0).astype(float)
    g = d.groupby(['date', 'sector'])
    s = pd.DataFrame({
        'n': g.size(),
        '섹터 1개월': g['sec_m1'].first(), '섹터 3개월': g['sec_m3'].first(),
        '종목 3개월 중위': g['r3'].median(),
        '신고가 근처 비율': g['near_hi'].mean(),
        '3개월 오른 종목 비율': g['above0_3m'].mean(),
        '영업이익 증가 중위': g['op_g'].median(),
        '이익률 개선 중위': g['opm_chg'].median() if 'opm_chg' in d else np.nan,
        '매출 증가 중위': g['ry'].median(),
        '관련 수출 증가': g['exp_g3'].first() if 'exp_g3' in d else np.nan,
        '외국인 60일 유입': g['frgn60_cap'].mean() if 'frgn60_cap' in d else np.nan,
        '저PBR(1/PBR 중위)': (1 / g['pbr'].median()) if 'pbr' in d else np.nan,
        'stage': g['stage'].first(), 'stage_ew': g['stage_ew'].first() if 'stage_ew' in d else None,
        'fwd63': (g['sec63'].first() - g['mkt63'].first()) * 100,
        'fwd126': (g['sec126'].first() - g['mkt126'].first()) * 100,
    }).reset_index()
    s = s[s['n'] >= 5]
    # 거래대금·낙폭(섹터 전체 종목 기준, 판단일까지 자료만)
    from sim import data as Dm
    W = Dm.prices()
    sec = Dm.sectors('theme').reindex(W['val'].columns)
    val, P = W['val'], W['adj']
    v20 = val.rolling(20, min_periods=15).mean(); v120 = val.rolling(120, min_periods=90).mean()
    tot20, tot120 = v20.sum(axis=1), v120.sum(axis=1)
    hi60 = P.rolling(60, min_periods=40).max(); lo20 = P.rolling(20, min_periods=15).min()
    extra = []
    for t in s['date'].unique():
        t = pd.Timestamp(t)
        if t not in val.index:
            continue
        a20 = v20.loc[t].groupby(sec).sum(); a120 = v120.loc[t].groupby(sec).sum()
        sh20 = a20 / tot20.loc[t] * 100; sh120 = a120 / tot120.loc[t] * 100
        dd = (P.loc[t] / hi60.loc[t] - 1).groupby(sec).median() * 100
        rb = (P.loc[t] / lo20.loc[t] - 1).groupby(sec).median() * 100
        surge = (v20.loc[t] / v120.loc[t]).groupby(sec).median()
        extra.append(pd.DataFrame({'date': t, '거래대금 비중': sh20, '거래대금 비중 변화': sh20 / sh120, '종목 거래대금 급증 중위': surge,
                                   '60일 고점 대비 낙폭 중위': dd, '20일 저점 대비 반등 중위': rb}).rename_axis('sector').reset_index())
    if extra:
        s = s.merge(pd.concat(extra), on=['date', 'sector'], how='left')
    s['종합(거래대금 비중 변화+신고가+이익)'] = s.groupby('date')[['거래대금 비중 변화', '신고가 근처 비율', '영업이익 증가 중위']].rank(pct=True).mean(axis=1)
    for c in s.columns:
        if c not in ('date', 'sector', 'stage', 'stage_ew'):
            s[c] = pd.to_numeric(s[c], errors='coerce')
    s['종합(이익+신고가+수출)'] = s.groupby('date')[['영업이익 증가 중위', '신고가 근처 비율', '관련 수출 증가']].rank(pct=True).mean(axis=1)
    s['종합(이익+신고가+외국인)'] = s.groupby('date')[['영업이익 증가 중위', '신고가 근처 비율', '외국인 60일 유입']].rank(pct=True).mean(axis=1)
    return s


def main():
    s = build()
    s = s[s['date'] <= pd.Timestamp('2026-12-31')]
    feats = [c for c in s.columns if c not in ('date', 'sector', 'n', 'stage', 'stage_ew', 'fwd63', 'fwd126')]
    rows, yr = [], {}
    for f in feats:
        r = {'기준': f}
        for h in ('fwd63', 'fwd126'):
            ics, tops = [], []
            for t, g in s.groupby('date'):
                ok = g[f].notna() & g[h].notna()
                if ok.sum() < 8:
                    continue
                ics.append((t, g.loc[ok, f].rank().corr(g.loc[ok, h].rank())))
                top = g[ok].nlargest(3, f)[h].mean()
                tops.append((t, top))
            if not ics:
                continue
            ic = pd.Series(dict(ics)); tp = pd.Series(dict(tops))
            pre = ic[ic.index < '2024-01-01']; ann = pre.groupby(pre.index.year).mean()
            r[f'ic{h[3:]}'] = pre.mean(); r[f'플러스 해{h[3:]}'] = f"{(ann > 0).sum()}/{len(ann)}"
            r[f'상위3 초과{h[3:]}'] = tp[tp.index < '2024-01-01'].mean()
            r[f'24~ 상위3{h[3:]}'] = tp[tp.index >= '2024-01-01'].mean()
            if h == 'fwd126':
                yr[f] = ic.groupby(ic.index.year).mean()
        rows.append(r)
    o = pd.DataFrame(rows).sort_values('ic126', ascending=False)
    st = s.groupby('stage')[['fwd63', 'fwd126']].mean().round(2)
    st24 = s[s['date'] >= '2024-01-01'].groupby('stage')[['fwd63', 'fwd126']].mean().round(2)
    if 'stage_ew' in s and s['stage_ew'].notna().any():
        se = s.groupby('stage_ew')[['fwd63', 'fwd126']].mean().round(2)
        se['해당 섹터·달 수'] = s.groupby('stage_ew').size()
        se24 = s[s['date'] >= '2024-01-01'].groupby('stage_ew')[['fwd63', 'fwd126']].mean().round(2)
        sy = s[s['stage_ew'].isin(['주도', '과열 주의'])].groupby(s['date'].dt.year)['fwd126'].mean().round(2)
        sk = s[s['stage'].isin(['주도', '과열 주의'])].groupby(s['date'].dt.year)['fwd126'].mean().round(2)
        ewtxt = ("\n\n## 동일가중 시장 기준 단계 판정별(전 기간)\n\n" + se.to_markdown() + "\n\n2024년 이후\n\n" + se24.to_markdown() +
                 "\n\n## 강한 섹터(주도+과열 주의) 연도별 6개월 초과: 코스피 기준 vs 동일가중 기준\n\n" + pd.DataFrame({'코스피 기준': sk, '동일가중 기준': sy}).to_markdown())
    else:
        ewtxt = ''
    # 교집합 시험: 거래대금 중심 섹터 ∩ 이익 증가 ∩ 신고가 비율 상위 절반 (기준 고정, 맞추지 않음)
    def combo(g):
        r = g['거래대금 비중'].rank(ascending=False)
        return g[(r <= 10) & (g['영업이익 증가 중위'] > 0) & (g['신고가 근처 비율'] >= g['신고가 근처 비율'].median())]
    def strong(g):
        return g[g['stage'].isin(['주도', '과열 주의'])]
    def strong_combo(g):
        r = g['거래대금 비중'].rank(ascending=False)
        return g[g['stage'].isin(['주도', '과열 주의']) & (r <= 10)]
    ctab = {}
    for nm, fn in (('거래대금 상위10 ∩ 이익 증가 ∩ 신고가 상위 절반', combo), ('강한 섹터(주도·과열)', strong), ('강한 섹터 ∩ 거래대금 상위10', strong_combo)):
        x = pd.concat([fn(g).assign(date=t) for t, g in s.groupby('date') if 'date' in g or True], ignore_index=True) if len(s) else pd.DataFrame()
        if x.empty:
            continue
        m = x.groupby('date')['fwd126'].mean()
        ctab[nm] = m.groupby(m.index.year).mean()
    ctxt = "\n\n## 교집합 시험(섹터 고르기, 이후 6개월 섹터 초과 %p, 해당 달 평균)\n\n" + pd.DataFrame(ctab).round(2).to_markdown() if ctab else ''
    p = os.path.join(ROOT, 'docs', 'sim', 'sector_test.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write("# 섹터 판정 검증\n\n매월 말 섹터(후보 5종목 이상)마다 기준값을 만들어 이후 섹터 지수의 후보군 동일가중 대비 성과와 비교. "
                 "ic: 섹터 간 순위 상관(2011~2023 월평균), 상위3 초과: 그 기준 상위 3개 섹터의 평균 초과(%p), 24~: 2024년 이후.\n\n")
        fh.write(o.round(3).to_markdown(index=False) + "\n\n## 지금 쓰는 단계 판정별 이후 성과(전 기간)\n\n" + st.to_markdown() +
                 "\n\n2024년 이후\n\n" + st24.to_markdown() + ewtxt + ctxt + "\n\n## 연도별 6개월 ic\n\n" + pd.DataFrame(yr).T.round(3).to_markdown() + "\n")
    print(open(p).read())


def walkforward(K=(3, 5)):
    """주도 섹터 고르기 걷기식: 해마다 그 이전 자료의 기준별 섹터 정보계수만 보고(t값 2 이상·플러스인 해 70% 이상) 쓸 기준을 정해
    그해 매월 말 섹터 점수 상위 K개를 고름. 이후 3·6개월 섹터 지수의 후보군 동일가중 대비. 지금 시점 섹터 순위도 저장."""
    import json
    s = build()
    feats = [c for c in s.columns if c not in ('date', 'sector', 'n', 'stage', 'stage_ew', 'fwd63', 'fwd126') and not c.startswith('종합')]
    ic = {}
    for t, g in s.groupby('date'):
        r = {}
        for f in feats:
            ok = g[f].notna() & g['fwd126'].notna()
            r[f] = g.loc[ok, f].rank().corr(g.loc[ok, 'fwd126'].rank()) if ok.sum() >= 8 else np.nan
        ic[t] = r
    IC = pd.DataFrame(ic).T.sort_index()

    def choose(Y):
        tr = IC[IC.index <= pd.Timestamp(f'{Y - 1}-06-30')]
        ann = tr.groupby(tr.index.year).mean()
        w = {}
        for f in feats:
            a = ann[f].dropna()
            if len(a) < 3 or a.std() == 0:
                continue
            t = a.mean() / a.std() * np.sqrt(len(a))
            if t >= 1.5 and (a > 0).mean() >= 0.7:
                w[f] = 1
            elif t <= -1.5 and (a < 0).mean() >= 0.7:
                w[f] = -1
        return w

    res, chosen = [], {}
    for Y in range(2014, 2027):
        w = choose(Y); chosen[Y] = w
        if not w:
            continue
        for t, g in s[s['date'].dt.year == Y].groupby('date'):
            sc = sum(v * g[f].rank(pct=True).fillna(0.5) for f, v in w.items())
            g = g.assign(sc=sc).sort_values('sc', ascending=False)
            r = {'date': t}
            for k in K:
                r[f'top{k}_63'] = g['fwd63'].head(k).mean(); r[f'top{k}_126'] = g['fwd126'].head(k).mean()
            r['bottom5_126'] = g['fwd126'].tail(5).mean()
            res.append(r)
    m = pd.DataFrame(res).set_index('date')
    ann = m.groupby(m.index.year).mean().round(2)
    ann['쓴 기준'] = [', '.join(('' if v > 0 else '감점 ') + f for f, v in chosen[y].items()) for y in ann.index]
    pre, post = m[m.index < '2024-01-01'], m[m.index >= '2024-01-01']
    # 지금 순위(최신 판단일, 최신 기준)
    w_now = choose(pd.Timestamp.today().year + 1)
    last = s[s['date'] == s['date'].max()].copy()
    last['sc'] = sum(v * last[f].rank(pct=True).fillna(0.5) for f, v in w_now.items()) if w_now else np.nan
    last = last.sort_values('sc', ascending=False)
    cur = {'asof': str(last['date'].max().date()), 'criteria': w_now,
           'top': [{'sector': r.sector, 'score': round(float(r.sc), 3)} for r in last.head(10).itertuples()],
           'table': last.head(12)[['sector', 'n', 'sc'] + list(w_now)].round(3).to_dict('records')}
    json.dump(cur, open(os.path.join(ROOT, 'docs', 'sim', 'sector_current.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    p = os.path.join(ROOT, 'docs', 'sim', 'sector_wf.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write("# 주도 섹터 고르기 걷기식\n\n해마다 그 이전 자료로만 기준을 정해 그해 매월 말 섹터 점수 상위 K개를 고름. 섹터 지수의 후보군 동일가중 대비 3·6개월(%p).\n"
                 "참고: 이 표의 '판단 시점'은 매월 말이며 그 달 판단에 쓴 자료(판단 후 성과)는 섞이지 않음. 판단 기준 후보는 섹터 판정 검증의 기준들.\n\n")
        fh.write(f"2014~2023: 상위3 6개월 {pre['top3_126'].mean():+.2f}%p(플러스 해 {(ann.loc[ann.index < 2024, 'top3_126'] > 0).sum()}/{(ann.index < 2024).sum()}), 상위5 {pre['top5_126'].mean():+.2f}%p, 하위5 {pre['bottom5_126'].mean():+.2f}%p\n\n"
                 f"2024년 이후: 상위3 {post['top3_126'].mean():+.2f}%p, 상위5 {post['top5_126'].mean():+.2f}%p, 하위5 {post['bottom5_126'].mean():+.2f}%p\n\n")
        fh.write(ann.to_markdown() + "\n\n## 지금 섹터 순위(" + cur['asof'] + ")\n\n" + last.head(12)[['sector', 'n', 'sc'] + list(w_now)].round(3).to_markdown(index=False) + "\n")
    print(open(p).read())


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'wf':
        walkforward()
    else:
        main()
