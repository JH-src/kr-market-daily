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
        'stage': g['stage'].first(),
        'fwd63': (g['sec63'].first() - g['mkt63'].first()) * 100,
        'fwd126': (g['sec126'].first() - g['mkt126'].first()) * 100,
    }).reset_index()
    s = s[s['n'] >= 5]
    s['종합(이익+신고가+수출)'] = s.groupby('date')[['영업이익 증가 중위', '신고가 근처 비율', '관련 수출 증가']].rank(pct=True).mean(axis=1)
    s['종합(이익+신고가+외국인)'] = s.groupby('date')[['영업이익 증가 중위', '신고가 근처 비율', '외국인 60일 유입']].rank(pct=True).mean(axis=1)
    return s


def main():
    s = build()
    s = s[s['date'] <= pd.Timestamp('2026-12-31')]
    feats = [c for c in s.columns if c not in ('date', 'sector', 'n', 'stage', 'fwd63', 'fwd126')]
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
    p = os.path.join(ROOT, 'docs', 'sim', 'sector_test.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write("# 섹터 판정 검증\n\n매월 말 섹터(후보 5종목 이상)마다 기준값을 만들어 이후 섹터 지수의 후보군 동일가중 대비 성과와 비교. "
                 "ic: 섹터 간 순위 상관(2011~2023 월평균), 상위3 초과: 그 기준 상위 3개 섹터의 평균 초과(%p), 24~: 2024년 이후.\n\n")
        fh.write(o.round(3).to_markdown(index=False) + "\n\n## 지금 쓰는 단계 판정별 이후 성과(전 기간)\n\n" + st.to_markdown() +
                 "\n\n2024년 이후\n\n" + st24.to_markdown() + "\n\n## 연도별 6개월 ic\n\n" + pd.DataFrame(yr).T.round(3).to_markdown() + "\n")
    print(open(p).read())


if __name__ == '__main__':
    main()
