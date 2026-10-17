"""요소 성적표에서 13년 내내 꾸준했던 요소만 같은 비중으로 묶은 종합 점수 시험(가중치 맞추기 없음 = 과적합 여지 작음).
매월 말 상위 N종목, 다음 날 종가 매수, 3·6개월 시장(상위 700 동일가중) 대비. 연도별로 정리해 docs/sim/composite.md.
python3 -m sim.composite"""
import os
import numpy as np
import pandas as pd
from sim import search as S
from sim.factors import F, ROOT

SETS = {
    '꾸준한 요소 6개': ['low_vol', 'bp', 'roe', 'op_g', 'opm_chg', 'hi52'],
    '6개 + 단기 과열 감점': ['low_vol', 'bp', 'roe', 'op_g', 'opm_chg', 'hi52', '-r1rel', '-val_surge'],
    '실적·밸류만': ['bp', 'roe', 'op_g', 'opm_chg'],
    '실적만': ['op_g', 'opm_chg'],
    '6개 + 외국인 60일': ['low_vol', 'bp', 'roe', 'op_g', 'opm_chg', 'hi52', 'frgn60_cap'],
    '지금 리포트 방식(섹터 대비 3개월)': ['rs3'],
}


def zrank(x):
    r = x.rank(pct=True)
    return (r - 0.5).fillna(0)   # 값 없으면 중립


def main(N=6):
    d = S.panel('theme').copy()
    d = d[(d['date'] >= S.D0) & (d['date'] <= pd.Timestamp('2023-12-31'))]
    if 'prsn20' in d and 'prsn20_cap' not in d:
        d['prsn20_cap'] = d['prsn20'] / d['cap'] * 100
    for k, (f, _) in F.items():
        try:
            v = f(d)
            d['_' + k] = pd.to_numeric(pd.Series(np.asarray(v, dtype=float) if v is not None else np.nan, index=d.index), errors='coerce')
        except Exception:
            d['_' + k] = np.nan
    rows, yearly = [], {}
    for scope in ('전체', '주도·초기 섹터'):
        dd = d if scope == '전체' else d[d['stage'].isin(['주도', '초기'])]
        for nm, fs in SETS.items():
            sc = []
            for t, g in dd.groupby('date'):
                if len(g) < 20:
                    continue
                s = sum((-1 if f.startswith('-') else 1) * zrank(g['_' + f.lstrip('-')]) for f in fs)
                top = g.loc[s.sort_values(ascending=False).index[:N]]
                sc.append({'date': t, 'ex63': top['ex63'].clip(-0.6, 1.5).mean() * 100, 'ex126': top['ex126'].clip(-0.6, 1.5).mean() * 100,
                           'hit': (top['ex126'] > 0).mean() * 100, 'med': top['ex126'].median() * 100})
            m = pd.DataFrame(sc).set_index('date')
            if m.empty:
                continue
            ann = m['ex126'].groupby(m.index.year).mean()
            # 2016년 이후만(실적 자료가 있는 기간) 따로
            m16 = m[m.index >= '2016-01-01']
            a16 = m16['ex126'].groupby(m16.index.year).mean()
            rows.append({'범위': scope, '조합': nm, '3개월': m['ex63'].mean(), '6개월': m['ex126'].mean(), '6개월 중위': m['med'].mean(),
                         '이긴 비율': m['hit'].mean(), '플러스 해': f"{(ann > 0).sum()}/{len(ann)}",
                         '16~23 6개월': m16['ex126'].mean(), '16~23 플러스 해': f"{(a16 > 0).sum()}/{len(a16)}",
                         '최악의 해': ann.min()})
            yearly[f'{scope}|{nm}'] = ann
    o = pd.DataFrame(rows)
    y = pd.DataFrame(yearly).T.round(1)
    p = os.path.join(ROOT, 'docs', 'sim', 'composite.md')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"# 종합 점수 시험\n\n요소마다 후보군 안 백분위 순위를 같은 비중으로 더함(값 없으면 중립). 매월 말 상위 {N}종목, 비용 0.5% 차감, 상위 700 동일가중 대비(%p). 2011~2023. 가중치를 성과에 맞춰 고르지 않음.\n\n")
        f.write(o.round(2).to_markdown(index=False) + "\n\n## 연도별 6개월 초과(%p)\n\n" + y.to_markdown() + "\n")
    print(o.round(2).to_string())


if __name__ == '__main__':
    main()
