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


# 교집합(반드시 통과해야 하는 문) + 통과한 종목 안에서 종합 점수 순
GATES = {
    '교집합: 실적 개선 + 과열 아님 → 6개 점수 순': lambda g: (g['_op_g'] > 0) & (g['_opm_chg'] > 0) & (g['_r1rel'].rank(pct=True) < 0.8) & (g['_val_surge'].rank(pct=True) < 0.8),
    '교집합: 실적 개선 + 과열 아님 + 저변동 절반 + 고점 근접 절반 → 6개 점수 순': lambda g: (g['_op_g'] > 0) & (g['_opm_chg'] > 0) & (g['_r1rel'].rank(pct=True) < 0.8) & (g['_val_surge'].rank(pct=True) < 0.8) & (g['_low_vol'].rank(pct=True) >= 0.5) & (g['_hi52'].rank(pct=True) >= 0.5),
    '교집합: 위 조건 + 외국인 60일 순매수': lambda g: (g['_op_g'] > 0) & (g['_opm_chg'] > 0) & (g['_r1rel'].rank(pct=True) < 0.8) & (g['_val_surge'].rank(pct=True) < 0.8) & (g['_low_vol'].rank(pct=True) >= 0.5) & (g['_hi52'].rank(pct=True) >= 0.5) & (g['_frgn60_cap'] > 0),
}
LIGHT = {
    'v2 + 고점 -30% 이내': lambda g: g['_hi52'] > -30,
    'v2 + 고점 -30% 이내 + 변동성 상위 30% 제외': lambda g: (g['_hi52'] > -30) & (g['_low_vol'].rank(pct=True) > 0.3),
    'v2 + 영업이익 증가 필수': lambda g: (g['_op_g'] > 0),
}
for k in GATES:
    SETS[k] = ['low_vol', 'bp', 'roe', 'op_g', 'opm_chg', 'hi52']
for k, fn in LIGHT.items():
    GATES[k] = fn
    SETS[k] = ['low_vol', 'bp', 'roe', 'op_g', 'opm_chg', 'hi52', '-r1rel', '-val_surge']


def zrank(x):
    r = x.rank(pct=True)
    return (r - 0.5).fillna(0)   # 값 없으면 중립


def main(N=6, a=None, b='2023-12-31', name='composite'):
    d = S.panel('theme').copy()
    d = d[(d['date'] >= (pd.Timestamp(a) if a else S.D0)) & (d['date'] <= pd.Timestamp(b))]
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
                if nm in GATES:
                    ok = GATES[nm](g).fillna(False)
                    s = s[ok]
                    if s.empty:
                        sc.append({'date': t, 'ex63': 0.0, 'ex126': 0.0, 'hit': np.nan, 'med': np.nan, 'n': 0}); continue
                top = g.loc[s.sort_values(ascending=False).index[:N]]
                sc.append({'date': t, 'ex63': top['ex63'].clip(-0.6, 1.5).mean() * 100, 'ex126': top['ex126'].clip(-0.6, 1.5).mean() * 100,
                           'hit': (top['ex126'] > 0).mean() * 100, 'med': top['ex126'].median() * 100, 'n': len(top)})
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
                         '최악의 해': ann.min(), '평균 종목 수': m['n'].mean(), '빈 달': int((m['n'] == 0).sum())})
            yearly[f'{scope}|{nm}'] = ann
    o = pd.DataFrame(rows)
    y = pd.DataFrame(yearly).T.round(1)
    p = os.path.join(ROOT, 'docs', 'sim', f'{name}.md')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"# 종합 점수 시험\n\n요소마다 후보군 안 백분위 순위를 같은 비중으로 더함(값 없으면 중립). 매월 말 상위 {N}종목, 비용 0.5% 차감, 상위 700 동일가중 대비(%p). {d['date'].min():%Y.%m}~{d['date'].max():%Y.%m}. 가중치를 성과에 맞춰 고르지 않음.\n\n")
        f.write(o.round(2).to_markdown(index=False) + "\n\n## 연도별 6개월 초과(%p)\n\n" + y.to_markdown() + "\n")
    print(o.round(2).to_string())


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'holdout':
        # 남겨 둔 2024년 이후 구간: 요소·조합을 고르는 데 쓰지 않은 기간에서 한 번 확인
        main(a='2024-01-01', b='2099-12-31', name='composite_holdout')
    else:
        main()
