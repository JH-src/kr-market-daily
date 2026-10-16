"""회차별 설정 묶음을 돌려 '설계 구간' 요약(월 단위 묶음 신뢰구간 포함)을 만든다.
2024년 이후(최종 확인 구간) 결과는 따로 저장만 하고 요약표에는 넣지 않는다.
python3 -m sim.grid <회차> <설정 json>   # {"이름": {설정 변경}, ...}
"""
import copy, json, os, sys
import numpy as np
import pandas as pd
from . import engine as E

DESIGN = ('2016-06-01', '2023-12-31')
SUBS = {'2016.6-18': ('2016-06-01', '2018-12-31'), '2019-23': ('2019-01-01', '2023-12-31')}


def month_stats(d, rng):
    m = d.groupby('date')['ex_univ'].mean() * 100
    if len(m) < 6:
        return np.nan, np.nan, np.nan, len(m)
    boots = [rng.choice(m.values, len(m)).mean() for _ in range(2000)]
    return m.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(m)


def block_ci(m, rng, block=6, n=2000):
    v = m.values
    k = len(v)
    if k < block * 2:
        return np.nan, np.nan
    starts = np.arange(k - block + 1)
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in rng.choice(starts, k // block + 1)])[:k]
        out.append(v[idx].mean())
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def table(res, name, rng, a, b):
    rows = []
    for h, df in res.items():
        d = df[(df['date'] >= a) & (df['date'] <= b)] if not df.empty else df
        r = {'설정': name, '기간': '3개월' if h == 63 else '6개월', '추천 수': len(d)}
        if len(d):
            mm, lo, hi, nm = month_stats(d, rng)
            mser = d.groupby('date')['ex_univ'].mean() * 100
            blo, bhi = block_ci(mser, rng, 6 if h == 126 else 3)
            top = d.nlargest(5, 'ret').index
            trim = d.drop(top).groupby('date')['ex_univ'].mean().mean() * 100
            r.update({'월 수': nm, '시장 평균 대비(월평균)': mm, '95% 구간': f'{lo:+.1f}~{hi:+.1f}',
                      '95% 구간(기간 겹침 반영)': f'{blo:+.1f}~{bhi:+.1f}', '상위 5건 제외': trim,
                      '중앙값(종목)': d['ex_univ'].median() * 100,
                      '시장 평균 이긴 비율': (d['ex_univ'] > 0).mean() * 100, '평균 수익률': d['ret'].mean() * 100,
                      '하위 10%': d['ret'].quantile(.1) * 100, '손절 비율': d['stopped'].mean() * 100})
            for k, (x, y) in SUBS.items():
                dd = d[(d['date'] >= x) & (d['date'] <= y)]
                r[k] = dd.groupby('date')['ex_univ'].mean().mean() * 100 if len(dd) else np.nan
        rows.append(r)
    return rows


def main(tag, cfg_file):
    cfg = json.load(open(cfg_file, encoding='utf-8'))
    C = E.Ctx()
    rng = np.random.default_rng(7)
    rows, hold = [], []
    for name, ch in cfg.items():
        L = copy.deepcopy(E.BASE); L.update(ch)
        res, regs = E.run(C, L, start='2016-06-01')
        rows += table(res, name, rng, *DESIGN)
        hold += table(res, name, rng, '2024-01-01', '2100-01-01')
    df = pd.DataFrame(rows)
    out = os.path.join(E.Dm.ROOT, 'docs', 'sim')
    fmt = df.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == 'f':
            fmt[c] = fmt[c].map(lambda x: '' if pd.isna(x) else f'{x:.1f}')
    md = [f'# 시뮬레이션 {tag} (설계 구간 2016.6~2023, 월말 판단)', '',
          '시장 평균 대비(월평균) = 매월 추천 종목들의 시장 평균 대비 초과수익을 달마다 평균한 뒤 다시 월끼리 평균. 95% 구간은 월 단위 재표본(2,000회).', '',
          '설정: ' + json.dumps(cfg, ensure_ascii=False), '', fmt.to_markdown(index=False)]
    open(os.path.join(out, f'{tag}.md'), 'w', encoding='utf-8').write('\n'.join(md))
    work = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
    pd.DataFrame(hold).to_csv(os.path.join(work, f'{tag}_holdout.csv'), index=False)
    log = os.path.join(out, 'log.md')
    with open(log, 'a', encoding='utf-8') as f:
        f.write(f'\n- {tag}: 시험한 설정 {len(cfg)}개 — ' + ', '.join(cfg) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
