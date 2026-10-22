"""미리 정한 후보 로직 몇 개를 그대로 따라가는 기록(2026-10-22 등록).
- 무작위 조합 대결에서 '작년 1등'이 이어지지 않는 것을 확인해, 가설로 정한 소수 로직만 고정해 둔다.
- 요소와 부호는 가설로 정하고 과거 성과로 고치지 않는다. 단, 요소 선택에 지난 분석 결과가 반영됐으므로 과거 성과(백테스트)는 '참고'로만 쓴다.
- 판정은 등록일 이후 실제로 쌓이는 기록(전진 기록)으로 한다. 로직을 바꾸려면 이유를 적고 새 이름으로 다시 등록한다(기존 기록은 그대로 둔다).
- 매월 첫 거래일에 다시 고르고 그달에는 그대로 둔다(하루 만에 종목이 바뀌지 않게).
python3 -m sim.candidates → docs/sim/candidates.md, data/picks/candidates.csv(전진 기록, 누적)
"""
import os
import numpy as np
import pandas as pd
from . import daily_search as D

REG = '2026-10-22'
N, CAP = 10, 2
LOGICS = {
    'A 품질·안정(전체 시장)': {'sec': None, 'fac': ['low_vol', 'hi52', 'op_g', 'opm_chg']},
    'B 품질·안정(주도 섹터 안)': {'sec': '추세', 'fac': ['low_vol', 'hi52', 'op_g', 'opm_chg']},
    'C 품질·안정+외기 60일(주도 섹터 안)': {'sec': '추세', 'fac': ['low_vol', 'hi52', 'op_g', 'opm_chg', 'fo60_cap']},
    'D 품질·안정+가치(주도 섹터 안)': {'sec': '추세', 'fac': ['low_vol', 'hi52', 'op_g', 'opm_chg', 'bp', 'ep']},
}
# 공통 관문(가설): 영업이익 증가 또는 이익률 개선, 섹터 대비 1개월 급등 상위 20%·거래대금 급증 상위 20% 제외(이미 달린 종목)


def picks_of(lg, X, S, meta):
    uni, idx = meta['uni'], meta['idx']
    ok = uni & ((X['op_g'] > 0) | (X['opm_chg'] > 0))
    ok &= ~((X['r1rel'].rank(axis=1, pct=True) > 0.8) | (X['val_surge'].rank(axis=1, pct=True) > 0.8))
    if lg['sec'] == '추세':
        secs = meta['secs']
        ok &= D.sec_to_stock((S['tp_above60'][secs] >= 0.6) & (S['tp_align'][secs] >= 0.5) & (meta['nS'][secs] >= 5), meta)
    sc = sum(X[f].where(ok).rank(axis=1, pct=True).fillna(0) for f in lg['fac']).where(ok)
    first = pd.Series(idx.to_period('M'), index=idx)
    first = first != first.shift(1)                      # 매월 첫 거래일만 고름
    sc = sc.where(np.repeat(first.values[:, None], sc.shape[1], axis=1)).ffill(limit=25)  # 그달에는 같은 점수 → 같은 종목
    sc = sc.where(uni)                                    # 상장폐지 등으로 빠진 종목은 제외
    return sc


def main():
    X, Y, S, meta = D.build()
    nm, sec = meta['names'], meta['sec']
    rows, yr, now = [], {}, []
    for name, lg in LOGICS.items():
        sc = picks_of(lg, X, S, meta)
        R, pick = D.finish({'N': N, 'cap': CAP}, sc, Y, meta)
        y = R.groupby(R.index.year)[['ex63', 'ex126', 'sx63', 'fx63']].mean()
        yr[name] = y
        pre = y.loc[2016:2023]
        rows.append({'로직': name, '3개월 16~23': pre['ex63'].mean(), '플러스 해': f"{(pre['ex63'] > 0).sum()}/{pre['ex63'].count()}",
                     '6개월 16~23': pre['ex126'].mean(), '6개월 플러스 해': f"{(pre['ex126'] > 0).sum()}/{pre['ex126'].count()}",
                     '최악(3개월)': pre['ex63'].min(), '2024~ 3개월': y.loc[2024:, 'ex63'].mean(), '안정성': R['stab'].mean()})
        last = pick.index[-1]
        codes = pick.columns[pick.iloc[-1].values]
        for c in codes:
            now.append({'date': f'{last:%Y-%m-%d}', 'logic': name, 'code': c, 'name': nm.get(c, c), 'sector': sec.get(c)})
    # 전진 기록: 같은 날·같은 로직이 이미 있으면 덮어쓰지 않음(사후 수정 방지)
    f = os.path.join(D.ROOT, 'data', 'picks', 'candidates.csv')
    old = pd.read_csv(f, dtype=str) if os.path.exists(f) else pd.DataFrame(columns=['date', 'logic', 'code', 'name', 'sector'])
    new = pd.DataFrame(now)
    have = set(zip(old['date'], old['logic']))
    new = new[[(d, l) not in have for d, l in zip(new['date'], new['logic'])]]
    log = pd.concat([old, new]).sort_values(['date', 'logic'])
    log.to_csv(f, index=False, encoding='utf-8-sig')
    # 전진 성과: 등록일 이후 기록만, 결과가 나온 것만(시장 = 후보군 동일가중, 비용 0.5%)
    fw = []
    idx = meta['idx']
    for (d, l), g in log[log['date'] >= REG].groupby(['date', 'logic']):
        t = pd.Timestamp(d)
        if t not in idx:
            continue
        for h in D.H:
            r = Y[f'ret{h}'].loc[t]
            if r.reindex(g['code']).notna().sum() == 0:
                continue
            mk = r.where(meta['uni'].loc[t]).mean()
            fw.append({'logic': l, 'h': h, 'date': d, 'ex': ((r.reindex(g['code']) - D.COST).mean() - mk) * 100})
    L = [f'# 후보 로직 전진 기록(등록 {REG})', '',
         '무작위 조합 대결에서 해마다 고른 상위 방법이 운과 구별되지 않아(그해 백분위 약 53%), 가설로 정한 소수 로직만 고정해 따라간다.',
         '공통: 영업이익 증가 또는 이익률 개선 / 섹터 대비 1개월 급등·거래대금 급증 상위 20% 제외 / 매월 첫 거래일 선정, 그달 유지 / 10종목, 섹터당 2개.',
         '주도 섹터 = 최근 3개월 중 60일선 위 날 60% 이상이면서 20일선이 60일선 위인 날 50% 이상.', '',
         '## 판정: 등록일 이후 실제 기록', '']
    if fw:
        F = pd.DataFrame(fw)
        L += [F.groupby(['logic', 'h'])['ex'].agg(['count', 'mean']).round(2).rename(columns={'count': '선정일 수', 'mean': '시장 대비 %p'}).to_markdown(), '']
    else:
        L += ['아직 결과가 나온 기록 없음(가장 빠른 결과는 등록 후 21거래일).', '']
    L += ['## 참고: 과거 성과(요소 선택에 과거 분석이 반영돼 있어 채택 근거로 쓰지 않음)', '', pd.DataFrame(rows).round(2).to_markdown(index=False), '']
    for name, y in yr.items():
        L += [f'### {name} 연도별(%p)', '', y.round(2).rename(columns={'ex63': '3개월', 'ex126': '6개월', 'sx63': '섹터 대비 3개월', 'fx63': '수급 유입 3개월'}).to_markdown(), '']
    L += [f'## 현재 선정({now[0]["date"] if now else "-"})', '']
    for name in LOGICS:
        L.append(f'- {name}: ' + ', '.join(f"{r['name']}({r['sector']})" for r in now if r['logic'] == name))
    open(os.path.join(D.ROOT, 'docs', 'sim', 'candidates.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
