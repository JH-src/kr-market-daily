"""종목 선정 엔진 v2: 처음부터 끝까지 걷기식(그해 이전 자료로만 정하고 그해에 적용).

1) 요소 고르기: 해마다 그 이전 달들의 요소 성적(6개월 정보계수)만 보고 고정 규칙으로 쓸 요소와 방향을 정함
   (연도별 정보계수 t값 2 이상·플러스인 해 70% 이상이면 +, 반대면 감점). 같은 비중.
2) 후보 방식(변형) 16개: 범위(전체 / 주도·초기 섹터) × 통과 조건(없음 / 고점 -30% 이내 / 영업이익 증가 / 고점 -30% 이내+변동성 상위 30% 제외)
   × 섹터당 최대 종목 수(제한 없음 / 2개). 각 변형도 1)의 요소로 매년 처음 보는 해에 적용한 성과만 쌓음.
3) 변형 고르기: 해마다 직전 3년(6개월 보유가 끝난 달까지)의 변형별 성과 평균이 가장 좋은 변형을 그해에 씀.
4) 평가: 비용 0.5% 차감, 시총 상위 후보군 동일가중 대비와 시가총액 가중 대비 둘 다. 연도별, 2024년 이후 따로.
결과: docs/sim/engine.md, 지금 쓸 설정 docs/sim/engine_current.json(리포트가 읽음).
python3 -m sim.engine2
"""
import json, os
import numpy as np
import pandas as pd
from sim import search as S
from sim.factors import F, ROOT

N = 6
SCOPES = ('전체', '주도·초기', '주도·과열', '섹터점수 상위5', '섹터점수 상위8')
GATES = {
    '없음': lambda g: pd.Series(True, index=g.index),
    '고점-30%이내': lambda g: g['_hi52'] > -30,
    '영업이익증가': lambda g: g['_op_g'] > 0,
    '고점-30%이내+고변동제외': lambda g: (g['_hi52'] > -30) & (g['_low_vol'].rank(pct=True) > 0.3),
}
CAPS = (None, 2)
SKIP = ('sec_m3',)


def zr(x):
    return (x.rank(pct=True) - 0.5).fillna(0)


def load():
    d = S.panel('theme').copy()
    if 'prsn20' in d and 'prsn20_cap' not in d:
        d['prsn20_cap'] = d['prsn20'] / d['cap'] * 100
    keys = [k for k in F if k not in SKIP]
    for k in keys:
        try:
            v = F[k][0](d)
            d['_' + k] = pd.to_numeric(pd.Series(np.asarray(v, dtype=float) if v is not None else np.nan, index=d.index), errors='coerce')
        except Exception:
            d['_' + k] = np.nan
    keys = [k for k in keys if d['_' + k].notna().mean() > 0.05]
    # 섹터 점수(섹터 판정 검증에서 꾸준했던 것): 섹터 안 영업이익 증가 중위·이익률 개선 중위·신고가 근처 비율의 섹터 간 순위 평균
    d['_near_hi'] = (d['_hi52'] > -10).astype(float)
    sg = d[d['sector'].notna() & (d['sector'] != '기타')].groupby(['date', 'sector']).agg(
        n=('code', 'size'), opg=('_op_g', 'median'), opm=('_opm_chg', 'median'), nh=('_near_hi', 'mean')).reset_index()
    sg = sg[sg['n'] >= 5]
    sg['sec_score'] = sg.groupby('date')[['opg', 'opm', 'nh']].rank(pct=True).mean(axis=1)
    sg['sec_rank'] = sg.groupby('date')['sec_score'].rank(ascending=False)
    d = d.merge(sg[['date', 'sector', 'sec_rank']], on=['date', 'sector'], how='left')
    # 시가총액 가중 기준선(후보군 전체)
    for h in (63, 126):
        cw = d.groupby('date').apply(lambda g: (g['cap'] * g[f'ret{h}']).sum() / g['cap'][g[f'ret{h}'].notna()].sum())
        d[f'cw{h}'] = d['date'].map(cw)
    return d, keys


def monthly_ic(dd, keys):
    rows = {}
    for t, g in dd.groupby('date'):
        y = g['ex126']
        r = {}
        for k in keys:
            x = g['_' + k]
            ok = x.notna() & y.notna()
            r[k] = x[ok].rank().corr(y[ok].rank()) if ok.sum() >= 30 and x[ok].nunique() >= 5 else np.nan
        rows[t] = r
    return pd.DataFrame(rows).T.sort_index()


def choose_weights(IC, Y):
    tr = IC[IC.index <= pd.Timestamp(f'{Y - 1}-06-30')]
    ann = tr.groupby(tr.index.year).mean()
    w = {}
    for k in IC.columns:
        a = ann[k].dropna()
        if len(a) < 3 or a.std() == 0:
            continue
        t = a.mean() / a.std() * np.sqrt(len(a))
        if t >= 2 and (a > 0).mean() >= 0.7:
            w[k] = 1
        elif t <= -2 and (a < 0).mean() >= 0.7:
            w[k] = -1
    return w


def pick_month(g, w, gate, cap_per_sector):
    s = sum(v * zr(g['_' + k]) for k, v in w.items())
    ok = GATES[gate](g).fillna(False)
    s = s[ok].sort_values(ascending=False)
    out, cnt = [], {}
    for c in s.index:
        sec = g.at[c, 'sector']
        if cap_per_sector and cnt.get(sec, 0) >= cap_per_sector:
            continue
        out.append(c); cnt[sec] = cnt.get(sec, 0) + 1
        if len(out) >= N:
            break
    return g.loc[out]


def scope_df(d, sc):
    if sc == '전체':
        return d
    if sc == '주도·초기':
        return d[d['stage'].isin(['주도', '초기'])]
    if sc == '주도·과열':
        return d[d['stage'].isin(['주도', '과열 주의'])]
    if sc == '섹터점수 상위5':
        return d[d['sec_rank'] <= 5]
    if sc == '섹터점수 상위8':
        return d[d['sec_rank'] <= 8]
    return d


def run():
    d, keys = load()
    years = range(2013, 2027)
    ICs = {sc: monthly_ic(scope_df(d, sc), keys) for sc in SCOPES}
    W = {sc: {Y: choose_weights(ICs[sc], Y) for Y in years} for sc in SCOPES}
    series = {}
    for sc in SCOPES:
        dd = scope_df(d, sc)
        for gate in GATES:
            for cp in CAPS:
                name = f"{sc}|{gate}|{'섹터당2' if cp else '제한없음'}"
                rows = []
                for t, g in dd.groupby('date'):
                    w = W[sc].get(t.year, {})
                    if not w or len(g) < 20:
                        continue
                    top = pick_month(g, w, gate, cp)
                    if top.empty:
                        rows.append({'date': t, 'ex63': 0.0, 'ex126': 0.0, 'cw63': 0.0, 'cw126': 0.0, 'med': np.nan, 'n': 0}); continue
                    r = {'date': t, 'n': len(top), 'med': (top['ex126'].median()) * 100}
                    for h in (63, 126):
                        rr = (top[f'ret{h}'] - S.COST).clip(-0.6, 1.5)
                        r[f'ex{h}'] = (rr - top[f'mkt{h}']).mean() * 100
                        r[f'cw{h}'] = (rr - top[f'cw{h}']).mean() * 100
                    rows.append(r)
                series[name] = pd.DataFrame(rows).set_index('date')
    # 변형 고르기: 직전 3년(그해 전년 6월까지) 평균 6개월 초과가 가장 좋은 변형
    meta, picks = [], {}
    for Y in range(2016, 2027):
        lo, hi = pd.Timestamp(f'{Y - 3}-07-01'), pd.Timestamp(f'{Y - 1}-06-30')
        sc = {k: v.loc[(v.index >= lo) & (v.index <= hi), 'ex126'].mean() for k, v in series.items()}
        sc = {k: v for k, v in sc.items() if pd.notna(v)}
        if not sc:
            continue
        best = max(sc, key=sc.get)
        picks[Y] = best
        m = series[best][series[best].index.year == Y]
        if len(m):
            meta.append(m.assign(variant=best))
    M = pd.concat(meta) if meta else pd.DataFrame()
    return d, keys, W, series, M, picks


FIXED = '주도·초기|고점-30%이내|제한없음'   # 원칙으로 고정: 섹터 먼저(주도·초기) + 크게 무너진 종목 제외. 변형을 해마다 바꾸면 오히려 흔들림


def size_blend(d, base):
    """대형주 장세 대응 시험: 직전 6개월 시총가중이 동일가중보다 앞섰으면 절반을 시총가중 지수(대형주 묶음)로 들고 감.
    직전 6개월 값은 6개월 전 판단일의 실현 성과(판단 시점에 이미 알 수 있는 값)를 씀."""
    g = d.groupby('date').agg(cw126=('cw126', 'first'), mkt126=('mkt126', 'first'))
    g['size_gap'] = (g['cw126'] - g['mkt126']) * 100
    past = g['size_gap'].shift(6)        # 월말 기준 6개월 전 판단분 = 지금 막 끝난 6개월
    b = base.join(past.rename('size_prev'), how='left')
    on = b['size_prev'] > 0
    # 시총가중 묶음의 동일가중 대비 = cw - mkt, 시총가중 대비 = 0
    idx_ex = g['size_gap'].reindex(b.index)
    b['ex126_b'] = np.where(on, 0.5 * b['ex126'] + 0.5 * idx_ex, b['ex126'])
    b['cw126_b'] = np.where(on, 0.5 * b['cw126'], b['cw126'])
    b['on'] = on
    return b


def write(d, keys, W, series, M, picks):
    p = os.path.join(ROOT, 'docs', 'sim', 'engine.md')
    L = ["# 종목 선정 엔진 v2 (전 과정 걷기식)", "",
         "요소·변형 모두 그해 이전 자료로만 정하고 그해에 적용. 매월 말 6종목, 비용 0.5% 차감, 6개월 보유. "
         "동일가중 = 후보군(시총 상위 1200) 평균 대비, 시총가중 = 후보군 시가총액 가중 평균 대비(대형주 장세 반영).", ""]
    if len(M):
        a = M.groupby(M.index.year).agg(ex63=('ex63', 'mean'), ex126=('ex126', 'mean'), cw126=('cw126', 'mean'), med=('med', 'mean'), n=('n', 'mean'))
        a['변형'] = [picks.get(y, '') for y in a.index]
        pre, post = M[M.index < '2024-01-01'], M[M.index >= '2024-01-01']
        yr = a['ex126']
        L += ["## 최종(변형까지 걷기식으로 고른 결과)", "",
              f"2016~2023: 3개월 {pre['ex63'].mean():+.2f}%p, 6개월 동일가중 대비 {pre['ex126'].mean():+.2f}%p, 시총가중 대비 {pre['cw126'].mean():+.2f}%p, 플러스 해 {(yr[yr.index < 2024] > 0).sum()}/{(yr.index < 2024).sum()}",
              f"2024년 이후: 3개월 {post['ex63'].mean():+.2f}%p, 6개월 동일가중 대비 {post['ex126'].mean():+.2f}%p, 시총가중 대비 {post['cw126'].mean():+.2f}%p", "",
              a.round(2).to_markdown(), ""]
    rows = []
    for k, v in series.items():
        v16 = v[(v.index >= '2016-01-01') & (v.index < '2024-01-01')]; v24 = v[v.index >= '2024-01-01']
        ann = v16['ex126'].groupby(v16.index.year).mean()
        rows.append({'변형': k, '16~23 6개월': v16['ex126'].mean(), '16~23 시총가중 대비': v16['cw126'].mean(), '플러스 해': f"{(ann > 0).sum()}/{len(ann)}",
                     '최악의 해': ann.min(), '24~ 6개월': v24['ex126'].mean(), '24~ 시총가중 대비': v24['cw126'].mean(), '평균 종목 수': v['n'].mean()})
    L += ["## 변형별(요소는 매년 걷기식으로 고름)", "", pd.DataFrame(rows).sort_values('16~23 6개월', ascending=False).round(2).to_markdown(index=False), ""]
    fx = series.get(FIXED)
    if fx is not None and len(fx):
        b = size_blend(d, fx[fx.index >= '2016-01-01'])
        a = b.groupby(b.index.year).agg(ex126=('ex126', 'mean'), cw126=('cw126', 'mean'), ex126_b=('ex126_b', 'mean'), cw126_b=('cw126_b', 'mean'), 대형주장세_달=('on', 'sum'))
        pre, post = b[b.index < '2024-01-01'], b[b.index >= '2024-01-01']
        L += [f"## 고정 변형({FIXED}) + 대형주 장세 섞기", "",
              "요소는 해마다 걷기식으로 고르고, 변형은 원칙으로 고정. 섞기: 직전 6개월 시총가중이 동일가중보다 앞섰으면 절반을 시총가중 묶음으로.", "",
              f"2016~2023 그대로: 동일가중 대비 {pre['ex126'].mean():+.2f}%p, 시총가중 대비 {pre['cw126'].mean():+.2f}%p / 섞기: {pre['ex126_b'].mean():+.2f}%p, {pre['cw126_b'].mean():+.2f}%p",
              f"2024년 이후 그대로: {post['ex126'].mean():+.2f}%p, {post['cw126'].mean():+.2f}%p / 섞기: {post['ex126_b'].mean():+.2f}%p, {post['cw126_b'].mean():+.2f}%p", "",
              a.round(2).to_markdown(), ""]
    L += ["## 해마다 고른 요소(주도·초기 범위)", ""] + [f"- {y}: " + ', '.join(('' if s > 0 else '감점 ') + F[k][1] for k, s in w.items()) for y, w in W['주도·초기'].items()]
    open(p, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    # 지금 쓸 설정: 올해 기준으로 다시 고름(최근 3년 변형 성과, 최신 요소)
    Y = pd.Timestamp.today().year
    lo, hi = pd.Timestamp(f'{Y - 3}-07-01'), pd.Timestamp(f'{Y}-12-31')
    sc = {k: v.loc[(v.index >= lo) & (v.index <= hi), 'ex126'].mean() for k, v in series.items()}
    sc = {k: v for k, v in sc.items() if pd.notna(v)}
    best = FIXED
    scope, gate, cp = best.split('|')
    ICs = monthly_ic(scope_df(d, scope), keys)
    w = choose_weights(ICs, Y + 1)    # 지금까지 자료 전부
    cur = {'asof': str(pd.Timestamp.today().date()), 'variant': best, 'scope': scope, 'gate': gate, 'cap_per_sector': 2 if cp == '섹터당2' else None,
           'weights': w, 'desc': {k: F[k][1] for k in w}}
    json.dump(cur, open(os.path.join(ROOT, 'docs', 'sim', 'engine_current.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n'.join(L)); print(cur)


if __name__ == '__main__':
    write(*run())
