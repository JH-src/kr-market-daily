"""엔진 v3 — 주도 섹터 안에서, 외국인·기관이 몰리기 전의 대표주. 전 과정 완전 걷기식(docs/design_v5.md).
1층: 섹터 네 축(가격·수급·이익·확산)을 섹터 간 순위로 합산 → 주도·과열 단계 섹터 중 상위 K
2층: 관문(영업이익 증가 또는 이익률 개선, 증자 90일, 이미 달린 종목 제외) → 요소 점수(부호·채택은 해마다 걷기식) → 구조별 선정
변형: 축 조합 4 × K 3 × 구조(대표+차기[대표 정의 5 × 덜오름 2] / 상위2 / 섹터당1) = 144, 기준(v2 고정 변형) 1
2단 걷기식: 해마다 이전 해들의 변형 성과(결과 확인 시점 차단)로 변형을 고름.
python3 -m sim.engine3 → docs/sim/engine3.md, docs/sim/engine3_current.json
"""
import json, os, pickle, time
import numpy as np
import pandas as pd
from . import engine2 as E2
from . import search as S
from .factors import F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
AXES = {'가격': ['sp_slope40w', 'sp_gap200', 'sp_ret120', 'sp_rs10w'], '수급': ['sf_frgn60', 'sf_orgn60', 'sf_pension60'],
        '이익': ['se_opg', 'se_opm', 'se_exp'], '확산': ['sb_nearhi', 'sb_above60']}
AXIS_SETS = {'가격만': ['가격'], '가격+수급': ['가격', '수급'], '가격+수급+이익': ['가격', '수급', '이익'], '네축': ['가격', '수급', '이익', '확산']}
KS = (2, 3, 5, 99)          # 99 = 주도·과열 섹터 전부(점수 상위 6종목, 섹터당 2)
REPS = {'시총1위': ('sec_cap_rank', True), '거래대금1위': ('sec_val_rank', True), '보유비율1위': ('frgn_hold', False),
        '외인기관60일1위': ('fo60', False), '시총거래대금합': ('rep_combo', True)}
STAGES = ('주도', '과열 주의')
Y0, Y1 = 2014, 2026          # 1단(요소·구조 적용) 시작, 마지막 해
WF0 = 2017                   # 2단(변형 고르기) 시작
LAG = pd.Timedelta(days=190)  # 6개월 결과가 확인되기까지


def factorize(df, keys):
    df = df.copy()
    df['rs3'] = df['r3'] - df['sec_m3']; df['r1rel'] = df['r1'] - df['sec_m1']; df['accel_d'] = df['ry'] - df['ryp']
    for k in keys:
        try:
            v = F[k][0](df)
            df['_' + k] = pd.to_numeric(pd.Series(np.asarray(v, dtype=float) if v is not None else np.nan, index=df.index), errors='coerce')
        except Exception:
            df['_' + k] = np.nan
    return df


def load():
    d, keys = E2.load()
    sf = pickle.load(open(os.path.join(WORK, 'sector_feat.pkl'), 'rb'))
    d = d.merge(sf, on=['date', 'sector'], how='left')
    for c in ('frgn60', 'orgn60', 'pension60', 'frgn_hold', 'sec_cap_rank', 'sec_val_rank', 'rep_combo', 'exp_g3'):
        if c not in d:
            d[c] = np.nan
    d['fo60'] = d['frgn60'].fillna(0) + d['orgn60'].fillna(0)
    d['fo60'] = d['fo60'].where(d['frgn60'].notna() | d['orgn60'].notna())
    raw = pickle.load(open(os.path.join(WORK, 'panel_theme_x.pkl'), 'rb'))
    raw = factorize(raw, keys)
    raw = raw.merge(sf, on=['date', 'sector'], how='left')
    for c in ('frgn60', 'orgn60', 'pension60', 'frgn_hold', 'sec_cap_rank', 'sec_val_rank', 'rep_combo', 'exp_g3'):
        if c not in raw:
            raw[c] = np.nan
    raw['fo60'] = raw['frgn60'].fillna(0) + raw['orgn60'].fillna(0)
    return d, keys, raw


def sector_table(g):
    """한 월말의 섹터 축 점수(주도·과열 단계, 5종목 이상)."""
    x = g[g['sector'].notna() & (g['sector'] != '기타') & g['stage'].isin(STAGES)]
    if x.empty:
        return pd.DataFrame()
    opg = (x['_op_g'] > 0).where(x['_op_g'].notna())
    t = pd.DataFrame({'sector': x['sector'], 'cap': x['cap'], 'frgn60': x['frgn60'], 'orgn60': x['orgn60'], 'pension60': x['pension60'],
                      'opg': opg, 'opm': x['_opm_chg'], 'exp': x['exp_g3'], 'nearhi': (x['_hi52'] > -10).where(x['_hi52'].notna()), 'above60': (x['g60'] > 0),
                      'sp_slope40w': x['sp_slope40w'], 'sp_gap200': x['sp_gap200'], 'sp_ret120': x['sp_ret120'], 'sp_rs10w': x['sp_rs10w']})
    a = t.groupby('sector').agg(n=('cap', 'size'), cap=('cap', 'sum'), frgn60=('frgn60', 'sum'), orgn60=('orgn60', 'sum'), pension60=('pension60', 'sum'),
                                se_opg=('opg', 'mean'), se_opm=('opm', 'median'), se_exp=('exp', 'mean'), sb_nearhi=('nearhi', 'mean'), sb_above60=('above60', 'mean'),
                                sp_slope40w=('sp_slope40w', 'first'), sp_gap200=('sp_gap200', 'first'), sp_ret120=('sp_ret120', 'first'), sp_rs10w=('sp_rs10w', 'first'))
    a = a[a['n'] >= 5]
    if a.empty:
        return a
    for c in ('frgn60', 'orgn60', 'pension60'):
        a['sf_' + c] = a[c] / a['cap'] * 100
        a.loc[a[c] == 0, 'sf_' + c] = np.nan
    for ax, cols in AXES.items():
        a['ax_' + ax] = a[cols].rank(pct=True).mean(axis=1)
    return a


def gates(g):
    ok = ((g['_op_g'] > 0) | (g['_opm_chg'] > 0)) & (g['rights_days'] > 90)
    ran = (g['_r1rel'].rank(pct=True) > 0.8) | (g['_val_surge'].rank(pct=True) > 0.8)
    return (ok & ~ran).fillna(False)


def stock_scores(g, w):
    return sum(v * E2.zr(g['_' + k]) for k, v in w.items()) if w else pd.Series(0.0, index=g.index)


def select(g, score, ok, sectors, struct, rep=None, nr=False, K=None):
    out = []
    if K == 99:
        c = g[g['sector'].isin(sectors) & ok]
        sc = score[c.index].sort_values(ascending=False); cnt = {}
        for i in sc.index:
            s = c.at[i, 'sector']
            if cnt.get(s, 0) >= 2:
                continue
            out.append(i); cnt[s] = cnt.get(s, 0) + 1
            if len(out) >= 6:
                break
        return out
    for s in sectors:
        c = g[(g['sector'] == s) & ok]
        if c.empty:
            continue
        sc = score[c.index].sort_values(ascending=False)
        if struct == '상위2':
            out += list(sc.index[:2]); continue
        if struct == '섹터당1':
            out += list(sc.index[:1]); continue
        col, asc = REPS[rep]
        r = c[col].dropna()
        if r.empty:
            continue
        top3 = r.sort_values(ascending=asc).index[:3]
        lead = sc[sc.index.isin(top3)].index[0]
        out.append(lead)
        rest = c.drop(index=lead)
        if nr:
            rest = rest[(rest['rs3'] >= -15) & (rest['rs3'] <= 0)]
        if not rest.empty:
            out.append(sc[sc.index.isin(rest.index)].index[0])
    return out


def perf(top):
    if len(top) == 0:
        return {'ex63': 0.0, 'ex126': 0.0, 'cw63': 0.0, 'cw126': 0.0, 'n': 0}
    r = {'n': len(top)}
    for h in (63, 126):
        rr = (top[f'ret{h}'] - S.COST).clip(-0.6, 1.5)
        r[f'ex{h}'] = (rr - top[f'mkt{h}']).mean() * 100
        r[f'cw{h}'] = (rr - top[f'cw{h}']).mean() * 100
    return r


def variants():
    V = []
    for ax in AXIS_SETS:
        V.append((f'{ax}|전체|상위6(섹터당2)', ax, 99, '상위6', None, False))
        for K in KS[:-1]:
            for rep in REPS:
                for nr in (True, False):
                    V.append((f"{ax}|K{K}|대표({rep})+차기({'덜오름' if nr else '조건없음'})", ax, K, '대표+차기', rep, nr))
            V.append((f'{ax}|K{K}|상위2', ax, K, '상위2', None, False))
            V.append((f'{ax}|K{K}|섹터당1', ax, K, '섹터당1', None, False))
    return V


def run(d, keys):
    scope = E2.scope_df(d, '주도·과열')
    IC = E2.monthly_ic(scope, keys)
    W = {Y: E2.choose_weights(IC, Y) for Y in range(Y0, Y1 + 2)}
    V = variants()
    rows = {v[0]: [] for v in V}; rows['기준 v2'] = []
    picks_log = {}
    t0 = time.time()
    for t, g in d.groupby('date'):
        Y = t.year
        if Y < Y0 or not W.get(Y):
            continue
        g = g.set_index('code')
        st = sector_table(g)
        score = stock_scores(g, W[Y]); ok = gates(g)
        tops = {}
        for ax, axes in AXIS_SETS.items():
            if st.empty:
                tops[ax] = {K: [] for K in KS}; continue
            tot = st[['ax_' + a for a in axes]].mean(axis=1).sort_values(ascending=False)
            tops[ax] = {K: list(tot.index[:K]) for K in KS}
        for name, ax, K, struct, rep, nr in V:
            if rep and g[REPS[rep][0]].notna().sum() < 50:
                continue                      # 그 달 자료가 없는 대표주 정의는 건너뜀(0으로 채우지 않음)
            sel = select(g, score, ok, tops[ax][K], struct, rep, nr, K)
            rows[name].append({'date': t, **perf(g.loc[sel])})
            picks_log[(name, t)] = sel
        base = E2.pick_month(g[g['stage'].isin(STAGES)], W[Y], '영업이익증가', 2) if (g['stage'].isin(STAGES)).sum() >= 20 else g.iloc[0:0]
        rows['기준 v2'].append({'date': t, **perf(base)})
    series = {k: pd.DataFrame(v).set_index('date') for k, v in rows.items() if v}
    print(f'변형 {len(series)}개, {time.time() - t0:.0f}초', flush=True)
    return series, W, picks_log, IC


def yearly(s, col='ex126'):
    return s[col].groupby(s.index.year).mean()


def walkforward(series):
    names = [k for k in series if k != '기준 v2']
    Yall = pd.DataFrame({k: yearly(series[k]) for k in names})
    out, picks, ens = [], {}, {}
    for Y in range(WF0, Y1 + 1):
        cut = pd.Timestamp(f'{Y}-01-01')
        past = pd.DataFrame({k: yearly(series[k][series[k].index + LAG < cut]) for k in names})
        past = past.loc[:, past.count() >= 3]
        if past.empty:
            continue
        t = past.mean() / (past.std(ddof=1) / np.sqrt(past.count()))
        pos = (past > 0).mean()
        cand = t[pos >= 0.7].sort_values(ascending=False)
        if cand.empty:
            cand = t.sort_values(ascending=False)
        best = cand.index[0]; top5 = list(cand.index[:5])
        picks[Y] = best; ens[Y] = top5
        row = {'year': Y, '고른 변형': best, '성과': Yall.at[Y, best] if Y in Yall.index else np.nan,
               '상위5 평균': Yall.loc[Y, top5].mean() if Y in Yall.index else np.nan, '전체 평균': Yall.loc[Y].mean() if Y in Yall.index else np.nan,
               '과거 t값': round(float(t[best]), 2)}
        for col in ('ex63', 'cw126'):
            s = series[best]; row[col] = s[s.index.year == Y][col].mean()
        out.append(row)
    return pd.DataFrame(out).set_index('year'), picks, ens


def stat_table(series, lo=2016, hi=2023):
    rows = []
    for k, s in series.items():
        y = yearly(s)
        y16 = y[(y.index >= lo) & (y.index <= hi)]; y24 = y[y.index >= 2024]
        rows.append({'변형': k, f'{lo % 100}~{hi % 100}': y16.mean(), '플러스 해': f'{(y16 > 0).sum()}/{len(y16)}', '최악': y16.min(),
                     '시총가중 대비': yearly(s, 'cw126')[(y.index >= lo) & (y.index <= hi)].mean(), '3개월': yearly(s, 'ex63')[(y.index >= lo) & (y.index <= hi)].mean(),
                     '24~': y24.mean(), '종목 수': s['n'].mean()})
    return pd.DataFrame(rows).set_index('변형')


def marginals(T):
    V = {v[0]: v for v in variants()}
    rows = []
    for dim, fn in (('축', lambda v: v[1]), ('섹터 수', lambda v: '전체' if v[2] == 99 else f'K{v[2]}'), ('구조', lambda v: v[3]), ('대표주 정의', lambda v: v[4] or '-'),
                    ('차기주자 조건', lambda v: ('덜오름' if v[5] else '조건없음') if v[3] == '대표+차기' else '-')):
        grp = {}
        for name, v in V.items():
            if name in T.index:
                grp.setdefault(fn(v), []).append(name)
        for val, names in grp.items():
            x = T.loc[names]
            pl = x['플러스 해'].str.split('/').str[0].astype(float)
            rows.append({'항목': dim, '값': val, '변형 수': len(names), '16~23 평균': x['16~23'].mean(), '플러스 해 평균': pl.mean(), '최악 평균': x['최악'].mean(), '24~ 평균': x['24~'].mean()})
    return pd.DataFrame(rows)


def current(raw, keys, W, picks, ens):
    t = raw['date'].max()
    g = raw[raw['date'] == t].set_index('code')
    st = sector_table(g)
    Yc = max(picks) if picks else Y1
    w = W.get(Y1 + 1) or W.get(Yc) or {}
    score = stock_scores(g, w); ok = gates(g)
    V = {v[0]: v for v in variants()}
    out = {'asof': f'{t:%Y-%m-%d}', 'variant': picks.get(Yc), 'weights': w, 'picks': {}, 'ensemble': {}}
    names = [picks.get(Yc)] + [n for n in ens.get(Yc, []) if n != picks.get(Yc)]
    cnt = {}
    for name in names:
        if not name or name not in V:
            continue
        _, ax, K, struct, rep, nr = V[name]
        if st.empty:
            continue
        tot = st[['ax_' + a for a in AXIS_SETS[ax]]].mean(axis=1).sort_values(ascending=False)
        sel = select(g, score, ok, list(tot.index[:K]), struct, rep, nr, K)
        out['picks'][name] = [{'code': c, 'sector': g.at[c, 'sector'], 'score': round(float(score[c]), 3)} for c in sel]
        for c in sel:
            cnt[c] = cnt.get(c, 0) + 1
    out['ensemble'] = dict(sorted(cnt.items(), key=lambda x: -x[1]))
    if not st.empty:
        out['sectors'] = st[[c for c in st.columns if c.startswith('ax_')] + ['n']].round(3).sort_values('ax_가격', ascending=False).to_dict('index')
    return out, st, g, score


def main():
    d, keys, raw = load()
    series, W, picks_log, IC = run(d, keys)
    wf, picks, ens = walkforward(series)
    T = stat_table(series).sort_values('16~23', ascending=False)
    M = marginals(T)
    base = T.loc['기준 v2'] if '기준 v2' in T.index else None
    pre = wf[wf.index <= 2023]
    L = ['# 엔진 v3 — 주도 섹터 안 기관 선호주(전 과정 완전 걷기식)', '',
         f'변형 {len(series) - 1}개 + 기준 v2. 요소 부호·채택은 해마다 과거 정보계수로, 변형은 해마다 이전 해들의 성과(6개월 결과 확인 뒤)로 고름. '
         f'성과: 6개월 보유, 비용 0.5%, 후보군 동일가중 대비 %p(시총가중 대비는 따로 표시).', '',
         '## 최종(변형까지 걷기식)', '',
         f"{WF0}~2023: 6개월 {pre['성과'].mean():+.2f}%p, 플러스 해 {(pre['성과'] > 0).sum()}/{len(pre)}, 3개월 {pre['ex63'].mean():+.2f}%p, 시총가중 대비 {pre['cw126'].mean():+.2f}%p "
         f"| 상위5 앙상블 {pre['상위5 평균'].mean():+.2f}%p({(pre['상위5 평균'] > 0).sum()}/{len(pre)}) | 전체 변형 평균 {pre['전체 평균'].mean():+.2f}%p({(pre['전체 평균'] > 0).sum()}/{len(pre)})",
         f"2024~: 6개월 {wf[wf.index >= 2024]['성과'].mean():+.2f}%p, 상위5 앙상블 {wf[wf.index >= 2024]['상위5 평균'].mean():+.2f}%p, 전체 평균 {wf[wf.index >= 2024]['전체 평균'].mean():+.2f}%p", '']
    if base is not None:
        L += [f"기준 v2(주도·과열|영업이익증가|섹터당2, 같은 요소 가중치): 2016~23 {base['16~23']:+.2f}%p, 플러스 해 {base['플러스 해']}, 최악 {base['최악']:+.2f}, 시총가중 대비 {base['시총가중 대비']:+.2f}, 2024~ {base['24~']:+.2f}", '']
    L += [wf.round(2).to_markdown(), '', '## 어느 축이 효과가 있나(변형별 2016~23 성과를 항목별로 평균)', '', M.round(2).to_markdown(index=False), '',
          '## 변형별(1단 걷기식, 2016~23 6개월 동일가중 대비) 상위 30', '', T.head(30).round(2).to_markdown(), '',
          '## 변형별 하위 10', '', T.tail(10).round(2).to_markdown(), '',
          '## 해마다 고른 요소(주도·과열 범위)', ''] + [f"- {y}: " + ', '.join(('' if s > 0 else '감점 ') + F[k][1] for k, s in w.items()) for y, w in W.items() if w]
    cur, st, g, score = current(raw, keys, W, picks, ens)
    nm = {}
    try:
        from . import data as Dm
        nm = Dm.prices()['name'].to_dict()
    except Exception:
        pass
    L += ['', f"## 현재 판정({cur['asof']}) · {cur['variant']}", '']
    if not st.empty:
        show = st[['n'] + [c for c in st.columns if c.startswith('ax_')]].copy()
        show['네축'] = show[[c for c in show.columns if c.startswith('ax_')]].mean(axis=1)
        L += ['### 섹터 축 점수(주도·과열 단계만)', '', show.sort_values('네축', ascending=False).round(2).to_markdown(), '']
    for name, ps in cur['picks'].items():
        L.append(f"- {name}: " + ', '.join(f"{nm.get(p['code'], p['code'])}({p['sector']})" for p in ps))
    L += ['', '### 상위 변형들이 겹쳐 고른 종목(횟수)', '', ', '.join(f"{nm.get(c, c)} {k}" for c, k in cur['ensemble'].items()), '']
    open(os.path.join(ROOT, 'docs', 'sim', 'engine3.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    cur['names'] = {c: nm.get(c, c) for ps in cur['picks'].values() for p in ps for c in [p['code']]}
    json.dump(cur, open(os.path.join(ROOT, 'docs', 'sim', 'engine3_current.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    T.round(3).to_csv(os.path.join(ROOT, 'docs', 'sim', 'engine3_variants.csv'), encoding='utf-8-sig')
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
