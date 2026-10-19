"""매 영업일 기준 무한 경쟁 탐색 — 1목적: 주도 섹터 안에서 외국인·기관이 몰리기 직전 종목.
각 방법(섹터 관문·섹터 순위 축·섹터 수·종목 요소 조합·구조)을 매 영업일 그날까지 자료로 적용해 종목을 뽑고,
이후 21·63거래일 실제 결과로 채점: ① 시장(후보군 동일가중) 대비 ② 섹터 대비 ③ 뽑은 뒤 외국인+기관 순매수 유입(같은 섹터 평균 대비) ④ 섹터 자체가 시장보다 올랐나 ⑤ 안정성(5일 전 선정과 겹침).
요소 부호는 해마다 그 이전 해들의 정보계수로(결과 확인 시점 차단), 방법 선택도 해마다 이전 해들 성과로(완전 걷기식).
겹치는 보유기간 때문에 판정은 연도별 평균과 연도 간 t값으로 한다.
python3 -m sim.daily_search <마감 시각 UTC, 예: 2026-10-19T20:15> → docs/sim/daily_search.md, docs/sim/daily_methods.csv, docs/sim/daily_current.json
"""
import glob, json, os, pickle, sys, time
import numpy as np
import pandas as pd
from . import engine as E
from . import engine3 as E3

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
D0 = pd.Timestamp('2014-01-01')
COST = 0.005
H = (21, 63)
STEP = 1                       # 판단 간격(영업일). 1 = 매 영업일
FUND = ['op_g', 'opm_chg', 'roe', 'ep', 'bp', 'op_yield', 'ry', 'accel_d', 'buyback90', 'major_foreign90', 'supply90']
rng = np.random.default_rng(int(time.time()))


def f32(x):
    return x.astype('float32')


def build():
    t0 = time.time()
    C = E.Ctx('theme')
    dates = C.dates[C.dates >= D0 - pd.Timedelta(days=400)]
    P = C.P.loc[dates]
    cap = C.W['cap'].loc[dates]; val = C.W['val'].loc[dates]; ret = C.W['ret'].loc[dates]
    sec = C.sec.reindex(P.columns)
    keep = P.columns[sec.notna() & (sec != '기타') & (cap.max() > 0)]
    P, cap, val, ret, sec = P[keep], cap[keep], val[keep], ret[keep], sec[keep]
    val20 = val.rolling(20, min_periods=15).mean()
    uni = (cap.rank(axis=1, ascending=False) <= 1200) & (val20.rank(axis=1, ascending=False) <= 1500)
    X = {}
    X['low_vol'] = -ret.rolling(60, min_periods=40).std()
    X['hi52'] = P / P.rolling(250, min_periods=120).max() - 1
    r1 = P / P.shift(21) - 1; r3 = P / P.shift(63) - 1
    X['mom12_1'] = P.shift(21) / P.shift(250) - 1
    X['val_surge'] = val20 / val.rolling(120, min_periods=60).mean()
    # 섹터 지수(시총 가중)와 섹터 기준 상대 수익
    SI = C.secidx.loc[dates]
    secs = sorted(set(sec.dropna()) & set(SI.columns))
    s1 = SI / SI.shift(21) - 1; s3 = SI / SI.shift(63) - 1
    def to_stock(M):
        out = pd.DataFrame(np.nan, index=dates, columns=keep, dtype='float32')
        for s in secs:
            cols = sec.index[sec == s]
            out[cols] = np.repeat(M[s].values[:, None], len(cols), axis=1)
        return out
    X['r1rel'] = r1 - to_stock(s1); X['rs3'] = r3 - to_stock(s3)
    X['r1'] = r1
    # 섹터 안 대표성
    rc, rv = pd.DataFrame(index=dates, columns=keep, dtype='float32'), pd.DataFrame(index=dates, columns=keep, dtype='float32')
    for s in secs:
        cols = sec.index[sec == s]
        rc[cols] = cap[cols].where(uni[cols]).rank(axis=1, ascending=False)
        rv[cols] = val20[cols].where(uni[cols]).rank(axis=1, ascending=False)
    X['rep_cap'] = -rc; X['rep_val'] = -rv
    # 수급(한투 과거 + 일별): 금액(백만원) → 원
    fr, og, pn = [pd.DataFrame(np.nan, index=dates, columns=keep, dtype='float32') for _ in range(3)]
    daily = glob.glob(os.path.join(ROOT, 'data', 'kis', 'flows', '*.csv'))
    DF = pd.concat([pd.read_csv(f, dtype={'date': str, 'code': str}) for f in daily]) if daily else pd.DataFrame()
    if len(DF):
        DF['date'] = pd.to_datetime(DF['date'])
    have = []
    for f in glob.glob(os.path.join(ROOT, 'archive', 'kis_flows', '*.csv.gz')):
        c = os.path.basename(f)[:6]
        if c not in keep:
            continue
        d = pd.read_csv(f, dtype={'date': str})
        if d.empty:
            continue
        d['date'] = pd.to_datetime(d['date'])
        if len(DF):
            x = DF[(DF['code'] == c) & (DF['date'] > d['date'].max())]
            d = pd.concat([d, x[d.columns.intersection(x.columns)]])
        d = d.drop_duplicates('date', keep='last').set_index('date').reindex(dates)
        fr[c] = d['frgn'].values * 1e6; og[c] = d['orgn'].values * 1e6; pn[c] = d['pension'].values * 1e6
        have.append(c)
    hasf = pd.Series(False, index=keep); hasf[have] = True
    fo = (fr.fillna(0) + og.fillna(0)).where(fr.notna() | og.notna())
    fo20 = fo.rolling(20, min_periods=15).sum(); fo60 = fo.rolling(60, min_periods=45).sum()
    X['fo60_cap'] = fo60 / cap; X['fo20_cap'] = fo20 / cap
    X['fo_accel'] = (fo20 - (fo60 - fo20) / 2) / cap
    X['frgn60_cap'] = fr.rolling(60, min_periods=45).sum() / cap
    X['orgn20_cap'] = og.rolling(20, min_periods=15).sum() / cap
    X['pension60_cap'] = pn.rolling(60, min_periods=45).sum() / cap
    # 실현 결과: 뽑은 뒤 수급 유입(t+1~t+h 외국인+기관 순매수/시총)
    Y = {}
    for h in H:
        fwd = P.shift(-(h + 1)) / P.shift(-1) - 1
        Y[f'ret{h}'] = fwd
        fl = fo.shift(-1).iloc[::-1].rolling(h, min_periods=int(h * 0.7)).sum().iloc[::-1]
        Y[f'flow{h}'] = fl / cap
        Y[f'secret{h}'] = to_stock(SI.shift(-(h + 1)) / SI.shift(-1) - 1)
    # 실적·공시(월말 패널 값을 다음 월말까지 사용)
    pan = pickle.load(open(os.path.join(WORK, 'panel_theme_x.pkl'), 'rb'))
    pan = E3.factorize(pan, [k for k in FUND if k in E3.F])
    for k in FUND:
        if '_' + k in pan:
            m = pan.pivot_table(index='date', columns='code', values='_' + k).reindex(columns=keep)
            X[k] = m.reindex(dates, method='ffill')
    # 섹터 단위 지표(날짜×섹터)
    L = SI[secs]
    k = C.kospi.reindex(dates).ffill()
    ma20, ma60 = L.rolling(20, min_periods=20).mean(), L.rolling(60, min_periods=60).mean()
    rs = L.div(k, axis=0); rs60 = rs.rolling(60, min_periods=60).mean()
    q = lambda b: b.astype(float).where(ma60.notna()).rolling(60, min_periods=40).mean()
    S = {}
    S['tp_above60'] = q(L > ma60); S['tp_align'] = q(ma20 > ma60); S['tp_above20'] = q(L > ma20); S['tp_rs60'] = q(rs > rs60)
    S['tp_slope60'] = ma60 / ma60.shift(20) - 1; S['tp_rsslope'] = rs60 / rs60.shift(20) - 1
    wk = L.resample('W-FRI').last(); m40 = wk.rolling(40, min_periods=40).mean()
    S['sp_slope40w'] = (m40 / m40.shift(4) - 1).reindex(dates, method='ffill')
    S['sp_ret120'] = L / L.shift(120) - 1
    S['sp_gap200'] = L / L.rolling(200, min_periods=200).mean() - 1
    below = (L < ma60).astype(int)
    S['br_below'] = (below.apply(lambda c: c.groupby((c == 0).cumsum()).cumsum()) >= 10).astype(float)
    S['br_dead'] = ((ma20 < ma60) & (ma60 < ma60.shift(10))).astype(float)
    up = (L > ma20).astype(int)
    runup = up.apply(lambda c: c.groupby((c == 0).cumsum()).cumsum())
    cross = ((L > ma20) & (L.shift(1) <= ma20.shift(1))).astype(int).rolling(20, min_periods=1).max()
    S['rc_start'] = ((runup >= 10) & (cross > 0) & (ma20 > ma20.shift(5))).astype(float)
    # 그날 단계(engine.sector_stage와 같은 규칙, 코스피 기준)
    kr = lambda n: (k / k.shift(n) - 1) * 100
    lr = lambda n: (L / L.shift(n) - 1) * 100
    rs1, rs3_, rs6 = lr(21).sub(kr(21), axis=0), lr(63).sub(kr(63), axis=0), lr(126).sub(kr(126), axis=0)
    g60 = (L / L.rolling(60).mean() - 1) * 100; g120 = (L / L.rolling(120).mean() - 1) * 100
    S['stage_strong'] = (~((g60 < 0) & (rs3_ < 0)) & ((g60 > 20) | (lr(21) > 25) | ((rs3_ > 0) & (rs6 > 0) & (g120 > 0)))).astype(float)
    # 섹터 수급·이익 확산
    capS = pd.DataFrame({s: cap[sec.index[sec == s]].where(uni[sec.index[sec == s]]).sum(axis=1) for s in secs})
    f20S = pd.DataFrame({s: fo20[sec.index[(sec == s) & hasf]].where(uni[sec.index[(sec == s) & hasf]]).sum(axis=1, min_count=1) for s in secs})
    f60S = pd.DataFrame({s: fo60[sec.index[(sec == s) & hasf]].where(uni[sec.index[(sec == s) & hasf]]).sum(axis=1, min_count=1) for s in secs})
    S['sf_fo60'] = f60S / capS; S['sf_fo20'] = f20S / capS; S['sf_accel'] = (f20S - (f60S - f20S) / 2) / capS
    S['br_flow'] = ((f20S < 0) & ((f60S <= 0) | (-f20S >= 0.5 * f60S))).astype(float)
    if 'op_g' in X:
        S['se_opg'] = pd.DataFrame({s: (X['op_g'][sec.index[sec == s]].where(uni[sec.index[sec == s]]) > 0).mean(axis=1) for s in secs})
    S['sb_nearhi'] = pd.DataFrame({s: (X['hi52'][sec.index[sec == s]].where(uni[sec.index[sec == s]]) > -0.1).mean(axis=1) for s in secs})
    nS = pd.DataFrame({s: uni[sec.index[sec == s]].sum(axis=1) for s in secs})
    # 판단일: D0 이후, 63일 뒤 결과가 있는 날까지(결과 없는 최근일은 현재 판정에만)
    idx = dates[dates >= D0]
    X = {kk: f32(v.loc[idx]) for kk, v in X.items()}
    Y = {kk: f32(v.loc[idx]) for kk, v in Y.items()}
    S = {kk: v.loc[idx] for kk, v in S.items()}
    meta = {'idx': idx, 'uni': uni.loc[idx], 'sec': sec, 'secs': secs, 'nS': nS.loc[idx], 'hasf': hasf, 'cap': f32(cap.loc[idx]),
            'names': C.W['name'].reindex(keep), 'flow_cov': float(hasf.mean())}
    print(f'준비 {time.time() - t0:.0f}초, 종목 {len(keep)}, 날짜 {len(idx)}, 수급 보유 {hasf.mean():.0%}', flush=True)
    return X, Y, S, meta


# ---------- 요소 부호(해마다 걷기식) ----------
def yearly_ic(X, Y, meta):
    """요소별 연도 정보계수(5영업일마다 표본, 63일 시장 대비)."""
    uni = meta['uni']; idx = meta['idx'][::5]
    y = Y['ret63'].loc[idx].where(uni.loc[idx])
    yr = y.rank(axis=1)
    out = {}
    for k, x in X.items():
        xr = x.loc[idx].where(uni.loc[idx] & y.notna()).rank(axis=1)
        a = xr.sub(xr.mean(axis=1), axis=0); b = yr.sub(yr.mean(axis=1), axis=0)
        ic = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        ic[xr.notna().sum(axis=1) < 50] = np.nan
        out[k] = ic
    IC = pd.DataFrame(out)
    known = pd.Series(IC.index, index=IC.index) + pd.Timedelta(days=95)
    signs = {}
    for Yr in range(D0.year, meta['idx'].max().year + 2):
        past = IC[known < pd.Timestamp(f'{Yr}-01-01')]
        a = past.groupby(past.index.year).mean()
        s = {}
        for k in IC.columns:
            v = a[k].dropna()
            if len(v) < 2:
                continue
            t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)) + 1e-9)
            s[k] = 1 if (t >= 1.5 and (v > 0).mean() >= 0.6) else (-1 if (t <= -1.5 and (v < 0).mean() >= 0.6) else 0)
        signs[Yr] = s
    return IC, signs


# ---------- 방법 ----------
SEC_GATES = ['전체', '단계', '추세', '추세-꺾임', '추세+초입']
SEC_AXES = {'가격': ['sp_slope40w', 'sp_ret120', 'sp_gap200'], '추세흐름': ['tp_above60', 'tp_align', 'tp_above20', 'tp_rs60', 'tp_slope60', 'tp_rsslope'],
            '수급': ['sf_fo60'], '수급기세': ['sf_accel', 'sf_fo20'], '이익': ['se_opg'], '확산': ['sb_nearhi']}
POOL = ['low_vol', 'hi52', 'mom12_1', 'val_surge', 'r1rel', 'rs3', 'r1', 'rep_cap', 'rep_val', 'fo60_cap', 'fo20_cap', 'fo_accel', 'frgn60_cap',
        'orgn20_cap', 'pension60_cap'] + FUND


def random_method():
    m = {'gate': rng.choice(SEC_GATES), 'axes': sorted(rng.choice(list(SEC_AXES), size=rng.integers(1, 4), replace=False).tolist()),
         'K': int(rng.choice([2, 3, 5, 8, 99])), 'facs': sorted(rng.choice(POOL, size=rng.integers(2, 9), replace=False).tolist()),
         'N': int(rng.choice([5, 8, 12])), 'cap': int(rng.choice([1, 2, 3])), 'profit': bool(rng.random() < 0.6), 'noran': bool(rng.random() < 0.6),
         'sign': rng.choice(['걷기식', '걷기식+고정가점'])}
    return m


def mname(m):
    return f"{m['gate']}|{'+'.join(m['axes'])}|K{'전체' if m['K'] == 99 else m['K']}|{','.join(m['facs'])}|N{m['N']}c{m['cap']}|{'이익' if m['profit'] else '-'}|{'안달림' if m['noran'] else '-'}|{m['sign']}"


def sector_mask(m, S, meta):
    secs = meta['secs']
    ok = meta['nS'][secs] >= 5
    if m['gate'] == '단계':
        ok &= S['stage_strong'][secs] > 0
    elif m['gate'] in ('추세', '추세-꺾임', '추세+초입'):
        tr = (S['tp_above60'][secs] >= 0.6) & (S['tp_align'][secs] >= 0.5)
        if m['gate'] == '추세-꺾임':
            tr &= (S['br_below'][secs].fillna(0) + S['br_dead'][secs].fillna(0) + S['br_flow'][secs].fillna(0)) < 2
        if m['gate'] == '추세+초입':
            tr |= S['rc_start'][secs] > 0
        ok &= tr
    cols = [c for a in m['axes'] for c in SEC_AXES[a] if c in S]
    sc = sum(S[c][secs].where(ok).rank(axis=1, pct=True).fillna(0.5).where(ok) for c in cols) / max(len(cols), 1)
    if m['K'] != 99:
        ok &= sc.rank(axis=1, ascending=False, method='first') <= m['K']
    return ok


def evaluate(m, X, Y, S, meta, signs):
    idx, uni, sec = meta['idx'], meta['uni'], meta['sec']
    okS = sector_mask(m, S, meta)
    # 섹터 허용 → 종목 마스크
    ok = pd.DataFrame(False, index=idx, columns=uni.columns)
    for s in meta['secs']:
        cols = sec.index[sec == s]
        ok[cols] = np.repeat(okS[s].fillna(False).values[:, None], len(cols), axis=1)
    ok &= uni
    if m['profit'] and 'op_g' in X:
        ok &= (X['op_g'] > 0) | (X['opm_chg'] > 0)
    if m['noran']:
        ok &= ~((X['r1rel'].rank(axis=1, pct=True) > 0.8) | (X['val_surge'].rank(axis=1, pct=True) > 0.8))
    # 점수: 요소별 순위(0~1) × 해마다 부호
    years = idx.year
    score = pd.DataFrame(0.0, index=idx, columns=uni.columns, dtype='float32')
    used = 0
    for f in m['facs']:
        if f not in X:
            continue
        sg = np.array([signs.get(y, {}).get(f, 0) for y in years], dtype='float32')
        if m['sign'] == '걷기식+고정가점':
            sg = np.where(sg == 0, 1, sg).astype('float32')   # 부호가 아직 안 정해진 요소도 가점으로(가설 그대로)
        if not np.any(sg):
            continue
        r = X[f].where(ok).rank(axis=1, pct=True) - 0.5
        score += r.fillna(0).mul(sg, axis=0); used += 1
    if used == 0:
        return None
    score = score.where(ok)
    # 섹터당 cap, 전체 N
    srank = pd.DataFrame(np.nan, index=idx, columns=uni.columns, dtype='float32')
    for s in meta['secs']:
        cols = sec.index[sec == s]
        srank[cols] = score[cols].rank(axis=1, ascending=False, method='first')
    sc2 = score.where(srank <= m['cap'])
    pick = (sc2.rank(axis=1, ascending=False, method='first') <= m['N']) & sc2.notna()
    n = pick.sum(axis=1)
    res = {'n': n}
    ew_ok = uni
    for h in H:
        r = Y[f'ret{h}']
        mk = r.where(ew_ok).mean(axis=1)
        pr = (r.where(pick) - COST).clip(-0.6, 1.5).mean(axis=1)
        res[f'ex{h}'] = (pr - mk) * 100
        res[f'sx{h}'] = ((r - Y[f'secret{h}']).where(pick)).mean(axis=1) * 100
        res[f'secx{h}'] = (Y[f'secret{h}'].where(pick).mean(axis=1) - mk) * 100
        fl = Y[f'flow{h}']
        # 수급 유입: 뽑은 종목의 이후 외국인+기관 순매수/시총 - 같은 날 후보군 평균(수급 자료 있는 종목만)
        hf = meta['hasf'].reindex(uni.columns).values
        res[f'fx{h}'] = (fl.where(pick & hf).mean(axis=1) - fl.where(uni & hf).mean(axis=1)) * 100
    prev = pick.shift(5).fillna(False).astype(bool)
    res['stab'] = (pick & prev).sum(axis=1) / n.replace(0, np.nan)
    R = pd.DataFrame(res)
    R.loc[R['n'] == 0, [c for c in R.columns if c != 'n']] = np.nan
    return R, pick


def yearly_stats(R):
    y = R.groupby(R.index.year)
    return y.mean()


def summarize(name, R):
    Yr = yearly_stats(R)
    pre = Yr.loc[2016:2023]; post = Yr.loc[2024:]
    t = lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(v.count())) if v.count() > 2 else np.nan
    return {'방법': name, '63일 16~23': pre['ex63'].mean(), '플러스 해': int((pre['ex63'] > 0).sum()), '해 수': int(pre['ex63'].count()), 't': t(pre['ex63']),
            '최악': pre['ex63'].min(), '21일 16~23': pre['ex21'].mean(), '섹터대비 63': pre['sx63'].mean(), '섹터 선택 63': pre['secx63'].mean(),
            '수급유입 63': pre['fx63'].mean(), '24~ 63일': post['ex63'].mean(), '안정성': R['stab'].mean(), '평균 종목 수': R['n'].mean()}


def walkforward(store, start=2017):
    """해마다 이전 해들의 연도 성과(63일 결과 확인 뒤)로 방법 고르기: 1등, 상위5 평균, 상위20 평균."""
    names = list(store)
    Yall = pd.DataFrame({k: yearly_stats(v)['ex63'] for k, v in store.items()})
    rows, picks = [], {}
    for Yr in range(start, int(Yall.index.max()) + 1):
        cut = pd.Timestamp(f'{Yr}-01-01') - pd.Timedelta(days=95)
        past = pd.DataFrame({k: store[k].loc[:cut, 'ex63'].groupby(store[k].loc[:cut].index.year).mean() for k in names})
        past = past.loc[:, past.count() >= 3]
        if past.empty:
            continue
        t = past.mean() / (past.std(ddof=1) / np.sqrt(past.count()))
        pos = (past > 0).mean()
        cand = t[pos >= 0.7].sort_values(ascending=False)
        if len(cand) < 20:
            cand = t.sort_values(ascending=False)
        top = list(cand.index[:20]); picks[Yr] = top
        rows.append({'year': Yr, '1등': Yall.at[Yr, top[0]], '상위5': Yall.loc[Yr, top[:5]].mean(), '상위20': Yall.loc[Yr, top].mean(),
                     '전체 평균': Yall.loc[Yr].mean(), '1등 방법': top[0]})
    return pd.DataFrame(rows).set_index('year'), picks


def marginals(T):
    rows = []
    parts = T.index.str.split('|')
    dims = {'섹터 관문': 0, '섹터 축': 1, '섹터 수': 2, '구조': 4, '이익 관문': 5, '안 달린 종목': 6, '부호': 7}
    for dim, i in dims.items():
        v = parts.str[i]
        for val in sorted(set(v)):
            x = T[v == val]
            rows.append({'항목': dim, '값': val, '방법 수': len(x), '63일 16~23': x['63일 16~23'].mean(), '플러스 해': x['플러스 해'].mean(),
                         '수급유입 63': x['수급유입 63'].mean(), '24~': x['24~ 63일'].mean()})
    fac = {}
    for nm, r in T.iterrows():
        for f in nm.split('|')[3].split(','):
            fac.setdefault(f, []).append(r['63일 16~23'])
    base = T['63일 16~23'].mean()
    for f, v in fac.items():
        rows.append({'항목': '종목 요소 포함 시', '값': f, '방법 수': len(v), '63일 16~23': np.mean(v) - base, '플러스 해': np.nan, '수급유입 63': np.nan, '24~': np.nan})
    return pd.DataFrame(rows)


def write(store, picks_now, meta, IC, signs, wf, wfp, elapsed):
    T = pd.DataFrame([summarize(k, v) for k, v in store.items()]).set_index('방법').sort_values('t', ascending=False)
    T.round(3).to_csv(os.path.join(ROOT, 'docs', 'sim', 'daily_methods.csv'), encoding='utf-8-sig')
    M = marginals(T)
    pre = wf[wf.index <= 2023]
    nm = meta['names']
    L = ['# 매 영업일 기준 무한 경쟁 탐색(1차)', '',
         f'방법 {len(store)}개, 판단일 {len(meta["idx"])}일(매 영업일), 탐색 {elapsed / 60:.0f}분. 수급 자료 보유 종목 {meta["flow_cov"]:.0%}(수급 결론은 자료 부족 감안).',
         '채점: 63거래일(약 3개월)·21거래일 뒤, 비용 0.5%, 후보군 동일가중 대비 %p. 섹터대비 = 같은 섹터 지수 대비, 섹터 선택 = 고른 섹터 자체의 시장 대비, 수급유입 = 뽑은 뒤 63일 외국인+기관 순매수/시총 - 후보군 평균(%p).',
         '요소 부호와 방법 선택 모두 해마다 그 이전 결과(63일 결과 확인 뒤)만으로 정함.', '',
         '## 완전 걷기식 결과(해마다 이전 성과로 방법 고르기)', '',
         f"2017~2023: 1등 {pre['1등'].mean():+.2f}%p({(pre['1등'] > 0).sum()}/{len(pre)}) · 상위5 {pre['상위5'].mean():+.2f}%p({(pre['상위5'] > 0).sum()}/{len(pre)}) · 상위20 {pre['상위20'].mean():+.2f}%p({(pre['상위20'] > 0).sum()}/{len(pre)}) · 전체 평균 {pre['전체 평균'].mean():+.2f}%p",
         f"2024~: 1등 {wf.loc[2024:, '1등'].mean():+.2f}%p · 상위5 {wf.loc[2024:, '상위5'].mean():+.2f}%p · 상위20 {wf.loc[2024:, '상위20'].mean():+.2f}%p", '',
         wf.round(2).to_markdown(), '', '## 무엇이 효과가 있나(방법들의 항목별 평균, 63일 16~23 %p)', '', M.round(2).to_markdown(index=False), '',
         '## 상위 30(사후 순위, 참고용)', '', T.head(30).round(2).to_markdown(), '',
         '## 요소 부호(올해 기준, 걷기식)', '', ', '.join(f"{k}{'+' if v > 0 else '-'}" for k, v in signs.get(meta['idx'].max().year, {}).items() if v), '']
    if picks_now:
        L += [f"## 현재 판정({meta['idx'].max():%Y-%m-%d}): 올해 걷기식 상위 20 방법이 고른 종목(겹친 횟수)", '',
              ', '.join(f"{nm.get(c, c)}({meta['sec'].get(c)}) {k}" for c, k in picks_now[:30]), '']
    open(os.path.join(ROOT, 'docs', 'sim', 'daily_search.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump({'asof': f"{meta['idx'].max():%Y-%m-%d}", 'methods': len(store), 'top20': wfp.get(max(wfp)) if wfp else [],
               'picks': [{'code': c, 'name': nm.get(c, c), 'sector': meta['sec'].get(c), 'count': int(k)} for c, k in picks_now[:30]]},
              open(os.path.join(ROOT, 'docs', 'sim', 'daily_current.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n'.join(L[:16]), flush=True)


def main(deadline, shard=None):
    global rng
    rng = np.random.default_rng(int(time.time()) + 7919 * int(shard or 0))
    t0 = time.time()
    X, Y, S, meta = build()
    IC, signs = yearly_ic(X, Y, meta)
    store, pk = {}, {}
    seen = set()
    # 기준 방법들(지금까지 결론)부터
    seeds = [{'gate': '단계', 'axes': ['가격'], 'K': 99, 'facs': ['low_vol', 'hi52', 'op_g', 'opm_chg', 'roe', 'ep', 'bp', 'op_yield', 'buyback90', 'major_foreign90', 'rs3', 'r1rel', 'val_surge'][:8], 'N': 6 if False else 5, 'cap': 2, 'profit': True, 'noran': False, 'sign': '걷기식'},
             {'gate': '추세', 'axes': ['추세흐름', '수급기세'], 'K': 3, 'facs': ['low_vol', 'hi52', 'op_g', 'opm_chg', 'fo_accel', 'fo60_cap', 'r1rel'], 'N': 5, 'cap': 2, 'profit': True, 'noran': True, 'sign': '걷기식'},
             {'gate': '추세+초입', 'axes': ['추세흐름', '수급기세', '이익'], 'K': 5, 'facs': ['low_vol', 'hi52', 'op_g', 'fo_accel', 'pension60_cap', 'rep_cap'], 'N': 8, 'cap': 2, 'profit': True, 'noran': True, 'sign': '걷기식+고정가점'}]
    last_write = time.time()
    i = 0
    while time.time() < deadline:
        m = seeds[i] if (i < len(seeds) and shard in (None, '1')) else random_method(); i += 1
        nmk = mname(m)
        if nmk in seen:
            continue
        seen.add(nmk)
        try:
            out = evaluate(m, X, Y, S, meta, signs)
        except Exception as e:
            print('오류', nmk, e, flush=True); continue
        if out is None:
            continue
        R, pick = out
        store[nmk] = R
        pk[nmk] = pick.iloc[-1]
        if len(store) % 50 == 0:
            print(len(store), f'{(time.time() - t0) / 60:.0f}분', flush=True)
        if shard is not None and time.time() - last_write > 900:
            dump(store, pk, meta, IC, signs, shard, t0); last_write = time.time()
    if shard is not None:
        dump(store, pk, meta, IC, signs, shard, t0); return
    wf, wfp = walkforward(store)
    write(store, current(pk, wfp), meta, IC, signs, wf, wfp, time.time() - t0)


def dump(store, pk, meta, IC, signs, shard, t0):
    os.makedirs('/tmp/out', exist_ok=True)
    light = {k: meta[k] for k in ('idx', 'sec', 'names', 'flow_cov')}
    tmp = f'/tmp/out/store_{shard}.pkl.tmp'
    pickle.dump({'store': store, 'pk': {k: v[v].index.tolist() for k, v in pk.items()}, 'meta': light, 'signs': signs, 'sec': time.time() - t0}, open(tmp, 'wb'))
    os.replace(tmp, f'/tmp/out/store_{shard}.pkl')
    print('저장', shard, len(store), flush=True)


def merge(paths):
    store, pk, meta, signs, el = {}, {}, None, None, 0
    for p in paths:
        d = pickle.load(open(p, 'rb'))
        store.update(d['store']); meta = d['meta']; signs = d['signs']; el = max(el, d['sec'])
        for k, v in d['pk'].items():
            pk[k] = pd.Series(True, index=v)
    wf, wfp = walkforward(store)
    write(store, current(pk, wfp), meta, None, signs, wf, wfp, el)


def current(pk, wfp):
    if not wfp:
        return []
    top = wfp[max(wfp)]
    cnt = {}
    for k in top:
        if k in pk:
            for c in pk[k][pk[k]].index:
                cnt[c] = cnt.get(c, 0) + 1
    return sorted(cnt.items(), key=lambda x: -x[1])


if __name__ == '__main__':
    if sys.argv[1] == 'merge':
        merge(sorted(glob.glob(sys.argv[2] + '/**/store_*.pkl', recursive=True)))
    else:
        dl = pd.Timestamp(sys.argv[1], tz='UTC').timestamp()
        main(dl, sys.argv[2] if len(sys.argv) > 2 else None)
