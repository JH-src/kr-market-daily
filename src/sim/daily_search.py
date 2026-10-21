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
from . import data as Dm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
D0 = pd.Timestamp('2014-01-01')
COST = 0.005
H = (21, 63, 126)
STEP = 1                       # 판단 간격(영업일). 1 = 매 영업일
FUND = ['op_g', 'opm_chg', 'roe', 'ep', 'bp', 'op_yield', 'ry', 'accel_d', 'buyback90', 'major_foreign90', 'supply90']
rng = np.random.default_rng(int(time.time()))


def f32(x):
    return x.astype('float32')


def persist(L, ma20):
    """20일선 위 흐름이 이어진 날 수. 하루이틀 이탈은 봐주고, 4거래일 연속 20일선 아래면 흐름이 끊긴 것으로 봄(기간을 3·6개월로 미리 정하지 않음)."""
    below = (L < ma20).astype(int).where(ma20.notna(), 1)
    brk = (below.rolling(4, min_periods=4).sum() >= 4).astype(int)
    return brk.apply(lambda c: (c == 0).astype(int).groupby(c.cumsum()).cumsum()).astype('float32')


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
    ma20, ma60, ma120 = P.rolling(20, min_periods=20).mean(), P.rolling(60, min_periods=60).mean(), P.rolling(120, min_periods=120).mean()
    X['g20'] = P / ma20 - 1; X['g60'] = P / ma60 - 1; X['g120'] = P / ma120 - 1
    X['align'] = ((ma20 > ma60) & (ma60 > ma120)).astype('float32').where(ma120.notna())
    X['brk20'] = (P >= P.rolling(20, min_periods=20).max() * 0.999).astype('float32').where(ma20.notna())
    X['r3'] = r3
    dP = P.diff(); up = dP.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(); dn = (-dP.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    X['rsi'] = 100 - 100 / (1 + up / dn)
    X['logcap'] = np.log(cap)
    X['persist20'] = persist(P, ma20)
    X['val_surge'] = val20 / val.rolling(120, min_periods=60).mean()
    # 기술적 지표(리서치 보고서 반영): 볼린저 밴드, MACD, 1년 변동성, 6개월 모멘텀
    sd20 = P.rolling(20, min_periods=20).std()
    X['pctb'] = (P - (ma20 - 2 * sd20)) / (4 * sd20)                       # %B: 0=하단, 1=상단
    bw = 4 * sd20 / ma20
    X['bb_squeeze'] = (bw <= bw.rolling(120, min_periods=60).quantile(0.2)).astype('float32').where(bw.notna())   # 밴드폭이 6개월 중 하위 20%
    e12, e26 = P.ewm(span=12, adjust=False, min_periods=26).mean(), P.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    X['macd_h'] = (macd - sig) / P
    X['macd_pos'] = (macd > 0).astype('float32').where(macd.notna())
    gc = ((macd > sig) & (macd.shift(1) <= sig.shift(1))).astype('float32')
    X['macd_gc5'] = gc.rolling(5, min_periods=1).max().where(macd.notna())   # 최근 5일 안 골든크로스
    X['low_vol252'] = -ret.rolling(252, min_periods=200).std() * np.sqrt(252)
    X['mom6'] = P.shift(21) / P.shift(126) - 1
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
    X['fo20_amt'] = fo20 / 1e8; X['fo60_amt'] = fo60 / 1e8      # 외국인+기관 순매수 금액(억 원) — 대장주는 '얼마나 많이 샀나'
    X['fo_accel'] = (fo20 - (fo60 - fo20) / 2) / cap
    X['frgn60_cap'] = fr.rolling(60, min_periods=45).sum() / cap
    X['orgn20_cap'] = og.rolling(20, min_periods=15).sum() / cap
    X['pension60_cap'] = pn.rolling(60, min_periods=45).sum() / cap
    pos = (fr > 0).astype(int)
    X['frgn_streak'] = pos.apply(lambda c: c.groupby((c == 0).cumsum()).cumsum()).astype('float32').where(fr.notna())
    X['orgn60_cap'] = og.rolling(60, min_periods=45).sum() / cap
    X['fo_vs_prsn'] = ((fo20 > 0) & (X['orgn20_cap'].notna())).astype('float32').where(fo20.notna())
    X['fo20_val'] = fo20 / (val20 * 20)          # 수급을 거래대금으로 정규화(보고서: 시총 대비와 비교)
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
    for k, col in (('per', 'per'), ('psr', 'psr'), ('debt_ratio', 'debt_ratio')):
        if col in pan:
            X[k] = pan.pivot_table(index='date', columns='code', values=col).reindex(columns=keep).reindex(dates, method='ffill')
    # PEG(PER ÷ 순이익 1년 성장률%), 마법공식(이익수익률 순위 + 자본수익률 근사 순위)
    piv = lambda c: pan.pivot_table(index='date', columns='code', values=c).reindex(columns=keep).sort_index()
    if {'ni_ttm', 'op_ttm', 'eq', 'debt'} <= set(pan.columns):
        ni = piv('ni_ttm')
        nim = ni.resample('ME').last()
        prev = nim.shift(12).reindex(ni.index, method='ffill')          # 1년 전 월말 순이익
        g = ((ni / prev - 1) * 100).where((prev > 0) & (ni > 0))
        perm = piv('per')
        X['peg'] = (perm / g).where((perm > 0) & (g > 0)).reindex(dates, method='ffill')
        opt = piv('op_ttm')
        X['roc'] = (opt / (piv('eq') + piv('debt'))).where(opt > 0).reindex(dates, method='ffill')   # 자본수익률 근사(영업이익 ÷ 총자산)
        capm = piv('cap')
        X['ey_m'] = (opt / capm).reindex(dates, method='ffill')     # 이익수익률 근사(영업이익 ÷ 시총, 순부채 미반영)
        mey = ni.where(ni > -1e30).sum(axis=1, min_count=100) / capm.where(ni.notna()).sum(axis=1, min_count=100)   # 시장 이익수익률(일드갭용)
    else:
        mey = None
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
    S['sec_r1'] = L / L.shift(21) - 1; S['sec_r3'] = L / L.shift(63) - 1
    S['sec_persist20'] = persist(L, ma20)
    S['sec_ret'] = L.pct_change(fill_method=None)
    S['rs_w1'] = (L / L.shift(5) - 1).sub(k / k.shift(5) - 1, axis=0)
    S['rs_m1'] = (L / L.shift(21) - 1).sub(k / k.shift(21) - 1, axis=0)
    S['rs_m3'] = (L / L.shift(63) - 1).sub(k / k.shift(63) - 1, axis=0)
    dL = L.diff(); u_ = dL.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(); d_ = (-dL.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    S['sec_rsi'] = 100 - 100 / (1 + u_ / d_)
    vS = pd.DataFrame({s: val[sec.index[sec == s]].sum(axis=1, min_count=1) for s in secs})
    S['sec_val_chg'] = vS.rolling(20, min_periods=15).mean() / vS.rolling(120, min_periods=60).mean()
    tot = vS.sum(axis=1)
    S['sec_val_share_chg'] = vS.div(tot, axis=0).rolling(20, min_periods=15).mean() / vS.div(tot, axis=0).rolling(120, min_periods=60).mean()
    # 섹터 대장(시총 상위 3) 종목들의 20일선 위 흐름 평균
    S['lead_persist20'] = pd.DataFrame({s: X['persist20'][sec.index[sec == s]].where(rc[sec.index[sec == s]] <= 3).mean(axis=1) for s in secs})
    S['sec_above60_share'] = pd.DataFrame({s: (X['g60'][sec.index[sec == s]].where(uni[sec.index[sec == s]]) > 0).mean(axis=1) for s in secs})
    if 'peg' in X:
        X['peg_inv'] = -X['peg']
        X['magic'] = X['ey_m'].where(uni).rank(axis=1, pct=True) + X['roc'].where(uni).rank(axis=1, pct=True)   # 그린블라트 마법공식(순위 합)
    if 'debt_ratio' in X:
        X['debt_low'] = -X['debt_ratio']
    G = market_gates(C, dates, mey)
    # 판단일: D0 이후, 63일 뒤 결과가 있는 날까지(결과 없는 최근일은 현재 판정에만)
    idx = dates[dates >= D0]
    X = {kk: f32(v.loc[idx]) for kk, v in X.items()}
    Y = {kk: f32(v.loc[idx]) for kk, v in Y.items()}
    S = {kk: v.loc[idx] for kk, v in S.items()}
    meta = {'G': {kk: v.reindex(idx) for kk, v in G.items()}, 'idx': idx, 'uni': uni.loc[idx], 'sec': sec, 'secs': secs, 'nS': nS.loc[idx], 'hasf': hasf, 'cap': f32(cap.loc[idx]),
            'names': C.W['name'].reindex(keep), 'flow_cov': float(hasf.mean())}
    print(f'준비 {time.time() - t0:.0f}초, 종목 {len(keep)}, 날짜 {len(idx)}, 수급 보유 {hasf.mean():.0%}', flush=True)
    return X, Y, S, meta


def _fred(sid, lag_days):
    """data/macro/<id>.csv(Actions에서 FRED로 받아 둔 것). 발표 지연만큼 날짜를 미룸. 없으면 None."""
    f = os.path.join(ROOT, 'data', 'macro', f'{sid}.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d.columns = ['date', 'v']
    d['v'] = pd.to_numeric(d['v'], errors='coerce')
    s = pd.Series(d['v'].values, index=pd.to_datetime(d['date']) + pd.Timedelta(days=lag_days)).dropna().sort_index()
    return s if len(s) else None


def market_gates(C, dates, mey):
    """시장 국면 신호(날짜별 값). 리서치 보고서의 매크로 필터: 추세, 국면 점수, 자체 공포·탐욕, 일드갭, 장단기 금리차, 하이일드 스프레드, 물가."""
    k = C.kospi.reindex(dates).ffill()
    G = {'kospi_ma200': k / k.rolling(200, min_periods=200).mean() - 1}
    b = C.breadth.reindex(dates); vk = C.vk.reindex(dates).ffill()
    sc = np.sign(G['kospi_ma200']).fillna(0) + np.where(b > 50, 1, np.where(b < 30, -1, 0))
    vr = vk / vk.shift(1).rolling(50, min_periods=50).mean()
    sc = sc + np.where(vr > 1.2, -1, np.where(vr < 0.9, 1, 0))
    G['regime'] = pd.Series(sc, index=dates).where(G['kospi_ma200'].notna())
    try:
        bd = Dm._read('idx_bond', ['BAS_DD', 'BND_IDX_GRP_NM', 'TOT_EARNG_IDX', 'BND_IDX_AVG_YD'])
        bd['BAS_DD'] = pd.to_datetime(bd['BAS_DD'])
        num = lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce')
        ktb = bd[bd['BND_IDX_GRP_NM'].str.strip() == 'KTB 지수'].drop_duplicates('BAS_DD').set_index('BAS_DD').sort_index()
        prime = bd[bd['BND_IDX_GRP_NM'].str.strip() == '국고채프라임지수'].drop_duplicates('BAS_DD').set_index('BAS_DD').sort_index()
        ktb_tr = num(ktb['TOT_EARNG_IDX']).reindex(dates).ffill()
        yld = num(prime['BND_IDX_AVG_YD']).reindex(dates).ffill()
    except Exception as e:
        print('채권 자료 생략', e); ktb_tr = yld = None
    pr = lambda x: x.rolling(250, min_periods=120).rank(pct=True) * 100
    comp = [pr(k / k.rolling(125, min_periods=125).mean() - 1), pr(b), pr(-(vk / vk.rolling(50, min_periods=50).mean()))]
    if ktb_tr is not None:
        comp.append(pr(k.pct_change(20) - ktb_tr.pct_change(20)))
    G['fg'] = pd.concat(comp, axis=1).mean(axis=1, skipna=True).where(pd.concat(comp, axis=1).notna().sum(axis=1) >= 3)
    if mey is not None and yld is not None:
        G['yield_gap'] = mey.reindex(dates, method='ffill') * 100 - yld        # %p
    for nm, sid, lag in (('t10y2y', 'T10Y2Y', 1), ('hy', 'BAMLH0A0HYM2', 1)):
        v = _fred(sid, lag)
        if v is not None:
            G[nm] = v.reindex(v.index.union(dates)).ffill(limit=10).reindex(dates)
    v = _fred('CPALTT01KRM659N', 45)           # 한국 CPI 전년비(월, 발표 지연 감안 45일 미룸)
    if v is None:
        v = _fred('KORCPIALLMINMEI', 45)
        if v is not None:
            v = (v / v.shift(12) - 1) * 100
    if v is not None:
        G['cpi'] = v.reindex(v.index.union(dates)).ffill(limit=70).reindex(dates)
    for kk, vv in G.items():
        print('국면', kk, f'{vv.notna().mean():.0%}', flush=True)
    return G


GATES = {'코스피200일선위': lambda G: G['kospi_ma200'] > 0, '국면점수-2초과': lambda G: G['regime'] > -2, '국면점수0이상': lambda G: G['regime'] >= 0,
         '공포탐욕75미만': lambda G: G['fg'] < 75, '공포탐욕25~75': lambda G: (G['fg'] > 25) & (G['fg'] < 75),
         '일드갭3%p+': lambda G: G['yield_gap'] > 3, '일드갭플러스': lambda G: G['yield_gap'] > 0,
         '미국장단기차양수': lambda G: G['t10y2y'] > 0, '하이일드6%미만': lambda G: G['hy'] < 6, 'CPI4%미만': lambda G: G['cpi'] < 4}
GATE_NEED = {'일드갭3%p+': 'yield_gap', '일드갭플러스': 'yield_gap', '미국장단기차양수': 't10y2y', '하이일드6%미만': 'hy', 'CPI4%미만': 'cpi',
             '공포탐욕75미만': 'fg', '공포탐욕25~75': 'fg', '코스피200일선위': 'kospi_ma200', '국면점수-2초과': 'regime', '국면점수0이상': 'regime'}


def gate_mask(name, meta):
    """투자 여부(True=투자). 신호 자료가 없는 날은 판단 불가 → 투자(관문 없음과 같게)."""
    G = meta.get('G', {})
    if not name or name == '없음' or GATE_NEED[name] not in G:
        return None
    g = GATES[name](G)
    return (g | G[GATE_NEED[name]].isna()).astype(bool)


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
        'orgn20_cap', 'pension60_cap', 'peg_inv', 'magic', 'pctb', 'macd_h', 'low_vol252', 'mom6', 'fo20_val', 'debt_low'] + FUND


SEC_CONDS = ['섹:3개월추세', '섹:그날강함', '섹:꺾임아님', '섹:초입', '섹:가격상위5', '섹:수급가속', '섹:수급유입60', '섹:이익확산',
             '섹:1개월상위10', '섹:3개월상위10', '섹:60일선위종목60%+', '섹:신고가비율상위10', '섹:3개월하위아님',
             '섹:20일선위흐름20일+', '섹:20일선위흐름40일+', '섹:20일선위흐름60일+', '섹:대장주20일선위흐름20일+', '섹:대장주20일선위흐름40일+']
STK_CONDS = ['영업이익증가', '이익률개선', '신고가-10%내', '신고가-25%내', '저변동상위50%', '외기20일매수', '외기60일매수', '외기가속', '연기금60일매수',
             '외국인60일매수', '섹터시총3위내', '섹터거래대금3위내', '안달림', '섹터대비1개월덜오름', 'PBR낮은50%', 'ROE상위50%', '자사주90일', '외국계5%90일', '이익수익률상위50%',
             '20일선위', '60일선위', '120일선위', '정배열', '20일신고가', '60일선0~10%위', 'RSI50~65', 'RSI70미만', '1개월-10~+10%', '3개월상위30%', '저변동하위30%아님',
             '외국인연속5일', '기관60일매수', '외기매수개인매도', '매출증가', '매출가속', 'PER하위30%', 'PSR하위30%', '부채비율100%미만', '공급계약90일',
             '대형(시총200위내)', '중형(200~700위)', '거래대금증가1.2~2배', '섹터대비3개월덜오름', '12개월모멘텀상위30%',
             '20일선위흐름20일+', '20일선위흐름40일+', '20일선위흐름10일이하(초입)',
             'PEG1미만', 'PEG0.5이하', '피터린치(PEG1미만+부채100%미만)', '마법공식상위30%', '마법공식상위10%', '볼린저하단(%B<0.2)', '볼린저스퀴즈상단돌파',
             'RSI30이하', 'RSI30+볼린저하단', 'MACD골든크로스5일내', 'MACD0선위', '1년저변동하위30%', '외기20일/거래대금상위30%', '6개월모멘텀상위10%']
TIEB = ['low_vol', 'hi52', 'fo_accel', 'cap']


CW = {}
try:
    CW = json.load(open(os.path.join(ROOT, 'docs', 'sim', 'cond_weights.json')))['weights']
except Exception:
    pass


def _p(lst):
    w = np.array([CW.get(c, 1.0) for c in lst], dtype=float)
    w = np.clip(w, 0.2, 5); return w / w.sum()


def random_inter():
    k_s = int(rng.integers(0, 3)); k_t = int(rng.integers(1, 6))
    return {'mode': '교집합', 'conds': sorted(rng.choice(SEC_CONDS, size=k_s, replace=False, p=_p(SEC_CONDS)).tolist() + rng.choice(STK_CONDS, size=k_t, replace=False, p=_p(STK_CONDS)).tolist()),
            'tie': str(rng.choice(TIEB)), 'N': int(rng.choice([5, 10])), 'cap': int(rng.choice([2, 3]))}


ST_AX = {'1주강도': 'rs_w1', '1개월강도': 'rs_m1', '3개월강도': 'rs_m3', '20일선흐름': 'sec_persist20', '거래대금증가': 'sec_val_chg',
         '거래대금비중증가': 'sec_val_share_chg', '외기60일': 'sf_fo60', '외기20일': 'sf_fo20', '외기가속': 'sf_accel', 'RSI': 'sec_rsi', '60일선위비율': 'tp_above60'}
MIXED = ['기타', '전문서비스', '사업지원', '금융', '지주', '기타제조', '인쇄']
REP = {'시총1위': 'rep_cap', '거래대금1위': 'rep_val', '외기60일1위': 'fo60_cap'}
CROWD = ['없음', '외기60일상위20%', 'RSI70+', '1개월급등상위20%', '셋중하나']
VAL = ['bp', 'ep', 'op_yield', 'roe', 'op_g', 'opm_chg', 'low_vol', 'peg_inv', 'magic', 'debt_low']


def random_state():
    ax = sorted(rng.choice(list(ST_AX), size=int(rng.integers(2, 6)), replace=False, p=_p(list(ST_AX))).tolist())
    K = int(rng.choice([2, 3, 5]))
    return {'mode': '2단계', 'axes': ax, 'K': K, 'Kout': int(K * rng.choice([1.5, 2, 3])), 'brk': bool(rng.random() < 0.6),
            'rsi_cap': bool(rng.random() < 0.5), 'rep': str(rng.choice(list(REP))), 'crowd': str(rng.choice(CROWD)),
            'm': int(rng.choice([1, 2])), 'lag': bool(rng.random() < 0.5), 'profit': bool(rng.random() < 0.6),
            'lead_free': bool(rng.random() < 0.5), 'mix_ex': bool(rng.random() < 0.5),
            'val': sorted(rng.choice(VAL, size=int(rng.integers(3, 8)), replace=False).tolist())}


SUPPLY = {  # 손으로 정리한 공급망·유사 테마 지도(가설)
    '반도체 소자': ['반도체 장비', '반도체 소재', '반도체 부품', 'PCB·기판', '전자 수동부품', '디스플레이·전자부품'],
    '반도체 장비': ['반도체 소재', '반도체 부품', 'PCB·기판', '반도체 소자'], '반도체 부품': ['반도체 장비', 'PCB·기판', '반도체 소재'],
    '반도체 소재': ['반도체 장비', '반도체 부품', '화학'], 'PCB·기판': ['반도체 부품', '반도체 장비', '전자 수동부품'], '전자 수동부품': ['PCB·기판', '반도체 부품', '스마트폰 부품'],
    '2차전지': ['비철금속', '화학', '전기장비', '금속가공', '철강·금속'], '조선': ['금속가공', '기계', '철강', '운송장비'], '방산·우주': ['기계', '금속가공', '항공·방산', '운송장비'],
    '원자력': ['전력기기', '건설', '엔지니어링', '금속가공'], '전력기기': ['원자력', '전기장비', '비철금속', '건설'], '자동차': ['자동차부품', '자동차 전장·부품', '로봇'],
    '로봇': ['기계', '자동차 전장·부품', '전기장비'], '바이오·헬스케어': ['제약·바이오', '바이오연구', '의료기기', '의료·정밀기기'], '제약·바이오': ['바이오·헬스케어', '바이오연구'],
    '엔터·콘텐츠': ['미디어·콘텐츠', '게임', '레저'], '게임': ['엔터·콘텐츠', '소프트웨어·게임'], '화장품': ['음식료', '유통'], '증권': ['은행', '보험', '지주'], '은행': ['증권', '보험', '지주'],
    '건설': ['건자재', '원자력', '엔지니어링'], '통신장비': ['통신·RF', '스마트폰 부품'], '디스플레이': ['디스플레이·전자부품', '반도체 장비']}
LEADX = {'외기20일금액': 'fo20_amt', '외기60일금액': 'fo60_amt', '시총': 'logcap', '외기20일': 'fo20_cap', '외기60일': 'fo60_cap', '외기가속': 'fo_accel', '1개월섹터대비': 'r1rel', '3개월섹터대비': 'rs3', '거래대금순위': 'rep_val', '신고가근접': 'hi52', '20일선흐름': 'persist20'}


def random_third():
    m = random_state()
    m.update({'mode': '3단계', 'lead_x': sorted(rng.choice(list(LEADX), size=int(rng.integers(1, 5)), replace=False).tolist()),
              'nb': str(rng.choice(['상관', '순위상승', '공급망', '셋다'], p=[0.4, 0.2, 0.2, 0.2])), 'nb_n': int(rng.choice([1, 2, 3])),
              'parts': str(rng.choice(['대장+차기가치', '대장만', '차기가치만'], p=[0.3, 0.1, 0.6]))})   # 1차 결과에서 차기가치만이 가장 나아 더 자주
    return m


def random_switch():
    """국면 전환(리서치 보고서 6장): 위험선호 국면엔 모멘텀(중소형), 위험회피 국면엔 방어 멀티팩터(대형: 1년 저변동+ROE+저부채, 순위 또는 Z점수)."""
    return {'mode': '국면전환', 'sig': str(rng.choice([g for g in GATES])), 'on': str(rng.choice(['6개월모멘텀상위10%+MACD골든', '6개월모멘텀+MACD점수'])),
            'on_size': str(rng.choice(['중소형', '전체'])), 'off': str(rng.choice(['방어3', '방어3+마법공식', '현금'])), 'agg': str(rng.choice(['순위', 'Z'])),
            'sec': str(rng.choice(['전체', '추세'])), 'N': int(rng.choice([10, 20])), 'cap': int(rng.choice([2, 3]))}


def random_method():
    u = rng.random()
    if u < 0.35:
        m = random_third()
    elif u < 0.5:
        m = random_state()
    elif u < 0.8:
        m = random_inter()
    elif u < 0.9:
        m = _random_score()
    else:
        return random_switch()
    if rng.random() < 0.4:                    # 시장 국면 관문(꺼지면 현금)
        m['mg'] = str(rng.choice(list(GATES)))
    return m


_CORR = {}


def sector_corr(S, meta, step=10, win=60):
    """섹터 지수 일간 수익률의 최근 60일 상관(10일마다 계산, 사이는 직전 값)."""
    if 'c' in _CORR:
        return _CORR['c']
    secs = meta['secs']; R = S['sec_ret'][secs].values
    idx = meta['idx']; out = {}
    for i in range(win, len(idx), step):
        x = R[i - win:i]
        ok = ~np.isnan(x).any(axis=0)
        c = np.full((len(secs), len(secs)), np.nan)
        if ok.sum() > 2:
            c[np.ix_(ok, ok)] = np.corrcoef(x[:, ok].T)
        out[i] = c
    _CORR['c'] = out
    return out


def neighbors(m, okS, S, meta):
    """주도 섹터의 차기·유사 섹터(주도 섹터 자신은 제외)."""
    secs = meta['secs']; idx = meta['idx']
    nb = np.zeros(okS.shape, bool); lead = okS.values
    sidx = {s_: j for j, s_ in enumerate(secs)}
    big = (meta['nS'][secs] >= 5).values
    if m['nb'] in ('상관', '셋다'):
        C = sector_corr(S, meta); keys = sorted(C)
        cur = None; k = 0
        for i in range(len(idx)):
            while k < len(keys) and keys[k] <= i:
                cur = C[keys[k]]; k += 1
            if cur is None or not lead[i].any():
                continue
            for j in np.where(lead[i])[0]:
                c = np.where(big[i] & ~lead[i], cur[j], np.nan)
                if np.all(np.isnan(c)):
                    continue
                top = np.argsort(np.nan_to_num(c, nan=-9))[::-1][:m['nb_n']]
                nb[i, top] = True
    if m['nb'] in ('순위상승', '셋다'):
        sc = sum(S[ST_AX[a]][secs].where(meta['nS'][secs] >= 5).rank(axis=1, pct=True) for a in m['axes']) / len(m['axes'])
        up = (sc - sc.shift(20)).values
        for i in range(len(idx)):
            if not lead[i].any():
                continue
            u = np.where(~lead[i] & big[i], up[i], np.nan)
            if np.all(np.isnan(u)):
                continue
            nb[i, np.argsort(np.nan_to_num(u, nan=-9))[::-1][:m['nb_n'] * max(1, lead[i].sum())]] = True
    if m['nb'] in ('공급망', '셋다'):
        for s_, adj in SUPPLY.items():
            if s_ not in sidx:
                continue
            j = sidx[s_]; cols = [sidx[a] for a in adj if a in sidx]
            if cols:
                nb[np.ix_(lead[:, j], cols)] = True
        nb &= ~lead & big
    return pd.DataFrame(nb, index=okS.index, columns=secs)


def evaluate_third(m, X, Y, S, meta):
    uni, sec, idx = meta['uni'], meta['sec'], meta['idx']
    okS = state_sectors(m, S, meta)
    score = pd.DataFrame(np.nan, index=idx, columns=uni.columns, dtype='float32')
    if m['parts'] in ('대장+차기가치', '대장만'):
        # 대장주: 실적과 무관, 외기 매수·섹터 대비 강도·거래대금·신고가·20일선 흐름으로
        base = sec_to_stock(okS, meta) & uni
        ls = sum(X[LEADX[a]].where(base).rank(axis=1, pct=True).fillna(0) for a in m['lead_x']).where(base)
        for s_ in meta['secs']:
            if not okS[s_].any():
                continue
            cols = sec.index[sec == s_]
            r = ls[cols].rank(axis=1, ascending=False, method='first')
            score[cols] = np.where(r.values == 1, 2.0, score[cols].values)
    if m['parts'] in ('대장+차기가치', '차기가치만'):
        nbS = neighbors(m, okS, S, meta)
        okn = sec_to_stock(nbS, meta) & uni
        if m['profit']:
            okn &= (X['op_g'] > 0) | (X['opm_chg'] > 0)
        vs = sum(X[k].where(okn).rank(axis=1, pct=True).fillna(0) for k in m['val'])
        if m['lag']:
            vs = vs.where(X['rs3'] < 0)
        vs = vs.where(okn)
        for s_ in meta['secs']:
            if not nbS[s_].any():
                continue
            cols = sec.index[sec == s_]
            r = vs[cols].rank(axis=1, ascending=False, method='first')
            sel = (r.values <= m['m']) & np.isnan(score[cols].values)
            score[cols] = np.where(sel, 1.0 + 0.5 * vs[cols].values / len(m['val']), score[cols].values)
    return finish(dict(m, cap=99, N=m['K'] + 6), score, Y, meta)   # 대장주(점수 2) 먼저, 차기 가치주는 합쳐 6개까지


def state_sectors(m, S, meta):
    """주도 섹터를 상태로 관리: 여러 기간·거래대금·RSI·외기 수급 순위로 들어오고(K 안), 순위가 Kout 밖으로 밀리거나 꺾이면(4일 연속 20일선 이탈) 나감."""
    secs = meta['secs']
    ok = (meta['nS'][secs] >= 5)
    if m.get('mix_ex'):
        ok = ok & ~ok.columns.isin(MIXED)            # 업종이 섞인 묶음(전문서비스·사업지원 등) 제외
    sc = sum(S[ST_AX[a]][secs].where(ok).rank(axis=1, pct=True) for a in m['axes']) / len(m['axes'])
    if m['rsi_cap']:
        sc = sc.where(~(S['sec_rsi'][secs] > 80), sc - 0.3)      # 과열(RSI 80 넘음)은 점수 깎음
    rk = sc.rank(axis=1, ascending=False).values
    brk = (S['sec_persist20'][secs] == 0).values if m['brk'] else np.zeros(rk.shape, bool)
    held = np.zeros(len(secs), bool)
    out = np.zeros(rk.shape, bool)
    for i in range(rk.shape[0]):
        r = rk[i]
        stay = held & (r <= m['Kout']) & ~brk[i]
        new = (r <= m['K']) & ~brk[i]
        h = stay | new
        if h.sum() > m['K'] + 2:          # 너무 많아지면 순위 좋은 것만
            keep = np.argsort(np.where(h, r, 1e9))[:m['K'] + 2]
            h = np.zeros_like(h); h[keep] = True
        held = h; out[i] = h
    return pd.DataFrame(out, index=sc.index, columns=secs)


def evaluate_state(m, X, Y, S, meta):
    uni, sec, idx = meta['uni'], meta['sec'], meta['idx']
    okS = state_sectors(m, S, meta)
    base = sec_to_stock(okS, meta) & uni
    ok = base.copy()
    if m['profit']:
        ok &= (X['op_g'] > 0) | (X['opm_chg'] > 0)
    rep = X[REP[m['rep']]].where(base if m.get('lead_free') else ok)   # 대장주는 실적 관문 없이 고를 수 있게(실적 회복 초입 대장주)
    pr = lambda k: X[k].rank(axis=1, pct=True)
    crowd = pd.DataFrame(False, index=idx, columns=uni.columns)
    if m['crowd'] in ('외기60일상위20%', '셋중하나'):
        crowd |= pr('fo60_cap') > 0.8
    if m['crowd'] in ('RSI70+', '셋중하나'):
        crowd |= X['rsi'] > 70
    if m['crowd'] in ('1개월급등상위20%', '셋중하나'):
        crowd |= pr('r1rel') > 0.8
    vs = sum(X[k].where(ok).rank(axis=1, pct=True).fillna(0) for k in m['val'])
    if m['lag']:
        vs = vs.where(X['rs3'] < 0)        # 섹터보다 덜 오른 것만
    vs = vs.where(ok)
    score = pd.DataFrame(np.nan, index=idx, columns=uni.columns, dtype='float32')
    for s_ in meta['secs']:
        cols = sec.index[sec == s_]
        if not okS[s_].any():
            continue
        lr = rep[cols].rank(axis=1, ascending=False, method='first')
        lead = (lr == 1)
        lead_crowded = (lead & crowd[cols]).any(axis=1)
        vr = vs[cols].where(~lead).rank(axis=1, ascending=False, method='first')
        pick_lead = lead & ~crowd[cols]
        pick_val = (vr <= m['m']) & lead_crowded.values[:, None]
        sc_ = pd.DataFrame(np.nan, index=idx, columns=cols)
        sc_[pick_val] = 1.0; sc_[pick_lead] = 2.0
        score[cols] = sc_.values
    mm = dict(m, cap=99, N=999)
    return finish(mm, score, Y, meta)


def evaluate(m, X, Y, S, meta, signs):
    if m.get('mode') == '국면전환':
        return evaluate_switch(m, X, Y, S, meta)
    if m.get('mode') == '3단계':
        return evaluate_third(m, X, Y, S, meta)
    if m.get('mode') == '2단계':
        return evaluate_state(m, X, Y, S, meta)
    if m.get('mode') == '교집합':
        return evaluate_inter(m, X, Y, S, meta)
    return evaluate_score(m, X, Y, S, meta, signs)


def _random_score():
    m = {'gate': rng.choice(SEC_GATES), 'axes': sorted(rng.choice(list(SEC_AXES), size=rng.integers(1, 4), replace=False).tolist()),
         'K': int(rng.choice([2, 3, 5, 8, 99])), 'facs': sorted(rng.choice(POOL, size=rng.integers(2, 9), replace=False).tolist()),
         'N': int(rng.choice([5, 8, 12])), 'cap': int(rng.choice([1, 2, 3])), 'profit': bool(rng.random() < 0.6), 'noran': bool(rng.random() < 0.6),
         'sign': rng.choice(['걷기식', '걷기식+고정가점']), 'agg': str(rng.choice(['순위', 'Z']))}
    return m


def mname(m):
    g = f"|국면:{m['mg']}" if m.get('mg') and m['mg'] != '없음' else ''
    return _mname(m) + g


def _mname(m):
    if m.get('mode') == '국면전환':
        return f"국면전환|신호:{m['sig']}|켜짐:{m['on']}({m['on_size']})|꺼짐:{m['off']}|{m['agg']}|섹:{m['sec']}|N{m['N']}c{m['cap']}"
    if m.get('mode') == '3단계':
        return (f"3단계|섹:{'+'.join(m['axes'])}|K{m['K']}~{m['Kout']}|{'꺾임교체' if m['brk'] else '-'}|{'과열감점' if m['rsi_cap'] else '-'}"
                f"|대장:{'+'.join(m['lead_x'])}|차기:{m['nb']}{m['nb_n']}|가치{m['m']}:{','.join(m['val'])}{'(덜오름)' if m['lag'] else ''}|{'이익' if m['profit'] else '-'}"
                f"|{'혼합섹터제외' if m.get('mix_ex') else '-'}|{m['parts']}")
    if m.get('mode') == '2단계':
        return (f"2단계|섹:{'+'.join(m['axes'])}|K{m['K']}~{m['Kout']}|{'꺾임교체' if m['brk'] else '-'}|{'과열감점' if m['rsi_cap'] else '-'}"
                f"|대장:{m['rep']}{'(관문없음)' if m.get('lead_free') else ''}|꽉참:{m['crowd']}|가치{m['m']}:{','.join(m['val'])}{'(덜오름)' if m['lag'] else ''}|{'이익' if m['profit'] else '-'}|{'혼합섹터제외' if m.get('mix_ex') else '-'}")
    if m.get('mode') == '교집합':
        return f"교집합|{'∩'.join(m['conds'])}|순서:{m['tie']}|N{m['N']}c{m['cap']}"
    return f"{m['gate']}|{'+'.join(m['axes'])}|K{'전체' if m['K'] == 99 else m['K']}|{','.join(m['facs'])}|N{m['N']}c{m['cap']}|{'이익' if m['profit'] else '-'}|{'안달림' if m['noran'] else '-'}|{m['sign']}" + ('|Z점수' if m.get('agg') == 'Z' else '')


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


def sec_to_stock(B, meta):
    sec = meta['sec']
    out = pd.DataFrame(False, index=meta['idx'], columns=meta['uni'].columns)
    for s in meta['secs']:
        cols = sec.index[sec == s]
        out[cols] = np.repeat(B[s].fillna(False).values[:, None].astype(bool), len(cols), axis=1)
    return out


_CC = {}


def cond(c, X, S, meta):
    if c in _CC:
        return _CC[c]
    secs = meta['secs']
    pct = lambda k: X[k].rank(axis=1, pct=True)
    tr = (S['tp_above60'][secs] >= 0.6) & (S['tp_align'][secs] >= 0.5)
    m = {'섹:3개월추세': lambda: sec_to_stock(tr, meta), '섹:그날강함': lambda: sec_to_stock(S['stage_strong'][secs] > 0, meta),
         '섹:꺾임아님': lambda: sec_to_stock((S['br_below'][secs].fillna(0) + S['br_dead'][secs].fillna(0) + S['br_flow'][secs].fillna(0)) < 2, meta),
         '섹:초입': lambda: sec_to_stock(S['rc_start'][secs] > 0, meta),
         '섹:가격상위5': lambda: sec_to_stock(((S['sp_slope40w'][secs].rank(axis=1, pct=True) + S['sp_ret120'][secs].rank(axis=1, pct=True)).rank(axis=1, ascending=False, method='first') <= 5) & (meta['nS'][secs] >= 5), meta),
         '섹:수급가속': lambda: sec_to_stock(S['sf_accel'][secs] > 0, meta), '섹:수급유입60': lambda: sec_to_stock(S['sf_fo60'][secs] > 0, meta),
         '섹:이익확산': lambda: sec_to_stock(S['se_opg'][secs].gt(S['se_opg'][secs].median(axis=1), axis=0), meta) if 'se_opg' in S else None,
         '영업이익증가': lambda: X['op_g'] > 0, '이익률개선': lambda: X['opm_chg'] > 0, '신고가-10%내': lambda: X['hi52'] > -0.1, '신고가-25%내': lambda: X['hi52'] > -0.25,
         '저변동상위50%': lambda: pct('low_vol') > 0.5, '외기20일매수': lambda: X['fo20_cap'] > 0, '외기60일매수': lambda: X['fo60_cap'] > 0, '외기가속': lambda: X['fo_accel'] > 0,
         '연기금60일매수': lambda: X['pension60_cap'] > 0, '외국인60일매수': lambda: X['frgn60_cap'] > 0, '섹터시총3위내': lambda: X['rep_cap'] >= -3,
         '섹터거래대금3위내': lambda: X['rep_val'] >= -3, '안달림': lambda: ~((pct('r1rel') > 0.8) | (pct('val_surge') > 0.8)), '섹터대비1개월덜오름': lambda: X['r1rel'] < 0,
         'PBR낮은50%': lambda: pct('bp') > 0.5, 'ROE상위50%': lambda: pct('roe') > 0.5, '자사주90일': lambda: X['buyback90'] > 0, '외국계5%90일': lambda: X['major_foreign90'] > 0,
         '이익수익률상위50%': lambda: pct('ep') > 0.5,
         '섹:1개월상위10': lambda: sec_to_stock(S['sec_r1'][secs].where(meta['nS'][secs] >= 5).rank(axis=1, ascending=False) <= 10, meta),
         '섹:3개월상위10': lambda: sec_to_stock(S['sec_r3'][secs].where(meta['nS'][secs] >= 5).rank(axis=1, ascending=False) <= 10, meta),
         '섹:60일선위종목60%+': lambda: sec_to_stock(S['sec_above60_share'][secs] >= 0.6, meta),
         '섹:신고가비율상위10': lambda: sec_to_stock(S['sb_nearhi'][secs].where(meta['nS'][secs] >= 5).rank(axis=1, ascending=False) <= 10, meta),
         '섹:20일선위흐름20일+': lambda: sec_to_stock(S['sec_persist20'][secs] >= 20, meta), '섹:20일선위흐름40일+': lambda: sec_to_stock(S['sec_persist20'][secs] >= 40, meta),
         '섹:20일선위흐름60일+': lambda: sec_to_stock(S['sec_persist20'][secs] >= 60, meta),
         '섹:대장주20일선위흐름20일+': lambda: sec_to_stock(S['lead_persist20'][secs] >= 20, meta), '섹:대장주20일선위흐름40일+': lambda: sec_to_stock(S['lead_persist20'][secs] >= 40, meta),
         '20일선위흐름20일+': lambda: X['persist20'] >= 20, '20일선위흐름40일+': lambda: X['persist20'] >= 40, '20일선위흐름10일이하(초입)': lambda: (X['persist20'] >= 3) & (X['persist20'] <= 10),
         '섹:3개월하위아님': lambda: sec_to_stock(S['sec_r3'][secs].rank(axis=1, pct=True) > 0.3, meta),
         '20일선위': lambda: X['g20'] > 0, '60일선위': lambda: X['g60'] > 0, '120일선위': lambda: X['g120'] > 0, '정배열': lambda: X['align'] > 0,
         '20일신고가': lambda: X['brk20'] > 0, '60일선0~10%위': lambda: (X['g60'] > 0) & (X['g60'] < 0.1), 'RSI50~65': lambda: (X['rsi'] >= 50) & (X['rsi'] <= 65),
         'RSI70미만': lambda: X['rsi'] < 70, '1개월-10~+10%': lambda: (X['r1'] > -0.1) & (X['r1'] < 0.1), '3개월상위30%': lambda: pct('r3') > 0.7,
         '저변동하위30%아님': lambda: pct('low_vol') > 0.3, '외국인연속5일': lambda: X['frgn_streak'] >= 5, '기관60일매수': lambda: X['orgn60_cap'] > 0,
         '외기매수개인매도': lambda: X['fo_vs_prsn'] > 0, '매출증가': lambda: X['ry'] > 0, '매출가속': lambda: X['accel_d'] > 0,
         'PER하위30%': lambda: X['per'].where(X['per'] > 0).rank(axis=1, pct=True) < 0.3, 'PSR하위30%': lambda: X['psr'].where(X['psr'] > 0).rank(axis=1, pct=True) < 0.3,
         '부채비율100%미만': lambda: X['debt_ratio'] < 100, '공급계약90일': lambda: X['supply90'] > 0,
         '대형(시총200위내)': lambda: meta['cap'].rank(axis=1, ascending=False) <= 200, '중형(200~700위)': lambda: meta['cap'].rank(axis=1, ascending=False).between(201, 700) if False else ((meta['cap'].rank(axis=1, ascending=False) > 200) & (meta['cap'].rank(axis=1, ascending=False) <= 700)),
         '거래대금증가1.2~2배': lambda: (X['val_surge'] >= 1.2) & (X['val_surge'] <= 2), '섹터대비3개월덜오름': lambda: X['rs3'] < 0, '12개월모멘텀상위30%': lambda: pct('mom12_1') > 0.7,
         'PEG1미만': lambda: X['peg'] < 1, 'PEG0.5이하': lambda: X['peg'] <= 0.5, '피터린치(PEG1미만+부채100%미만)': lambda: (X['peg'] < 1) & (X['debt_ratio'] < 100),
         '마법공식상위30%': lambda: pct('magic') > 0.7, '마법공식상위10%': lambda: pct('magic') > 0.9,
         '볼린저하단(%B<0.2)': lambda: X['pctb'] < 0.2, '볼린저스퀴즈상단돌파': lambda: (X['bb_squeeze'].shift(1) > 0) & (X['pctb'] > 1),
         'RSI30이하': lambda: X['rsi'] <= 30, 'RSI30+볼린저하단': lambda: (X['rsi'] <= 30) & (X['pctb'] < 0.1),
         'MACD골든크로스5일내': lambda: X['macd_gc5'] > 0, 'MACD0선위': lambda: X['macd_pos'] > 0,
         '1년저변동하위30%': lambda: pct('low_vol252') > 0.7, '외기20일/거래대금상위30%': lambda: pct('fo20_val') > 0.7, '6개월모멘텀상위10%': lambda: pct('mom6') > 0.9}
    try:
        v = m[c]()
    except KeyError:
        v = None
    _CC[c] = v.fillna(False).astype(bool) if v is not None else None
    return _CC[c]





def evaluate_inter(m, X, Y, S, meta):
    uni = meta['uni']
    ok = uni.copy()
    for c in m['conds']:
        v = cond(c, X, S, meta)
        if v is None:
            return None
        ok &= v
    score = (meta['cap'] if m['tie'] == 'cap' else X[m['tie']]).where(ok)
    return finish(m, score, Y, meta)


def zs(x):
    """Z점수(같은 날 후보 안에서). 이상치는 ±3에서 자름(보고서: Z점수는 이상치에 민감)."""
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-12, axis=0).clip(-3, 3) / 6


def evaluate_switch(m, X, Y, S, meta):
    uni = meta['uni']
    g = gate_mask(m['sig'], meta)
    if g is None:
        return None
    capr = meta['cap'].rank(axis=1, ascending=False)
    base = uni.copy()
    if m['sec'] == '추세':
        secs = meta['secs']
        base &= sec_to_stock((S['tp_above60'][secs] >= 0.6) & (S['tp_align'][secs] >= 0.5), meta)
    agg = (lambda v, ok: zs(v.where(ok))) if m['agg'] == 'Z' else (lambda v, ok: v.where(ok).rank(axis=1, pct=True))
    on_ok = base & (capr > 200) if m['on_size'] == '중소형' else base
    if m['on'] == '6개월모멘텀상위10%+MACD골든':
        mo = X['mom6'].where(on_ok)
        on = mo.where((mo.rank(axis=1, pct=True) > 0.9) & (X['macd_gc5'] > 0))
    else:
        on = agg(X['mom6'], on_ok) + agg(X['macd_h'], on_ok)
    off_ok = base & (capr <= 200)
    if m['off'] == '현금':
        off = pd.DataFrame(np.nan, index=uni.index, columns=uni.columns)
    else:
        fs = ['low_vol252', 'roe', 'debt_low'] + (['magic'] if m['off'] == '방어3+마법공식' else [])
        off = sum(agg(X[f], off_ok).fillna(0) for f in fs).where(off_ok)
    gv = g.values[:, None]
    score = pd.DataFrame(np.where(gv, on.values, off.values), index=uni.index, columns=uni.columns)
    return finish(dict(m, mg=m['sig']) if m['off'] == '현금' else m, score, Y, meta)   # 꺼짐=현금이면 그날은 현금으로 채점


def evaluate_score(m, X, Y, S, meta, signs):
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
        r = zs(X[f].where(ok)) if m.get('agg') == 'Z' else X[f].where(ok).rank(axis=1, pct=True) - 0.5
        score += r.fillna(0).mul(sg, axis=0); used += 1
    if used == 0:
        return None
    score = score.where(ok)
    return finish(m, score, Y, meta)


def finish(m, score, Y, meta):
    idx, uni, sec = meta['idx'], meta['uni'], meta['sec']
    # 섹터당 cap, 전체 N
    srank = pd.DataFrame(np.nan, index=idx, columns=uni.columns, dtype='float32')
    for s in meta['secs']:
        cols = sec.index[sec == s]
        srank[cols] = score[cols].rank(axis=1, ascending=False, method='first')
    sc2 = score.where(srank <= m['cap'])
    pick = (sc2.rank(axis=1, ascending=False, method='first') <= m['N']) & sc2.notna()
    gm = gate_mask(m.get('mg'), meta)
    if gm is not None:
        pick &= gm.values[:, None]                 # 국면 관문이 꺼진 날은 현금
    cash = (~gm) if gm is not None else pd.Series(False, index=idx)
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
    for h in H:                                    # 현금인 날: 수익 0, 시장 대비 = -시장
        res[f'ex{h}'] = res[f'ex{h}'].where(~cash, -Y[f'ret{h}'].where(uni).mean(axis=1) * 100)
    R = pd.DataFrame(res)
    R['cash'] = cash.astype('float32')
    R.loc[(R['n'] == 0) & ~cash, [c for c in R.columns if c not in ('n', 'cash')]] = np.nan
    R = R.resample('ME').mean().astype('float32')        # 월 평균으로 저장(누적 용량 줄이기)
    return R, pick


def yearly_stats(R):
    y = R.groupby(R.index.year)
    return y.mean()


def summarize(name, R):
    Yr = yearly_stats(R)
    pre = Yr.loc[2016:2023]; post = Yr.loc[2024:]
    t = lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(v.count())) if v.count() > 2 else np.nan
    return {'방법': name, '63일 16~23': pre['ex63'].mean(), '플러스 해': int((pre['ex63'] > 0).sum()), '해 수': int(pre['ex63'].count()), 't': t(pre['ex63']),
            '최악': pre['ex63'].min(), '21일 16~23': pre['ex21'].mean(), '126일 16~23': pre['ex126'].mean() if 'ex126' in pre else np.nan,
            '126일 플러스 해': int((pre['ex126'] > 0).sum()) if 'ex126' in pre else np.nan, '섹터대비 63': pre['sx63'].mean(), '섹터 선택 63': pre['secx63'].mean(),
            '수급유입 63': pre['fx63'].mean(), '24~ 63일': post['ex63'].mean(), '안정성': R['stab'].mean(), '평균 종목 수': R['n'].mean()}


def walkforward(store, start=2017):
    """해마다 이전 해들의 연도 성과(63일 결과 확인 뒤)로 방법 고르기: 1등, 상위5 평균, 상위20 평균."""
    names = [k for k, v in store.items() if navg(v) >= 2]     # 평균 2종목 미만은 우연이 커서 제외
    Yall = pd.DataFrame({k: yearly_stats(store[k])['ex63'] for k in names})
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
        pr = Yall.loc[Yr].rank(pct=True)     # 그해 전체 방법 중 위치(운이면 50% 근처, 실력이면 꾸준히 위)
        rows.append({'year': Yr, '1등': Yall.at[Yr, top[0]], '상위5': Yall.loc[Yr, top[:5]].mean(), '상위20': Yall.loc[Yr, top].mean(),
                     '전체 평균': Yall.loc[Yr].mean(), '상위20 그해 백분위': pr[top].mean() * 100, '1등 그해 백분위': pr[top[0]] * 100, '1등 방법': top[0]})
    return pd.DataFrame(rows).set_index('year'), picks


def state_marg(T):
    x = T[T.index.str.startswith('2단계')]
    if x.empty:
        return pd.DataFrame()
    base = x['63일 16~23'].mean(); rows = []
    parts = x.index.str.split('|')
    tests = [(f'섹터재료:{a}', parts.str[1].str.contains(a, regex=False)) for a in ST_AX]
    for lab, fn in (('꺾임교체', 3), ('과열감점', 4)):
        tests.append((lab, parts.str[fn] == lab))
    for r in REP:
        tests.append((f'대장:{r}', parts.str[5].str.startswith(f'대장:{r}')))
    tests.append(('대장 관문없음', parts.str[5].str.contains('관문없음')))
    for c in CROWD:
        tests.append((f'꽉참:{c}', parts.str[6] == f'꽉참:{c}'))
    tests.append(('덜오름', parts.str[7].str.contains('덜오름')))
    tests.append(('혼합섹터제외', parts.str[-1] == '혼합섹터제외'))
    for lab, mk in tests:
        y = x[np.asarray(mk, dtype=bool)]
        if len(y):
            rows.append({'항목': lab, '방법 수': len(y), '63일 차이': y['63일 16~23'].mean() - base, '플러스 해': y['플러스 해'].mean(), '24~': y['24~ 63일'].mean()})
    return pd.DataFrame(rows).sort_values('63일 차이', ascending=False)


def third_marg(T):
    x = T[T.index.str.startswith('3단계')]
    if x.empty:
        return pd.DataFrame()
    base = x['63일 16~23'].mean(); parts = x.index.str.split('|'); rows = []
    tests = [(f'구성:{p}', parts.str[-1] == p) for p in ('대장+차기가치', '대장만', '차기가치만')]
    tests += [(f'차기:{n}', parts.str[6].str.startswith(f'차기:{n}')) for n in ('상관', '순위상승', '공급망', '셋다')]
    tests += [(f'대장재료:{a}', parts.str[5].str.contains(a, regex=False)) for a in LEADX]
    tests += [(f'섹터재료:{a}', parts.str[1].str.contains(a, regex=False)) for a in ST_AX]
    for lab, mk in tests:
        y = x[np.asarray(mk, dtype=bool)]
        if len(y):
            rows.append({'항목': lab, '방법 수': len(y), '63일 차이': y['63일 16~23'].mean() - base, '플러스 해': y['플러스 해'].mean(),
                         '섹터대비 63': y['섹터대비 63'].mean(), '수급유입 63': y['수급유입 63'].mean(), '24~': y['24~ 63일'].mean()})
    return pd.DataFrame(rows).sort_values('63일 차이', ascending=False)


def inter_marg(T):
    x = T[T.index.str.startswith('교집합')]
    if x.empty:
        return pd.DataFrame()
    base = x['63일 16~23'].mean(); bf = x['수급유입 63'].mean()
    rows = []
    for c in SEC_CONDS + STK_CONDS:
        y = x[x.index.str.contains(c, regex=False)]
        if len(y):
            rows.append({'조건': c, '방법 수': len(y), '63일 차이': y['63일 16~23'].mean() - base, '플러스 해': y['플러스 해'].mean(),
                         '수급유입 차이': y['수급유입 63'].mean() - bf, '24~': y['24~ 63일'].mean()})
    return pd.DataFrame(rows).sort_values('63일 차이', ascending=False)


def navg(v):
    """현금(국면 관문 꺼짐)인 날을 뺀 평균 종목 수."""
    c = v['cash'].mean() if 'cash' in v else 0
    return v['n'].mean() / max(1 - (c if c == c else 0), 0.05)


def gate_marg(T):
    """시장 국면 관문(보고서 매크로 필터): 같은 계열에서 관문 있는 방법 평균 - 관문 없는 방법 평균."""
    fam = T.index.str.split('|').str[0]
    fam = pd.Series(np.where(pd.Series(fam).isin(['2단계', '3단계', '교집합', '국면전환']), fam, '점수형'), index=T.index)
    has = T.index.str.contains('|국면:', regex=False)
    rows = []
    for g in GATES:
        mk = T.index.str.endswith(f'|국면:{g}')
        if mk.sum() == 0:
            continue
        d = []
        for f in set(fam[mk]):
            a = T[mk & (fam == f).values]; b = T[~has & (fam == f).values & ~T.index.str.startswith('국면전환')]
            if len(a) and len(b):
                d.append((len(a), a['63일 16~23'].mean() - b['63일 16~23'].mean(), a['플러스 해'].mean() - b['플러스 해'].mean(), a['24~ 63일'].mean() - b['24~ 63일'].mean()))
        if d:
            w = np.array([x[0] for x in d], float)
            rows.append({'관문': g, '방법 수': int(w.sum()), '63일 차이': np.average([x[1] for x in d], weights=w), '플러스 해 차이': np.average([x[2] for x in d], weights=w),
                         '24~ 차이': np.average([x[3] for x in d], weights=w)})
    sw = T[T.index.str.startswith('국면전환')]
    for lab, key in (('국면전환', 1), ('국면전환', 2), ('국면전환', 3), ('국면전환', 4)):
        if len(sw):
            v = sw.index.str.split('|').str[key]
            for val in sorted(set(v)):
                x = sw[v == val]
                rows.append({'관문': f'{lab}:{val}', '방법 수': len(x), '63일 차이': x['63일 16~23'].mean(), '플러스 해 차이': x['플러스 해'].mean(), '24~ 차이': x['24~ 63일'].mean()})
    sc = T[fam.values == '점수형']
    if len(sc):
        z = sc.index.str.endswith('|Z점수') | sc.index.str.contains('|Z점수|', regex=False)
        for lab, mk in (('점수형 순위합산', ~z), ('점수형 Z점수합산', z)):
            x = sc[mk]
            if len(x):
                rows.append({'관문': lab, '방법 수': len(x), '63일 차이': x['63일 16~23'].mean(), '플러스 해 차이': x['플러스 해'].mean(), '24~ 차이': x['24~ 63일'].mean()})
    return pd.DataFrame(rows)


def marginals(T):
    T = T[~T.index.str.startswith(('교집합', '2단계', '3단계', '국면전환'))]
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
         f"판정 구간(2020~2023, 조건 샘플링 학습에 안 쓴 해): 1등 {wf.loc[2020:2023, '1등'].mean():+.2f}%p · 상위20 {wf.loc[2020:2023, '상위20'].mean():+.2f}%p, 상위20 백분위 {wf.loc[2020:2023, '상위20 그해 백분위'].mean():.0f}%", '',
         '운인지 실력인지: 걷기식으로 고른 방법들이 그해 전체 방법 중 몇 % 위치였나(50% = 운과 같음). 꾸준히 60% 이상이면 실력 쪽.', '',
         f"2017~2023 상위20 평균 백분위 {pre['상위20 그해 백분위'].mean():.0f}%, 1등 평균 백분위 {pre['1등 그해 백분위'].mean():.0f}%, 상위20이 50% 넘은 해 {(pre['상위20 그해 백분위'] > 50).sum()}/{len(pre)}", '',
         wf.round(2).to_markdown(), '', '## 교집합 조건별 효과(그 조건이 들어간 방법 평균 - 전체 평균, 63일 16~23 %p)', '', inter_marg(T).round(2).to_markdown(index=False), '',
         '## 3단계(주도 섹터 대장주 + 차기·유사 섹터 가치주) 항목별 효과(63일 16~23 %p, 3단계 평균 대비)', '', third_marg(T).round(2).to_markdown(index=False), '',
         '## 리서치 보고서 반영: 시장 국면 관문·국면 전환·Z점수(63일 16~23 %p; 관문은 같은 계열 관문 없는 방법 대비, 꺼진 날은 현금으로 채점)', '',
         '국면 신호 자료 범위: ' + ', '.join(f"{k} {v.notna().mean():.0%}" for k, v in meta.get('G', {}).items()) + '(자료 없는 날은 관문 없음과 같게 처리).', '',
         gate_marg(T).round(2).to_markdown(index=False), '',
         '## 2단계 상태형 로직 항목별 효과(63일 16~23 %p, 2단계 로직 평균 대비)', '', state_marg(T).round(2).to_markdown(index=False), '',
         '## 로직 계열별 평균', '', T.assign(계열=T.index.str.split('|').str[0].where(T.index.str.split('|').str[0].isin(['2단계', '3단계', '교집합', '국면전환']), '점수형')).groupby('계열')[['63일 16~23', '플러스 해', '126일 16~23', '126일 플러스 해', '24~ 63일', '안정성']].mean().round(2).to_markdown(), '', '## 무엇이 효과가 있나(방법들의 항목별 평균, 63일 16~23 %p)', '', M.round(2).to_markdown(index=False), '',
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
    rng = np.random.default_rng(int(time.time()) + (abs(hash(str(shard))) % 100000))
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
    light = {k: meta[k] for k in ('idx', 'sec', 'names', 'flow_cov', 'G') if k in meta}
    tmp = f'/tmp/out/store_{shard}.pkl.tmp'
    pickle.dump({'store': store, 'pk': {k: v[v].index.tolist() for k, v in pk.items()}, 'meta': light, 'signs': signs, 'sec': time.time() - t0}, open(tmp, 'wb'))
    os.replace(tmp, f'/tmp/out/store_{shard}.pkl')
    print('저장', shard, len(store), flush=True)


TRAIN_END = 2019


def update_weights(store):
    """조건별 효과를 학습 구간(2016~2019, 63일 결과)만으로 계산해 다음 회차 샘플링 확률에 반영(판정 구간 2020~에는 손대지 않음)."""
    rows = {}
    for k, v in store.items():
        if not k.startswith('교집합') or navg(v) < 2:
            continue
        y = v.loc['2016':f'{TRAIN_END}-09-30', 'ex63']
        if y.count() >= 24:
            rows[k] = y.mean()
    sr = pd.Series(rows) if rows else pd.Series(dtype=float); base = sr.mean() if len(sr) else 0; sd = sr.std() if len(sr) > 1 else 1
    w, eff = {}, {}
    for c in SEC_CONDS + STK_CONDS:
        x = sr[sr.index.str.contains(c, regex=False)]
        if len(x) >= 20:
            eff[c] = float((x.mean() - base) / (sd + 1e-9)); w[c] = float(np.exp(2 * eff[c]))
    # 2단계 로직의 섹터 판단 재료도 같은 방식(학습 구간만)으로 추림
    r2 = {k: v.loc['2016':f'{TRAIN_END}-09-30', 'ex63'].mean() for k, v in store.items() if k.startswith('2단계') and navg(v) >= 2 and v.loc['2016':f'{TRAIN_END}-09-30', 'ex63'].count() >= 24}
    if len(r2) >= 100:
        s2 = pd.Series(r2); b2 = s2.mean(); d2 = s2.std()
        for a in ST_AX:
            x = s2[s2.index.str.split('|').str[1].str.contains(a, regex=False)]
            if len(x) >= 20:
                eff[a] = float((x.mean() - b2) / (d2 + 1e-9)); w[a] = float(np.exp(2 * eff[a]))
    if not w:
        return
    json.dump({'train_end': TRAIN_END, 'n_methods': len(sr), 'weights': w, 'effect_sd': eff},
              open(os.path.join(ROOT, 'docs', 'sim', 'cond_weights.json'), 'w'), ensure_ascii=False, indent=1)


def merge(paths):
    store, pk, meta, signs, el = {}, {}, None, None, 0
    for p in sorted(paths, key=os.path.getmtime):
        try:
            d = pickle.load(open(p, 'rb'))
        except Exception:
            continue
        store.update(d['store']); meta = d['meta']; signs = d['signs']; el = max(el, d['sec'])
        for k, v in d['pk'].items():
            pk[k] = pd.Series(True, index=v)
    update_weights(store)
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
        dl = time.time() + 4.6 * 3600 if sys.argv[1] == 'auto' else pd.Timestamp(sys.argv[1], tz='UTC').timestamp()
        main(dl, sys.argv[2] if len(sys.argv) > 2 else None)
