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
    pos = (fr > 0).astype(int)
    X['frgn_streak'] = pos.apply(lambda c: c.groupby((c == 0).cumsum()).cumsum()).astype('float32').where(fr.notna())
    X['orgn60_cap'] = og.rolling(60, min_periods=45).sum() / cap
    X['fo_vs_prsn'] = ((fo20 > 0) & (X['orgn20_cap'].notna())).astype('float32').where(fo20.notna())
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


SEC_CONDS = ['섹:3개월추세', '섹:그날강함', '섹:꺾임아님', '섹:초입', '섹:가격상위5', '섹:수급가속', '섹:수급유입60', '섹:이익확산',
             '섹:1개월상위10', '섹:3개월상위10', '섹:60일선위종목60%+', '섹:신고가비율상위10', '섹:3개월하위아님',
             '섹:20일선위흐름20일+', '섹:20일선위흐름40일+', '섹:20일선위흐름60일+', '섹:대장주20일선위흐름20일+', '섹:대장주20일선위흐름40일+']
STK_CONDS = ['영업이익증가', '이익률개선', '신고가-10%내', '신고가-25%내', '저변동상위50%', '외기20일매수', '외기60일매수', '외기가속', '연기금60일매수',
             '외국인60일매수', '섹터시총3위내', '섹터거래대금3위내', '안달림', '섹터대비1개월덜오름', 'PBR낮은50%', 'ROE상위50%', '자사주90일', '외국계5%90일', '이익수익률상위50%',
             '20일선위', '60일선위', '120일선위', '정배열', '20일신고가', '60일선0~10%위', 'RSI50~65', 'RSI70미만', '1개월-10~+10%', '3개월상위30%', '저변동하위30%아님',
             '외국인연속5일', '기관60일매수', '외기매수개인매도', '매출증가', '매출가속', 'PER하위30%', 'PSR하위30%', '부채비율100%미만', '공급계약90일',
             '대형(시총200위내)', '중형(200~700위)', '거래대금증가1.2~2배', '섹터대비3개월덜오름', '12개월모멘텀상위30%',
             '20일선위흐름20일+', '20일선위흐름40일+', '20일선위흐름10일이하(초입)']
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
VAL = ['bp', 'ep', 'op_yield', 'roe', 'op_g', 'opm_chg', 'low_vol']


def random_state():
    ax = sorted(rng.choice(list(ST_AX), size=int(rng.integers(2, 6)), replace=False, p=_p(list(ST_AX))).tolist())
    K = int(rng.choice([2, 3, 5]))
    return {'mode': '2단계', 'axes': ax, 'K': K, 'Kout': int(K * rng.choice([1.5, 2, 3])), 'brk': bool(rng.random() < 0.6),
            'rsi_cap': bool(rng.random() < 0.5), 'rep': str(rng.choice(list(REP))), 'crowd': str(rng.choice(CROWD)),
            'm': int(rng.choice([1, 2])), 'lag': bool(rng.random() < 0.5), 'profit': bool(rng.random() < 0.6),
            'lead_free': bool(rng.random() < 0.5), 'mix_ex': bool(rng.random() < 0.5),
            'val': sorted(rng.choice(VAL, size=int(rng.integers(3, 8)), replace=False).tolist())}


def random_method():
    u = rng.random()
    if u < 0.5:
        return random_state()
    if u < 0.85:
        return random_inter()
    return _random_score()


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
    if m.get('mode') == '2단계':
        return evaluate_state(m, X, Y, S, meta)
    if m.get('mode') == '교집합':
        return evaluate_inter(m, X, Y, S, meta)
    return evaluate_score(m, X, Y, S, meta, signs)


def _random_score():
    m = {'gate': rng.choice(SEC_GATES), 'axes': sorted(rng.choice(list(SEC_AXES), size=rng.integers(1, 4), replace=False).tolist()),
         'K': int(rng.choice([2, 3, 5, 8, 99])), 'facs': sorted(rng.choice(POOL, size=rng.integers(2, 9), replace=False).tolist()),
         'N': int(rng.choice([5, 8, 12])), 'cap': int(rng.choice([1, 2, 3])), 'profit': bool(rng.random() < 0.6), 'noran': bool(rng.random() < 0.6),
         'sign': rng.choice(['걷기식', '걷기식+고정가점'])}
    return m


def mname(m):
    if m.get('mode') == '2단계':
        return (f"2단계|섹:{'+'.join(m['axes'])}|K{m['K']}~{m['Kout']}|{'꺾임교체' if m['brk'] else '-'}|{'과열감점' if m['rsi_cap'] else '-'}"
                f"|대장:{m['rep']}{'(관문없음)' if m.get('lead_free') else ''}|꽉참:{m['crowd']}|가치{m['m']}:{','.join(m['val'])}{'(덜오름)' if m['lag'] else ''}|{'이익' if m['profit'] else '-'}|{'혼합섹터제외' if m.get('mix_ex') else '-'}")
    if m.get('mode') == '교집합':
        return f"교집합|{'∩'.join(m['conds'])}|순서:{m['tie']}|N{m['N']}c{m['cap']}"
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
         '거래대금증가1.2~2배': lambda: (X['val_surge'] >= 1.2) & (X['val_surge'] <= 2), '섹터대비3개월덜오름': lambda: X['rs3'] < 0, '12개월모멘텀상위30%': lambda: pct('mom12_1') > 0.7}
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
        r = X[f].where(ok).rank(axis=1, pct=True) - 0.5
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
            '최악': pre['ex63'].min(), '21일 16~23': pre['ex21'].mean(), '섹터대비 63': pre['sx63'].mean(), '섹터 선택 63': pre['secx63'].mean(),
            '수급유입 63': pre['fx63'].mean(), '24~ 63일': post['ex63'].mean(), '안정성': R['stab'].mean(), '평균 종목 수': R['n'].mean()}


def walkforward(store, start=2017):
    """해마다 이전 해들의 연도 성과(63일 결과 확인 뒤)로 방법 고르기: 1등, 상위5 평균, 상위20 평균."""
    names = [k for k, v in store.items() if v['n'].mean() >= 2]     # 평균 2종목 미만은 우연이 커서 제외
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


def marginals(T):
    T = T[~T.index.str.startswith('교집합')]
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
         '## 2단계 상태형 로직 항목별 효과(63일 16~23 %p, 2단계 로직 평균 대비)', '', state_marg(T).round(2).to_markdown(index=False), '',
         '## 로직 계열별 평균', '', T.assign(계열=T.index.str.split('|').str[0].where(T.index.str.split('|').str[0].isin(['2단계', '교집합']), '점수형')).groupby('계열')[['63일 16~23', '플러스 해', '24~ 63일', '안정성']].mean().round(2).to_markdown(), '', '## 무엇이 효과가 있나(방법들의 항목별 평균, 63일 16~23 %p)', '', M.round(2).to_markdown(index=False), '',
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
    light = {k: meta[k] for k in ('idx', 'sec', 'names', 'flow_cov')}
    tmp = f'/tmp/out/store_{shard}.pkl.tmp'
    pickle.dump({'store': store, 'pk': {k: v[v].index.tolist() for k, v in pk.items()}, 'meta': light, 'signs': signs, 'sec': time.time() - t0}, open(tmp, 'wb'))
    os.replace(tmp, f'/tmp/out/store_{shard}.pkl')
    print('저장', shard, len(store), flush=True)


TRAIN_END = 2019


def update_weights(store):
    """조건별 효과를 학습 구간(2016~2019, 63일 결과)만으로 계산해 다음 회차 샘플링 확률에 반영(판정 구간 2020~에는 손대지 않음)."""
    rows = {}
    for k, v in store.items():
        if not k.startswith('교집합') or v['n'].mean() < 2:
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
    r2 = {k: v.loc['2016':f'{TRAIN_END}-09-30', 'ex63'].mean() for k, v in store.items() if k.startswith('2단계') and v['n'].mean() >= 2 and v.loc['2016':f'{TRAIN_END}-09-30', 'ex63'].count() >= 24}
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
