"""월별 과거 시점 추천 -> 3·6개월 뒤 성과 채점.

로직 설정(dict)을 받아, 매월 말 그날까지 알 수 있던 자료만으로 종목을 뽑고,
다음 거래일 종가에 사서 63·126거래일 뒤 종가(또는 손절)로 판다고 보고 성과를 매긴다.
"""
import numpy as np
import pandas as pd
from . import data as Dm

BASE = {
    'name': 'v4',
    'cap_rank': 700, 'val_rank': 1000,          # 유니버스: 시총 상위 700, 20일 거래대금 상위 1000
    'stages': ['주도', '초기'],
    'types': ['A', 'B'],
    'b_lag_floor': -15, 'b_ma60_floor': -5,
    'rsi_hot': 75,
    'gate_rights_days': 90, 'gate_loss': True,
    'need_earn': True, 'rev_yoy': 10,
    'n_max': {'위험선호': 5, '중립': 3, '위험회피': 2},
    'rank': 'ry',                                # 후보 순위: 최근 분기 매출 증가율
    'stop': True,                                # 최근 20일 최저 종가 이탈 시 정리
    'stop_pct': None,                            # 값이 있으면 진입가 대비 하락률 손절(예: 0.15)
    'cost': 0.005,                               # 왕복 비용(수수료·세금·충격) 0.5%
    'min_sector_n': 5,
}


class Ctx:
    def __init__(self):
        W = Dm.prices()
        self.W = W
        self.P = W['adj']
        self.close = W['close']
        self.dates = self.P.index
        self.kospi = Dm.index_series('idx_kospi', '코스피').reindex(self.dates).ffill()
        self.vk = Dm.index_series('idx_drv', '코스피 200 변동성지수').reindex(self.dates).ffill()
        sec = Dm.sectors()
        self.sec = sec.reindex(self.P.columns)
        r = W['ret'].fillna(0)
        capp = W['cap'].shift(1)
        idx = {}
        for s in self.sec.dropna().unique():
            cols = self.sec.index[self.sec == s]
            cw = capp[cols]
            n = cw.notna().sum(axis=1)
            ri = (cw * r[cols]).sum(axis=1) / cw.sum(axis=1)
            ri[n < BASE['min_sector_n']] = np.nan
            idx[s] = (1 + ri.fillna(0)).cumprod().where(n.cummax() >= BASE['min_sector_n'])
        self.secidx = pd.DataFrame(idx)
        ma60 = self.close.rolling(60, min_periods=60).mean()
        above = (self.close > ma60).where(ma60.notna())
        self.breadth = above.sum(axis=1) / above.notna().sum(axis=1).replace(0, np.nan) * 100
        d = self.P.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        self.rsi = 100 - 100 / (1 + up / dn)
        self.val20 = W['val'].rolling(20, min_periods=15).mean()
        self.low20 = self.P.rolling(20, min_periods=15).min()
        try:
            self.fin = Dm.fin_quarters()
        except Exception:
            self.fin = pd.DataFrame()
        try:
            self.ev = Dm.events()
        except Exception:
            self.ev = pd.DataFrame(columns=['dt', 'stock_code', 'kind'])
        me = pd.Series(self.dates, index=self.dates).groupby([self.dates.year, self.dates.month]).max()
        self.month_ends = list(me.values)


def regime(C, i):
    k = C.kospi.iloc[:i + 1].dropna()
    if len(k) < 200:
        return '중립'
    score = 1 if k.iloc[-1] > k.iloc[-200:].mean() else -1
    b = C.breadth.iloc[i]
    score += 1 if b > 50 else (-1 if b < 30 else 0)
    v = C.vk.iloc[:i + 1].dropna()
    if len(v) > 51:
        score += -1 if v.iloc[-1] > v.iloc[-51:-1].mean() * 1.2 else (1 if v.iloc[-1] < v.iloc[-51:-1].mean() * 0.9 else 0)
    return '위험선호' if score >= 2 else ('위험회피' if score <= -2 else '중립')


def sector_stage(C, i):
    k = C.kospi.iloc[:i + 1].dropna()
    out = {}
    if len(k) < 130:
        return out
    kr = lambda n: (k.iloc[-1] / k.iloc[-n - 1] - 1) * 100
    for s in C.secidx.columns:
        x = C.secidx[s].iloc[:i + 1].dropna()
        if len(x) < 130:
            continue
        r = lambda n: (x.iloc[-1] / x.iloc[-n - 1] - 1) * 100
        rs1, rs3, rs6 = r(21) - kr(21), r(63) - kr(63), r(126) - kr(126)
        g60 = (x.iloc[-1] / x.iloc[-60:].mean() - 1) * 100
        g120 = (x.iloc[-1] / x.iloc[-120:].mean() - 1) * 100
        if g60 < 0 and rs3 < 0:
            st = '쇠퇴'
        elif g60 > 20 or r(21) > 25:
            st = '과열 주의'
        elif rs3 > 0 and rs6 > 0 and g120 > 0:
            st = '주도'
        elif rs1 > 0 and rs3 > -5 and g60 > 0:
            st = '초기'
        else:
            st = '중립'
        out[s] = {'stage': st, 'm1': r(21), 'm3': r(63)}
    return out


def fin_at(C, t):
    """t 시점에 공시돼 있던 최근 두 분기."""
    f = C.fin
    if f.empty:
        return {}
    g = f[f['avail'] <= t].groupby('stock_code').tail(2)
    out = {}
    for sc, h in g.groupby('stock_code'):
        h = h.sort_values('ord')
        a = h.iloc[-1]
        if (t - a['avail']).days > 200:
            continue          # 너무 오래된 실적은 쓰지 않음
        yoy = lambda x: (x['rev'] / x['rev_py'] - 1) * 100 if pd.notna(x['rev']) and pd.notna(x['rev_py']) and x['rev_py'] > 0 else np.nan
        prev = h.iloc[-2] if len(h) > 1 and h.iloc[-2]['ord'] == a['ord'] - 1 else None
        out[sc] = {'ry': yoy(a), 'ryp': yoy(prev) if prev is not None else np.nan, 'op': a['op'], 'opp': a['op_py']}
    return out


def pick(C, i, L):
    t = C.dates[i]
    cap = C.W['cap'].iloc[i]
    ok = cap.notna() & C.close.iloc[i].notna()
    caprank = cap[ok].rank(ascending=False)
    valrank = C.val20.iloc[i][ok].rank(ascending=False)
    uni = caprank.index[(caprank <= L['cap_rank']) & (valrank.reindex(caprank.index) <= L['val_rank'])]
    reg = regime(C, i)
    st = sector_stage(C, i)
    P = C.P.iloc[:i + 1]
    r1 = (P.iloc[-1] / P.iloc[-22] - 1) * 100
    r3 = (P.iloc[-1] / P.iloc[-64] - 1) * 100
    g60 = (P.iloc[-1] / P.iloc[-60:].mean() - 1) * 100
    fin = fin_at(C, t) if L['need_earn'] or L['gate_loss'] else {}
    ev = C.ev[(C.ev['dt'] <= t) & (C.ev['dt'] > t - pd.Timedelta(days=L['gate_rights_days'])) & (C.ev['kind'] == 'rights')]
    rights = set(ev['stock_code'])
    rows = []
    rsi_row = C.rsi.iloc[i]
    for c in uni:
        s = C.sec.get(c)
        if s not in st or st[s]['stage'] not in L['stages']:
            continue
        S = st[s]
        typ = None
        if S['stage'] == '초기' and r1[c] > S['m1'] and r3[c] > S['m3']:
            typ = 'A'
        elif S['stage'] == '주도' and r1[c] < S['m1'] and r3[c] - S['m3'] > L['b_lag_floor'] and g60[c] > L['b_ma60_floor']:
            typ = 'B'
        if L['types'] and typ not in L['types']:
            continue
        if not L['types']:
            typ = typ or '-'
        if L['rsi_hot'] and rsi_row[c] >= L['rsi_hot']:
            continue
        if c in rights and L['gate_rights_days']:
            continue
        f = fin.get(c, {})
        if L['gate_loss'] and pd.notna(f.get('op')) and f['op'] < 0 and pd.notna(f.get('ry')) and f['ry'] < 0:
            continue
        earn = pd.notna(f.get('ry')) and f['ry'] >= L['rev_yoy'] and pd.notna(f.get('ryp')) and f['ryp'] > 0 \
            and pd.notna(f.get('op')) and f['op'] > 0 and (pd.isna(f.get('opp')) or f['opp'] <= 0 or f['op'] >= f['opp'])
        if L['need_earn'] and not earn:
            continue
        rows.append({'code': c, 'sector': s, 'type': typ, 'ry': f.get('ry', np.nan), 'rs3': r3[c] - S['m3']})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, reg
    key = L['rank']
    df = df.sort_values(key, ascending=False, na_position='last')
    n = L['n_max'][reg] if L['n_max'] else len(df)
    return df.head(n), reg


def ubase(C, i, h, L):
    """같은 날 유니버스(시총·거래대금 조건) 전 종목을 같은 비중으로 샀을 때의 수익률(비교 기준)."""
    key = (i, h, L['cap_rank'], L['val_rank'])
    if not hasattr(C, '_ub'):
        C._ub = {}
    if key not in C._ub:
        cap = C.W['cap'].iloc[i]
        ok = cap.notna() & C.close.iloc[i].notna()
        cr = cap[ok].rank(ascending=False)
        vr = C.val20.iloc[i][ok].rank(ascending=False)
        uni = cr.index[(cr <= L['cap_rank']) & (vr.reindex(cr.index) <= L['val_rank'])]
        a, b = i + 1, i + 1 + h
        r = C.P[uni].iloc[b] / C.P[uni].iloc[a] - 1
        C._ub[key] = r.mean() - L['cost']
    return C._ub[key]


def score(C, i, picks, L, h):
    """i일 신호 -> i+1 종가 매수 -> i+1+h 종가 매도(손절 시 그 날 종가)."""
    a, b = i + 1, i + 1 + h
    if b >= len(C.dates):
        return None
    out = []
    kret = C.kospi.iloc[b] / C.kospi.iloc[a] - 1
    for _, p in picks.iterrows():
        c = p['code']
        path = C.P[c].iloc[a:b + 1]
        e = path.iloc[0]
        if pd.isna(e):
            continue
        x = path.iloc[-1]
        stopped = False
        if L['stop']:
            lv = e * (1 - L['stop_pct']) if L.get('stop_pct') else C.low20[c].iloc[i]
            hit = path[path < lv]
            if len(hit):
                x = hit.iloc[0]; stopped = True
        ret = x / e - 1 - L['cost']
        sret = C.secidx[p['sector']].iloc[b] / C.secidx[p['sector']].iloc[a] - 1 if p['sector'] in C.secidx else np.nan
        out.append({'date': C.dates[i], 'code': c, 'sector': p['sector'], 'type': p['type'], 'ret': ret,
                    'ex_kospi': ret - kret, 'ex_sector': ret - sret, 'ex_univ': ret - ub, 'stopped': stopped})
    return out


def run(C, L, start='2011-01-01', end=None, horizons=(63, 126)):
    res = {h: [] for h in horizons}
    regs = []
    for t in C.month_ends:
        if t < pd.Timestamp(start) or (end and t > pd.Timestamp(end)):
            continue
        i = C.dates.get_loc(t)
        if i < 130:
            continue
        pk, reg = pick(C, i, L)
        regs.append({'date': t, 'regime': reg, 'n': len(pk)})
        for h in horizons:
            s = score(C, i, pk, L, h) if len(pk) else []
            if s:
                res[h] += s
    return {h: pd.DataFrame(v) for h, v in res.items()}, pd.DataFrame(regs)


def summary(res, regs, label):
    rows = []
    for h, df in res.items():
        if df.empty:
            rows.append({'logic': label, 'h': h, 'n': 0}); continue
        for per, (a, b) in {'전체': ('2000', '2100'), '2011-18': ('2011', '2019'), '2019-23': ('2019', '2024'), '2024-': ('2024', '2100')}.items():
            d = df[(df['date'] >= a) & (df['date'] < b)]
            if d.empty:
                continue
            rows.append({'logic': label, 'h': '3개월' if h == 63 else '6개월', '구간': per, '추천 수': len(d),
                         '평균 수익률': d['ret'].mean() * 100, '코스피 대비': d['ex_kospi'].mean() * 100,
                         '섹터 대비': d['ex_sector'].mean() * 100, '시장 평균 대비': d['ex_univ'].mean() * 100,
                         '시장 평균 이긴 비율': (d['ex_univ'] > 0).mean() * 100, '코스피 이긴 비율': (d['ex_kospi'] > 0).mean() * 100,
                         '수익 비율': (d['ret'] > 0).mean() * 100, '하위 10%': d['ret'].quantile(0.1) * 100,
                         '손절 비율': d['stopped'].mean() * 100})
    return pd.DataFrame(rows)
