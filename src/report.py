"""일별 리포트 생성 (예시판). 결과: reports/YYYYMMDD.html, .pdf

사용: python3 src/report.py [기준일 YYYYMMDD]
필요 환경변수: DART_KEY
"""
import base64, glob, gzip, io, json, os, re, sys, time, urllib.request, zipfile
from datetime import date, datetime, timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'reports')
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK-Regular' in f:
        font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'
UP, DOWN, ACC = '#c0392b', '#1f5fbf', '#2f6f4f'   # 한국식: 상승 빨강, 하락 파랑


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '').replace({'': np.nan, '-': np.nan}), errors='coerce')


def load(name, cols=None):
    fs = sorted(glob.glob(os.path.join(DATA, name, '20*', '*.csv.gz')))
    dfs = []
    for f in fs:
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def fmt_pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    s = f'{x:+.{d}f}%' if sign else f'{x:.{d}f}%'
    return s


def color_pct(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '<span class="mut">–</span>'
    c = 'up' if x > 0 else ('dn' if x < 0 else '')
    return f'<span class="{c}">{x:+.{d}f}%</span>'


def fig_b64(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=180, bbox_inches='tight'); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def style_ax(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis='y', color=GRID, linewidth=0.6)


# ---------------------------------------------------------------- 데이터 준비
def stock_panel():
    cols = ['BAS_DD', 'ISU_CD', 'ISU_NM', 'MKT_NM', 'TDD_CLSPRC', 'FLUC_RT', 'TDD_HGPRC', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS']
    df = pd.concat([load('stk', cols), load('ksq', cols)], ignore_index=True)
    for c in ['TDD_CLSPRC', 'FLUC_RT', 'TDD_HGPRC', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS']:
        df[c] = num(df[c])
    df['BAS_DD'] = pd.to_datetime(df['BAS_DD'])
    return df


def series_from(name, key_col, key, val_col, date_col='BAS_DD', extra=None):
    df = load(name)
    if df.empty:
        return pd.Series(dtype=float)
    d = df[df[key_col].str.strip() == key]
    if extra:
        for k, v in extra.items():
            d = d[d[k] == v]
    s = pd.Series(num(d[val_col]).values, index=pd.to_datetime(d[date_col])).sort_index()
    return s[~s.index.duplicated()].dropna()


def fut_front(prod, asof=None):
    """정규시장에서 기준일 거래량이 가장 많은 월물(스프레드 제외)을 고르고, 그 월물 하나의 종가 시계열을 돌려준다.
    월물을 섞지 않아 만기 교체로 인한 가격 점프가 없다."""
    df = load('fut', ['BAS_DD', 'PROD_NM', 'MKT_NM', 'ISU_NM', 'TDD_CLSPRC', 'ACC_TRDVOL'])
    if df.empty:
        return pd.Series(dtype=float)
    d = df[(df['PROD_NM'] == prod) & (df['MKT_NM'] == '정규') & ~df['ISU_NM'].str.contains(' SP ')].copy()
    d['v'] = num(d['ACC_TRDVOL']); d['c'] = num(d['TDD_CLSPRC'])
    d = d.dropna(subset=['c'])
    if asof is not None:
        d = d[pd.to_datetime(d['BAS_DD']) <= asof]
    last = d[d['BAS_DD'] == d['BAS_DD'].max()].sort_values('v')
    if last.empty:
        return pd.Series(dtype=float)
    nm = last['ISU_NM'].iloc[-1]
    d = d[d['ISU_NM'] == nm]
    return pd.Series(d['c'].values, index=pd.to_datetime(d['BAS_DD'])).sort_index()


# ---------------------------------------------------------------- 1. 시장 국면
def market_regime(px, asof):
    kospi = series_from('idx_kospi', 'IDX_NM', '코스피', 'CLSPRC_IDX')
    kosdaq = series_from('idx_kosdaq', 'IDX_NM', '코스닥', 'CLSPRC_IDX')
    vk = series_from('idx_drv', 'IDX_NM', '코스피 200 변동성지수', 'CLSPRC_IDX')
    out = {'kospi': kospi, 'kosdaq': kosdaq}
    rows = []
    for nm, s in [('코스피', kospi), ('코스닥', kosdaq)]:
        s = s[s.index <= asof]
        last = s.iloc[-1]
        rows.append({'name': nm, 'close': last, 'd1': (last / s.iloc[-2] - 1) * 100, 'm1': (last / s.iloc[-22] - 1) * 100 if len(s) > 22 else np.nan,
                     'ma200': (last / s.rolling(200).mean().iloc[-1] - 1) * 100 if len(s) >= 200 else np.nan,
                     'ma60': (last / s.rolling(60).mean().iloc[-1] - 1) * 100})
    out['idx_rows'] = rows
    # 시장 폭(우선주·스팩 제외 근사: 이름 끝 '우'/'우B', '스팩' 포함 제외)
    p = px[~px['ISU_NM'].str.contains('스팩') & ~px['ISU_NM'].str.match(r'.*우[B-C]?$')]
    wide = p.pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    wide = wide[wide.index <= asof]
    ma60 = wide.rolling(60, min_periods=60).mean()
    above = (wide > ma60).where(ma60.notna())
    breadth = above.sum(axis=1).astype(float) / above.notna().sum(axis=1).replace(0, np.nan).astype(float) * 100
    hi = wide.rolling(250, min_periods=240).max(); lo = wide.rolling(250, min_periods=240).min()
    nh = (wide >= hi).sum(axis=1); nl = (wide <= lo).where(lo.notna()).sum(axis=1)
    out['breadth'] = breadth.dropna()
    out['nh'], out['nl'] = int(nh.iloc[-1]), int(nl.iloc[-1])
    out['nh_ok'] = hi.iloc[-1].notna().sum() > 500
    out['vk'] = vk[vk.index <= asof]
    # 레버리지·인버스 좌수
    etf = load('etf', ['BAS_DD', 'ISU_CD', 'ISU_NM', 'LIST_SHRS'])
    lev = {}
    for code, nm in [('122630', 'KODEX 레버리지'), ('252670', 'KODEX 200선물인버스2X'), ('233740', 'KODEX 코스닥150레버리지'), ('251340', 'KODEX 코스닥150선물인버스')]:
        d = etf[etf['ISU_CD'] == code]
        s = pd.Series(num(d['LIST_SHRS']).values, index=pd.to_datetime(d['BAS_DD'])).sort_index()
        s = s[s.index <= asof]
        if len(s) > 21:
            lev[nm] = (s.iloc[-1] / s.iloc[-21] - 1) * 100
    out['lev'] = lev
    # 판정(출발값)
    k = rows[0]
    b = breadth.dropna().iloc[-1]
    score = 0
    score += 1 if (not np.isnan(k['ma200']) and k['ma200'] > 0) else -1
    score += 1 if b > 50 else (-1 if b < 30 else 0)
    vks = out['vk']
    vk_last = vks.iloc[-1]; vk_avg = vks.iloc[-51:-1].mean()
    score += -1 if vk_last > vk_avg * 1.2 else (1 if vk_last < vk_avg * 0.9 else 0)
    out['vk_last'], out['vk_avg'] = vk_last, vk_avg
    out['regime'] = '위험선호' if score >= 2 else ('위험회피' if score <= -2 else '중립')
    out['breadth_last'] = b
    return out


# ---------------------------------------------------------------- 금리·환율·해외
def macro(asof):
    items = []
    for prod, label, note in [('3개월무위험금리 선물', '3개월 무위험금리 선물', '가격↑ = 단기금리 기대↓'),
                              ('3년국채 선물', '3년 국채선물', '가격↑ = 금리↓'),
                              ('10년국채 선물', '10년 국채선물', '가격↑ = 금리↓'),
                              ('미국달러 선물', '미국달러 선물(원/달러)', '가격↑ = 원화 약세'),
                              ('엔 선물', '엔 선물(원/100엔)', '')]:
        s = fut_front(prod, asof)
        if len(s) < 22:
            continue
        items.append({'label': label, 'last': s.iloc[-1], 'd1': (s.iloc[-1] / s.iloc[-2] - 1) * 100,
                      'm1': (s.iloc[-1] / s.iloc[-21] - 1) * 100, 'note': note})
    etf = load('etf', ['BAS_DD', 'ISU_CD', 'ISU_NM', 'FLUC_RT_IDX', 'IDX_IND_NM'])
    glob_rows = []
    for code, label in [('360750', 'S&P 500'), ('133690', '나스닥 100'), ('381180', '필라델피아 반도체'),
                        ('261220', 'WTI 원유(선물)'), ('132030', '금(선물)'), ('160580', '구리(실물)')]:
        d = etf[etf['ISU_CD'] == code]
        if d.empty:
            continue
        r = pd.Series(num(d['FLUC_RT_IDX']).values, index=pd.to_datetime(d['BAS_DD'])).sort_index()
        r = r[r.index <= asof].dropna()
        if len(r) < 21:
            continue
        glob_rows.append({'label': label, 'd1': r.iloc[-1], 'm1': (np.prod(1 + r.iloc[-20:] / 100) - 1) * 100,
                          'src': d['ISU_NM'].iloc[-1]})
    return items, glob_rows


# ---------------------------------------------------------------- 2. 섹터 단계
SECTOR_IDX = ['KRX 반도체', 'KRX 자동차', 'KRX 헬스케어', 'KRX 은행', 'KRX 증권', 'KRX 보험', 'KRX 에너지화학', 'KRX 철강',
              'KRX 건설', 'KRX 기계장비', 'KRX 운송', 'KRX 방송통신', 'KRX 경기소비재', 'KRX 필수소비재', 'KRX K콘텐츠', 'KRX 유틸리티']
THEME_ETF = [('조선', '조선'), ('방산', '방산'), ('2차전지', '2차전지'), ('전력기기·원전', '전력|원자력|원전'),
             ('반도체 소부장', '소부장|반도체장비|반도체소재')]


def sector_table(asof, kospi):
    rows = []
    ser = {}
    for nm in SECTOR_IDX:
        s = series_from('idx_krx', 'IDX_NM', nm, 'CLSPRC_IDX')
        ser[nm.replace('KRX ', '')] = (s, 'KRX 지수')
    etf = load('etf', ['BAS_DD', 'ISU_CD', 'ISU_NM', 'TDD_CLSPRC', 'NAV', 'INVSTASST_NETASST_TOTAMT', 'LIST_SHRS'])
    last_day = etf['BAS_DD'].max()
    for label, pat in THEME_ETF:
        cand = etf[(etf['BAS_DD'] == last_day) & etf['ISU_NM'].str.contains(pat) & ~etf['ISU_NM'].str.contains('레버리지|인버스|채권|커버드|미국|글로벌|중국|일본|유럽')]
        if cand.empty:
            continue
        code = cand.assign(a=num(cand['INVSTASST_NETASST_TOTAMT'])).sort_values('a').iloc[-1]
        d = etf[etf['ISU_CD'] == code['ISU_CD']]
        s = pd.Series(num(d['NAV']).values, index=pd.to_datetime(d['BAS_DD'])).sort_index()
        u = pd.Series(num(d['LIST_SHRS']).values, index=pd.to_datetime(d['BAS_DD'])).sort_index()
        ser[label] = (s, code['ISU_NM'], u)
    k = kospi[kospi.index <= asof]
    for nm, v in ser.items():
        s = v[0][v[0].index <= asof].dropna()
        if len(s) < 130:
            continue
        r = lambda n: (s.iloc[-1] / s.iloc[-n - 1] - 1) * 100
        kr = lambda n: (k.iloc[-1] / k.iloc[-n - 1] - 1) * 100
        rs1, rs3, rs6 = r(21) - kr(21), r(63) - kr(63), (r(126) - kr(126)) if len(s) > 127 else np.nan
        g60 = (s.iloc[-1] / s.rolling(60).mean().iloc[-1] - 1) * 100
        g120 = (s.iloc[-1] / s.rolling(120).mean().iloc[-1] - 1) * 100
        units = None
        if len(v) > 2:
            u = v[2][v[2].index <= asof]
            units = (u.iloc[-1] / u.iloc[-21] - 1) * 100 if len(u) > 21 else None
        # 단계(가격 축만, 출발값)
        if g60 < 0 and rs3 < 0:
            st = '쇠퇴'
        elif g60 > 20 or r(21) > 25:
            st = '과열 주의'
        elif rs3 > 0 and (np.isnan(rs6) or rs6 > 0) and g120 > 0:
            st = '주도'
        elif rs1 > 0 and rs3 > -5 and g60 > 0:
            st = '초기'
        else:
            st = '중립'
        rows.append({'name': nm, 'src': v[1], 'm1': r(21), 'm3': r(63), 'm6': r(126) if len(s) > 127 else np.nan,
                     'rs1': rs1, 'rs3': rs3, 'rs6': rs6, 'g60': g60, 'units': units, 'stage': st})
    order = {'주도': 0, '초기': 1, '과열 주의': 2, '중립': 3, '쇠퇴': 4}
    return sorted(rows, key=lambda x: (order[x['stage']], -x['rs3']))


def sector_members(px, asof, sec):
    """섹터 ETF 구성종목(한투 API)으로 섹터별 대표 종목과, 섹터보다 1개월 덜 오른 종목을 뽑는다."""
    fs = sorted(glob.glob(os.path.join(DATA, 'kis', 'holdings', '*.csv')))
    if not fs:
        return {}, None
    hd = pd.read_csv(fs[-1], dtype=str)
    hd['w'] = num(hd['weight'])
    p = px[px['BAS_DD'] <= asof]
    days = sorted(p['BAS_DD'].unique())
    if len(days) < 22:
        return {}, None
    a, b = days[-1], days[-22]
    c1 = p[p['BAS_DD'] == a].set_index('ISU_CD')
    c0 = p[p['BAS_DD'] == b].set_index('ISU_CD')
    ret = (c1['TDD_CLSPRC'] / c0['TDD_CLSPRC'].reindex(c1.index) - 1) * 100
    # 1개월 안에 상장주식수가 3% 넘게 바뀐 종목(분할·병합·증자 등)은 시세가 수정주가가 아니라 수익률을 쓰지 않는다(소각 등 소폭 변동은 유지)
    chg = (c1['LIST_SHRS'] / c0['LIST_SHRS'].reindex(c1.index) - 1).abs() > 0.03
    ret[chg] = np.nan
    m1 = {s_['name']: s_['m1'] for s_ in sec}
    out = {}
    for name, g in hd.groupby('sector'):
        g = g[g['code'].isin(c1.index)].sort_values('w', ascending=False)
        if g.empty:
            continue
        g = g.assign(r=g['code'].map(ret), cap=g['code'].map(c1['MKTCAP']) / 1e8)
        top = [(x['name'], x['w'], x['r']) for _, x in g.head(3).iterrows()]
        base = m1.get(name, np.nan)
        lag = g.iloc[3:15]
        lag = lag[(lag['cap'] >= 5000) & lag['r'].notna() & (lag['r'] < base)] if not np.isnan(base) else lag.iloc[0:0]
        lag = [(x['name'], x['w'], x['r']) for _, x in lag.head(3).iterrows()]
        out[name] = {'etf': g['etf'].iloc[0], 'top': top, 'lag': lag, 'n': len(g)}
    return out, os.path.basename(fs[-1])[:8]


# ---------------------------------------------------------------- 3. 수출
def exports():
    p = os.path.join(DATA, 'customs', 'export10d.csv')
    df = pd.read_csv(p, dtype=str)
    items = ['전체', '반도체', '컴퓨터주변기기', '승용차', '자동차부품', '선박', '석유제품', '철강제품', '무선통신기기']
    for c in items:
        df[c] = pd.to_numeric(df[c].str.strip().str.replace(',', ''), errors='coerce') / 1e3  # 백만 달러
    df['end'] = df['period'].str[3:5].astype(int)
    # 개장일(거래소 기준) 집합: 조업일수 근사
    days = sorted({os.path.basename(f)[:8] for f in glob.glob(os.path.join(DATA, 'stk', '20*', '*.csv.gz'))})
    days = pd.to_datetime(pd.Series(days))
    def wdays(ym, end):
        y, m = int(ym[:4]), int(ym[4:])
        sel = days[(days.dt.year == y) & (days.dt.month == m) & (days.dt.day <= end)]
        return len(sel) if len(sel) else np.nan
    have_cal = set(days.dt.strftime('%Y%m'))
    latest = df.iloc[-1]
    ym, per, end = latest['yymm'], latest['period'], latest['end']
    py = f'{int(ym[:4]) - 1}{ym[4:]}'
    prev_y = df[(df['yymm'] == py) & (df['period'].str[:2] == '01')]
    prev_y = prev_y[prev_y['end'] == end] if end <= 20 else prev_y[prev_y['end'] > 20]
    pm = f'{int(ym[:4]) - (1 if ym[4:] == "01" else 0)}{12 if ym[4:] == "01" else int(ym[4:]) - 1:02d}'
    prev_m = df[(df['yymm'] == pm)]
    prev_m = prev_m[prev_m['end'] == end] if end <= 20 else prev_m[prev_m['end'] > 20]
    # 월 확정치(월말 행)로 연초 누계·전분기
    month = df[df['end'] > 20].copy()
    month['y'] = month['yymm'].str[:4]; month['m'] = month['yymm'].str[4:].astype(int)
    lastfull = month.iloc[-1]
    y, m = lastfull['y'], lastfull['m']
    ytd = month[(month['y'] == y) & (month['m'] <= m)][items].sum()
    ytd_p = month[(month['y'] == str(int(y) - 1)) & (month['m'] <= m)][items].sum()
    q = (m - 1) // 3
    qm = [3 * q + 1, 3 * q + 2, 3 * q + 3]
    cur_q = month[(month['y'] == y) & (month['m'].isin(qm))]
    res = {'ym': ym, 'per': per, 'last_full': f"{y}.{m:02d}", 'items': items, 'rows': [], 'q_note': ''}
    wd_cur = wdays(ym, end) if ym in have_cal else np.nan
    wd_py = wdays(py, end if end <= 20 else 31) if py in have_cal else np.nan
    wd_pm = wdays(pm, end if end <= 20 else 31) if pm in have_cal else np.nan
    res['wd'] = (wd_cur, wd_py, wd_pm)
    for it in items:
        cur = latest[it]
        yoy = (cur / prev_y[it].iloc[0] - 1) * 100 if len(prev_y) else np.nan
        mom = (cur / prev_m[it].iloc[0] - 1) * 100 if len(prev_m) else np.nan
        dyoy = ((cur / wd_cur) / (prev_y[it].iloc[0] / wd_py) - 1) * 100 if len(prev_y) and wd_cur and wd_py and not np.isnan(wd_py) else np.nan
        dmom = ((cur / wd_cur) / (prev_m[it].iloc[0] / wd_pm) - 1) * 100 if len(prev_m) and wd_cur and wd_pm and not np.isnan(wd_pm) else np.nan
        res['rows'].append({'item': it, 'cur': cur, 'yoy': yoy, 'dyoy': dyoy, 'mom': mom, 'dmom': dmom,
                            'ytd': ytd[it], 'ytd_yoy': (ytd[it] / ytd_p[it] - 1) * 100 if ytd_p[it] else np.nan})
    # 전분기 대비(직전 완결 분기 vs 그 전 분기, 월 확정치)
    month['q'] = (month['m'] - 1) // 3
    qs = month.groupby(['y', 'q'])[items].agg(['sum', 'count'])
    full = [k for k in qs.index if qs.loc[k, ('전체', 'count')] == 3]
    if len(full) >= 2:
        a, b = full[-1], full[-2]
        res['q_note'] = f"{a[0]}년 {a[1] + 1}분기"
        for r in res['rows']:
            r['qoq'] = (qs.loc[a, (r['item'], 'sum')] / qs.loc[b, (r['item'], 'sum')] - 1) * 100
    # 반도체 추이(그래프용): 월 확정치 전년비
    mm = month.set_index(pd.to_datetime(month['yymm'] + '01'))
    semi = mm['반도체']; tot = mm['전체']
    res['semi_yoy'] = (semi / semi.shift(12) - 1).dropna() * 100
    res['tot_yoy'] = (tot / tot.shift(12) - 1).dropna() * 100
    return res


def export_items():
    """관세청 품목별 월 수출(화장품·라면 등). 하루 평균은 거래소 개장일 수로 나눈 근사."""
    p = os.path.join(DATA, 'customs', 'items_monthly.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype={'yymm': str})
    pv = d.pivot(index='yymm', columns='item', values='exp_usd') / 1e6   # 백만 달러
    days = sorted({os.path.basename(f)[:8] for f in glob.glob(os.path.join(DATA, 'stk', '20*', '*.csv.gz'))})
    wd = pd.Series(days).str[:6].value_counts()
    first_cal = min(days)[:6]
    wdm = lambda ym: wd.get(ym, np.nan) if ym >= first_cal else np.nan
    ym = pv.index.max()
    y, m = int(ym[:4]), int(ym[4:])
    py = f'{y - 1}{m:02d}'
    pm = f'{y - (m == 1)}{12 if m == 1 else m - 1:02d}'
    qe = [f'{y}{x:02d}' for x in range(1, 13)]
    # 직전 완결 분기와 그 전 분기
    q = (m // 3) - 1 if m % 3 else (m // 3) - 1
    qn = m // 3  # 완결 분기 수(올해)
    def qmonths(yy, qq):
        return [f'{yy}{3 * qq + k:02d}' for k in (1, 2, 3)]
    lq = (y, qn - 1) if qn >= 1 else (y - 1, 3)
    pq = (lq[0], lq[1] - 1) if lq[1] >= 1 else (lq[0] - 1, 3)
    ytd = [f'{y}{k:02d}' for k in range(1, m + 1)]
    ytd_p = [f'{y - 1}{k:02d}' for k in range(1, m + 1)]
    wsum = lambda ms: sum(wdm(x) for x in ms)
    rows = []
    for it in pv.columns:
        v = pv[it]
        g = lambda k: v.get(k, np.nan)
        cur = g(ym)
        dy = (cur / wdm(ym)) / (g(py) / wdm(py)) * 100 - 100
        dm = (cur / wdm(ym)) / (g(pm) / wdm(pm)) * 100 - 100
        qa, qb = sum(g(x) for x in qmonths(*lq)), sum(g(x) for x in qmonths(*pq))
        ya, yb = sum(g(x) for x in ytd), sum(g(x) for x in ytd_p)
        rows.append({'item': it, 'cur': cur, 'yoy': (cur / g(py) - 1) * 100, 'dyoy': dy, 'mom': (cur / g(pm) - 1) * 100, 'dmom': dm,
                     'qoq': (qa / qb - 1) * 100, 'ytd': ya, 'ytd_yoy': (ya / yb - 1) * 100,
                     'dytd': (ya / wsum(ytd)) / (yb / wsum(ytd_p)) * 100 - 100,
                     'trend': [(v.get(f'{y}{k:02d}', np.nan) / v.get(f'{y - 1}{k:02d}', np.nan) - 1) * 100 for k in range(1, m + 1)]})
    return {'ym': ym, 'rows': rows, 'q': f'{lq[0]}년 {lq[1] + 1}분기', 'wd': (wdm(ym), wdm(py), wdm(pm))}


# ---------------------------------------------------------------- 4. 공시 이벤트
def dart_events(asof, days_back=5):
    key = os.environ['DART_KEY']
    end = asof
    start = asof - timedelta(days=days_back + 2)
    def lst(ty):
        out, p = [], 1
        while True:
            u = f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&bgn_de={start:%Y%m%d}&end_de={end:%Y%m%d}&pblntf_ty={ty}&page_no={p}&page_count=100"
            for i in range(3):
                try:
                    d = json.loads(urllib.request.urlopen(u, timeout=60).read()); break
                except Exception:
                    time.sleep(3)
            out += d.get('list', [])
            if p >= d.get('total_page', 1):
                break
            p += 1
        return out
    B, I = lst('B'), lst('I')
    allr = B + I
    pick = lambda kw: [x for x in allr if kw in x['report_nm'] and x['corp_cls'] in ('Y', 'K')]
    burn = pick('주식소각결정')
    supply = [x for x in pick('단일판매') if '해지' not in x['report_nm']]
    prelim = [x for x in allr if re.search(r'영업\(잠정\)실적|연결재무제표기준영업\(잠정\)실적', x['report_nm']) and x['corp_cls'] in ('Y', 'K')]
    rights = [x for x in allr if re.search(r'유상증자결정|전환사채권발행결정|신주인수권부사채권발행결정', x['report_nm']) and x['corp_cls'] in ('Y', 'K')]
    # 공급계약: 매출액 대비 비율 파싱
    sup_rows = []
    for x in supply[:60]:
        try:
            b = urllib.request.urlopen(f"https://opendart.fss.or.kr/api/document.xml?crtfc_key={key}&rcept_no={x['rcept_no']}", timeout=60).read()
            if b[:2] != b'PK':
                continue
            z = zipfile.ZipFile(io.BytesIO(b)); t = z.read(z.namelist()[0]).decode('utf-8', 'ignore')
            s = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', t))
            # 정정공시는 앞부분에 정정 전·후 표가 있으므로 본문 서식의 마지막 '계약금액-최근매출액' 쌍을 쓴다
            ms = re.findall(r'계약금액\s*(?:총액)?\s*\(원\)\s*([\d,]+)\s*최근\s*매출액\s*\(원\)\s*([\d,]+)', s)
            if not ms:
                continue
            amt, rev = (float(v.replace(',', '')) for v in ms[-1])
            sup_rows.append({'corp': x['corp_name'], 'dt': x['rcept_dt'], 'ratio': amt / rev * 100 if rev else np.nan,
                             'amt': amt / 1e8, 'nm': x['report_nm'].strip()})
            time.sleep(0.2)
        except Exception:
            continue
    seen, uniq = set(), []
    for r in sorted(sup_rows, key=lambda r: r['dt'], reverse=True):   # 같은 회사·같은 금액 중복(정정) 제거
        k = (r['corp'], round(r['amt']))
        if k not in seen:
            seen.add(k); uniq.append(r)
    sup_rows = sorted(uniq, key=lambda r: -(r['ratio'] if not np.isnan(r['ratio']) else -1))
    return {'start': start, 'end': end, 'burn': burn, 'supply': sup_rows, 'n_supply': len(supply),
            'prelim': prelim, 'rights': rights}


# ---------------------------------------------------------------- 5. 가격 신호 종목(관문 미적용 예시)
def breakout(px, asof):
    p = px[px['BAS_DD'] <= asof]
    p = p[~p['ISU_NM'].str.contains('스팩') & ~p['ISU_NM'].str.match(r'.*우[B-C]?$')]
    close = p.pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    val = p.pivot_table(index='BAS_DD', columns='ISU_CD', values='ACC_TRDVAL')
    hi20 = close.shift(1).rolling(20).max()
    med60 = val.shift(1).rolling(60).median()
    last = close.index[-1]
    cond = (close.loc[last] > hi20.loc[last]) & (val.loc[last] >= 2 * med60.loc[last])
    info = p[p['BAS_DD'] == last].set_index('ISU_CD')
    sel = info.loc[cond[cond].index.intersection(info.index)]
    sel = sel[sel['MKTCAP'] >= 5e11]
    sel = sel.assign(vr=(val.loc[last] / med60.loc[last]).reindex(sel.index))
    return sel.sort_values('vr', ascending=False).head(15)


# ---------------------------------------------------------------- 심리·수급
def _pct_rank(s, win=250, minp=100):
    """각 날짜 값이 직전 win일(최소 minp일) 안에서 몇 % 위치인지(0~100)."""
    return s.rolling(win, min_periods=minp).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)


def fear_greed(px, reg, asof):
    """자체 공포·탐욕 지수. 7개 구성요소를 각각 최근 1년 안의 위치(0~100)로 바꿔 평균한다(검증 전 참고 지표)."""
    comp = {}
    k = reg['kospi'][reg['kospi'].index <= asof]
    comp['모멘텀(코스피/125일선)'] = (k / k.rolling(125).mean() - 1).dropna()
    comp['시장 폭(60일선 위 종목 비율)'] = reg['breadth']
    p = px[(px['BAS_DD'] <= asof) & ~px['ISU_NM'].str.contains('스팩') & ~px['ISU_NM'].str.match(r'.*우[B-C]?$')]
    wide = p.pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    hi = wide.rolling(250, min_periods=240).max(); lo = wide.rolling(250, min_periods=240).min()
    n = hi.notna().sum(axis=1)
    hl = (((wide >= hi).sum(axis=1) - (wide <= lo).where(lo.notna()).sum(axis=1)) / n.replace(0, np.nan))[n > 500]
    if len(hl.dropna()) >= 60:
        comp['강도(신고가-신저가)'] = hl.dropna().rolling(5).mean().dropna()
    vk = reg['vk']
    comp['변동성(역방향)'] = -(vk / vk.rolling(50).mean()).dropna()
    o = load('opt_summary')
    if not o.empty:
        o = o[o['PROD_NM'].isin(['코스피200 옵션', '코스피200 위클리(목) 옵션', '코스피200 위클리(월) 옵션'])]
        o = o.assign(v=num(o['VOL'])).pivot_table(index='BAS_DD', columns='RGHT', values='v', aggfunc='sum')
        o.index = pd.to_datetime(o.index)
        if {'PUT', 'CALL'} <= set(o.columns):
            comp['풋/콜 거래량(역방향)'] = -(o['PUT'] / o['CALL']).rolling(5).mean().dropna()
    ktb = series_from('idx_bond', 'BND_IDX_GRP_NM', 'KTB 지수', 'TOT_EARNG_IDX')
    if len(ktb) > 30:
        sh = (k.pct_change(20) - ktb.reindex(k.index).ffill().pct_change(20)).dropna()
        comp['주식-국채 20일 수익률 차'] = sh
    etf = load('etf', ['BAS_DD', 'ISU_CD', 'LIST_SHRS'])
    def units(c):
        d = etf[etf['ISU_CD'] == c]
        return pd.Series(num(d['LIST_SHRS']).values, index=pd.to_datetime(d['BAS_DD'])).sort_index()
    lv, iv = units('122630'), units('252670')
    if len(lv) > 30 and len(iv) > 30:
        comp['레버리지/인버스 좌수 20일 변화'] = np.log(lv / iv).diff(20).dropna()
    ranks = {nm: _pct_rank(s[s.index <= asof]) for nm, s in comp.items()}
    df = pd.DataFrame(ranks).dropna(how='all')
    # 가격·변동성 요소로 시장 심리를, 풋콜·레버리지는 개인 포지션 쏠림으로 따로 본다(급락일에 저가 매수 쏠림이 심리 지수를 끌어올려 읽기를 흐리는 것 방지)
    crowd = [c for c in ['풋/콜 거래량(역방향)', '레버리지/인버스 좌수 20일 변화'] if c in df.columns]
    core = [c for c in df.columns if c not in crowd]
    total = df[core].mean(axis=1, skipna=True)[df[core].notna().sum(axis=1) >= 3]
    cr = df[crowd].mean(axis=1, skipna=True).dropna() if crowd else pd.Series(dtype=float)
    rows = [{'name': nm, 'val': df[nm].dropna().iloc[-1] if df[nm].notna().any() else np.nan, 'crowd': nm in crowd} for nm in df.columns]
    def label(x):
        return '극단적 공포' if x < 20 else '공포' if x < 40 else '중립' if x < 60 else '탐욕' if x < 80 else '극단적 탐욕'
    t = total.dropna()
    return {'rows': rows, 'now': t.iloc[-1] if len(t) else np.nan, 'w1': t.iloc[-6] if len(t) > 5 else np.nan,
            'm1': t.iloc[-22] if len(t) > 21 else np.nan, 'label': label(t.iloc[-1]) if len(t) else '확인 불가', 'series': t, 'label_fn': label,
            'crowd': cr.iloc[-1] if len(cr) else np.nan}


def market_flows(asof):
    """코스피·코스닥 투자자별 순매수(한투, 억원)와 증시자금."""
    out = {'mk': [], 'funds': None}
    a = asof.strftime('%Y%m%d')
    for mkt, nm in [('kospi', '코스피'), ('kosdaq', '코스닥')]:
        p = os.path.join(DATA, 'kis', f'mkt_investor_{mkt}.csv')
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, dtype=str)
        d = d[d['stck_bsop_date'] <= a].sort_values('stck_bsop_date')
        if d.empty or d['stck_bsop_date'].iloc[-1] != a:
            continue
        for col, who in [('frgn_ntby_tr_pbmn', '외국인'), ('orgn_ntby_tr_pbmn', '기관'), ('mrbn_ntby_tr_pbmn', '연기금'), ('prsn_ntby_tr_pbmn', '개인')]:
            v = num(d[col]) / 100
            out['mk'].append({'mkt': nm, 'who': who, 'd1': v.iloc[-1], 'd5': v.iloc[-5:].sum(), 'd20': v.iloc[-20:].sum()})
    p = os.path.join(DATA, 'kis', 'mktfunds.csv')
    if os.path.exists(p):
        f = pd.read_csv(p, dtype=str).sort_values('bsop_date')
        f = f[f['bsop_date'] <= a]
        if len(f) > 21:
            dep, crd = num(f['cust_dpmn_amt']), num(f['crdt_loan_rmnd'])
            out['funds'] = {'day': f['bsop_date'].iloc[-1], 'dep': dep.iloc[-1], 'dep20': (dep.iloc[-1] / dep.iloc[-21] - 1) * 100,
                            'crd': crd.iloc[-1], 'crd20': (crd.iloc[-1] / crd.iloc[-21] - 1) * 100}
    return out


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def watch_etfs(asof):
    """추적 ETF 보유 종목과 직전 수집일 대비 신규 편입·편출·비중 변화."""
    fs = sorted(glob.glob(os.path.join(DATA, 'kis', 'holdings', '*.csv')))
    fs = [f for f in fs if os.path.basename(f)[:8] <= (asof + timedelta(days=1)).strftime('%Y%m%d')] or fs
    if not fs:
        return None
    cur = pd.read_csv(fs[-1], dtype=str)
    prev = pd.read_csv(fs[-2], dtype=str) if len(fs) > 1 else None
    res = []
    for sec, g in cur[cur['sector'].str.startswith('추적:')].groupby('sector', sort=False):
        g = g.assign(w=num(g['weight'])).sort_values('w', ascending=False)
        g = g[g['code'].str.fullmatch(r'\d{6}')]
        item = {'name': sec[3:], 'n': len(g), 'top': list(zip(g['name'].head(5), g['w'].head(5))), 'codes': set(g['code']),
                'new': [], 'out': [], 'up': [], 'base': None, 'capped': len(g) >= 30}
        if prev is not None:
            pg = prev[prev['sector'] == sec]
            pg = pg.assign(w=num(pg['weight']))
            pg = pg[pg['code'].str.fullmatch(r'\d{6}')]
            if not pg.empty:
                item['base'] = os.path.basename(fs[-2])[:8]
                item['new'] = [(r['name'], r['w']) for _, r in g[~g['code'].isin(pg['code'])].iterrows()]
                item['out'] = list(pg[~pg['code'].isin(g['code'])]['name'])
                m = g.merge(pg[['code', 'w']], on='code', suffixes=('', '_p'))
                m = m.assign(dw=m['w'] - m['w_p']).sort_values('dw', ascending=False)
                item['up'] = [(r['name'], r['dw']) for _, r in m[m['dw'] >= 0.5].head(3).iterrows()]
        res.append(item)
    return {'day': os.path.basename(fs[-1])[:8], 'etfs': res}


def candidates(px, asof, sec, mem, wat, bo):
    """후보 종목과 들어갈 이유(근거)를 한 표로. 순위는 근거 개수 기준의 출발값이며 검증 전이다."""
    stage = {s['name']: s['stage'] for s in sec}
    pool = {}
    def add(code, why):
        pool.setdefault(code, set()).add(why)
    hfs = sorted(glob.glob(os.path.join(DATA, 'kis', 'holdings', '*.csv')))
    hd = pd.read_csv(hfs[-1], dtype=str) if hfs else pd.DataFrame(columns=['sector', 'code', 'name', 'weight'])
    secmap = hd[~hd['sector'].str.startswith('추적:')].assign(w=lambda x: num(x['weight'])).sort_values('w').drop_duplicates('code', keep='last').set_index('code')['sector']
    name2code = dict(zip(hd['name'], hd['code']))
    for nm, m in mem.items():
        if stage.get(nm) in ('주도', '초기'):
            for n, w, r in m['lag']:
                if n in name2code:
                    add(name2code[n], f'{nm}({stage[nm]}) 섹터 후발')
    cnt = {}
    if wat:
        for e in wat['etfs']:
            for c in e['codes']:
                cnt[c] = cnt.get(c, 0) + 1
            for n, w in e['new']:
                if n in name2code:
                    add(name2code[n], f"{e['name']} " + ('상위 30 진입' if e['capped'] else '신규 편입'))
        for c, n in cnt.items():
            if n >= 2:
                add(c, f'추적 ETF {n}곳 보유')
    for c in bo.index:
        add(c, '20일 신고가 돌파')
    if not pool:
        return []
    p = px[px['BAS_DD'] <= asof]
    close = p[p['ISU_CD'].isin(pool)].pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    last = p[p['BAS_DD'] == p['BAS_DD'].max()].set_index('ISU_CD')
    # 수급(한투, 최근 20거래일 순매수, 백만원)
    ff = sorted(glob.glob(os.path.join(DATA, 'kis', 'flows', '*.csv')))
    fl = pd.concat([pd.read_csv(f, dtype=str) for f in ff]) if ff else pd.DataFrame(columns=['date', 'code', 'frgn', 'orgn'])
    fl = fl[fl['date'] <= asof.strftime('%Y%m%d')]
    rows = []
    for c, why in pool.items():
        if c not in last.index or c not in close.columns:
            continue
        s = close[c].dropna()
        shr = p[p['ISU_CD'] == c].set_index('BAS_DD')['LIST_SHRS']
        ok = len(shr) > 22 and abs(shr.iloc[-1] / shr.iloc[-22] - 1) <= 0.03
        rv = rsi(s).iloc[-1] if ok and len(s) > 30 else np.nan
        m1 = (s.iloc[-1] / s.iloc[-22] - 1) * 100 if ok and len(s) > 22 else np.nan
        g = fl[fl['code'] == c].sort_values('date').tail(20)
        fr = num(g['frgn']).sum() / 100 if len(g) >= 15 else np.nan
        orr = num(g['orgn']).sum() / 100 if len(g) >= 15 else np.nan
        cap = last.loc[c, 'MKTCAP'] / 1e8
        why = set(why)
        if not np.isnan(fr) and fr > 0 and orr > 0:
            why.add('외국인·기관 동반 순매수(20일)')
        if not np.isnan(rv) and 50 <= rv <= 65:
            why.add('RSI 상승권(과열 전)')
        warn = []
        if not np.isnan(rv) and rv >= 70:
            warn.append('RSI 과열')
        if not np.isnan(rv) and rv <= 30:
            warn.append('RSI 과매도(반등 확인 전)')
        if not np.isnan(fr) and fr < 0 and orr < 0:
            warn.append('외국인·기관 동반 순매도')
        rows.append({'code': c, 'name': last.loc[c, 'ISU_NM'], 'sector': secmap.get(c, ''), 'stage': stage.get(secmap.get(c, ''), ''),
                     'm1': m1, 'rsi': rv, 'fr': fr, 'or': orr, 'flow_cap': (fr + orr) / cap * 100 if cap and not np.isnan(fr) else np.nan,
                     'n_etf': cnt.get(c, 0), 'why': sorted(why), 'warn': warn, 'cap': cap})
    rows.sort(key=lambda x: (-(len(x['why']) - len(x['warn'])), -(x['flow_cap'] if not np.isnan(x['flow_cap']) else -99)))
    return rows[:20]


# ---------------------------------------------------------------- 0. 원페이지 추천
PICK_RULE = {'min_cap': 5000, 'min_val': 50, 'rsi_hot': 75, 'b_lag_floor': -15, 'b_ma60_floor': -5,
             'rev_yoy': 10, 'n_max': {'위험선호': 5, '중립': 3, '위험회피': 2}}


def fin_table(asof, codes):
    """최근 분기 두 개의 매출·영업이익 전년 같은 분기 대비(연결 우선)."""
    import dart_fin as D
    ps, files = D.quarterly(asof, codes)
    q = [x for x in ps if x[1] != '11011'][:2] or ps[:2]
    t = {}
    for i, k in enumerate(q):
        d = pd.read_csv(files[k], dtype={'stock_code': str, 'rcept_no': str})
        for _, r in d.iterrows():
            e = t.setdefault(r['stock_code'], {})
            sfx = '' if i == 0 else '_prev'
            e['rev' + sfx], e['rev_py' + sfx], e['op' + sfx], e['op_py' + sfx] = r['rev'], r['rev_py'], r['op'], r['op_py']
            if i == 0:
                e['period'] = f"{str(k[0])[2:]}년 {D.RC[k[1]]}분기"
    return t


def _yoy(a, b):
    return (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def picks(px, asof, sec, wat, reg):
    import dart_fin as D
    R = PICK_RULE
    stage = {s['name']: s for s in sec}
    hfs = sorted(glob.glob(os.path.join(DATA, 'kis', 'holdings', '*.csv')))
    if not hfs:
        return None
    hd = pd.read_csv(hfs[-1], dtype=str)
    hd = hd[~hd['sector'].str.startswith('추적:') & hd['code'].str.fullmatch(r'\d{6}')]
    hd = hd.assign(w=num(hd['weight'])).sort_values('w').drop_duplicates('code', keep='last')
    hd = hd[hd['sector'].map(lambda s: stage.get(s, {}).get('stage') in ('주도', '초기'))]
    if hd.empty:
        return {'rows': [], 'watch': [], 'n_max': R['n_max'][reg['regime']], 'note': '주도·초기 섹터 없음'}
    p = px[px['BAS_DD'] <= asof]
    days = sorted(p['BAS_DD'].unique())
    pp = p[p['ISU_CD'].isin(hd['code']) & p['BAS_DD'].isin(days[-130:])]
    close = pp.pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    low = pp.pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC').rolling(20).min()
    val = pp.pivot_table(index='BAS_DD', columns='ISU_CD', values='ACC_TRDVAL')
    shr = pp.pivot_table(index='BAS_DD', columns='ISU_CD', values='LIST_SHRS')
    last = p[p['BAS_DD'] == days[-1]].set_index('ISU_CD')
    act = {}
    if wat:
        for e in wat['etfs']:
            for c in e['codes']:
                act.setdefault(c, []).append(e['name'])
            for n, w in e['new']:
                pass
    newin = {n for e in (wat['etfs'] if wat else []) for n, w in e['new']}
    ff = sorted(glob.glob(os.path.join(DATA, 'kis', 'flows', '*.csv')))
    fl = pd.concat([pd.read_csv(f, dtype=str) for f in ff]) if ff else pd.DataFrame(columns=['date', 'code', 'frgn', 'orgn'])
    fl = fl[fl['date'] <= asof.strftime('%Y%m%d')]
    fin = fin_table(asof, list(hd['code']))
    cand = []
    for _, h in hd.iterrows():
        c, sname = h['code'], h['sector']
        if c not in close.columns or c not in last.index:
            continue
        s = close[c].dropna()
        if len(s) < 64:
            continue
        sh = shr[c].dropna()
        if abs(sh.iloc[-1] / sh.iloc[-64] - 1) > 0.03:
            continue   # 3개월 안 분할·병합 등: 시세 비교 불가
        cap = last.loc[c, 'MKTCAP'] / 1e8
        v20 = val[c].iloc[-20:].mean() / 1e8
        if cap < R['min_cap'] or v20 < R['min_val']:
            continue
        S = stage[sname]
        r1, r3 = (s.iloc[-1] / s.iloc[-22] - 1) * 100, (s.iloc[-1] / s.iloc[-64] - 1) * 100
        g60 = (s.iloc[-1] / s.iloc[-60:].mean() - 1) * 100
        rv = rsi(s).iloc[-1]
        typ = None
        if S['stage'] == '초기' and r1 > S['m1'] and r3 > S['m3']:
            typ = 'A'
        elif S['stage'] == '주도' and r1 < S['m1'] and r3 - S['m3'] > R['b_lag_floor'] and g60 > R['b_ma60_floor']:
            typ = 'B'
        if not typ:
            continue
        g = fl[fl['code'] == c].sort_values('date').tail(20)
        fr = num(g['frgn']).sum() / 100 if len(g) >= 15 else np.nan
        orr = num(g['orgn']).sum() / 100 if len(g) >= 15 else np.nan
        f = fin.get(c, {})
        ry, ryp = _yoy(f.get('rev'), f.get('rev_py')), _yoy(f.get('rev_prev'), f.get('rev_py_prev'))
        op, opp = f.get('op', np.nan), f.get('op_py', np.nan)
        cand.append({'code': c, 'name': last.loc[c, 'ISU_NM'], 'sector': sname, 'type': typ, 'close': s.iloc[-1],
                     'r1': r1, 'r3': r3, 'rs1': r1 - S['m1'], 'rs3': r3 - S['m3'], 'g60': g60, 'rsi': rv, 'cap': cap,
                     'stop': low[c].iloc[-1], 'fr': fr, 'or': orr, 'ry': ry, 'ryp': ryp, 'op': op, 'opp': opp,
                     'period': f.get('period', ''), 'act': act.get(c, []), 'newin': h['name'] in newin})
    ev = D.corp_events([x['code'] for x in cand], asof)
    rows, watch, cut = [], [], []
    for x in cand:
        e = ev.get(x['code'], {'rights': [], 'supply': [], 'burn': [], 'prelim': []})
        gate = []
        if e['rights']:
            gate.append(f"{e['rights'][0][0][4:6]}/{e['rights'][0][0][6:]} " + re.sub(r'^\[.*?\]', '', e['rights'][0][1]).replace('결정', '').strip())
        if pd.notna(x['op']) and x['op'] < 0 and pd.notna(x['ry']) and x['ry'] < 0:
            gate.append('영업적자·매출 감소')
        if pd.notna(x['rsi']) and x['rsi'] >= R['rsi_hot']:
            gate.append(f"과열(RSI {x['rsi']:.0f})")
        if x['type'] == 'B' and pd.notna(x['fr']) and x['fr'] < 0 and x['or'] < 0:
            gate.append('외국인·기관 동반 매도')
        earn = pd.notna(x['ry']) and x['ry'] >= R['rev_yoy'] and pd.notna(x['ryp']) and x['ryp'] > 0 and pd.notna(x['op']) and x['op'] > 0 \
            and (pd.isna(x['opp']) or x['opp'] <= 0 or x['op'] >= x['opp'])
        why = []
        if earn:
            opt = '흑자전환' if pd.notna(x['opp']) and x['opp'] <= 0 else f"영업이익 {_yoy(x['op'], x['opp']):+.0f}%"
            why.append(f"{x['period']} 매출 {x['ry']:+.0f}%(직전 분기 {x['ryp']:+.0f}%), {opt}")
        if pd.notna(x['fr']) and x['fr'] + x['or'] > 0:
            why.append(f"외국인·기관 20일 {x['fr'] + x['or']:+,.0f}억")
        if x['act']:
            why.append(('액티브 ETF 신규 편입' if x['newin'] else f"액티브 ETF {len(x['act'])}곳 보유"))
        evs = (['공급계약'] if e['supply'] else []) + (['자사주 소각'] if e['burn'] else [])
        if evs:
            why.append('최근 30일 ' + '·'.join(evs) + ' 공시')
        x.update({'gate': gate, 'earn': earn, 'why': why, 'score': len(why), 'prelim': e['prelim'][:1]})
        if gate:
            cut.append(x)
        elif earn and x['score'] >= 2:
            rows.append(x)
        else:
            watch.append(x)
    key = lambda x: (-x['score'], -((x['fr'] if pd.notna(x['fr']) else 0) + (x['or'] if pd.notna(x['or']) else 0)) / x['cap'])
    rows.sort(key=key); watch.sort(key=key)
    n = R['n_max'][reg['regime']]
    return {'rows': rows[:n], 'more': rows[n:], 'watch': watch[:6], 'cut': cut, 'n_max': n, 'n_cand': len(cand)}


# ---------------------------------------------------------------- 그래프
def chart_market(reg, asof):
    k = reg['kospi'][reg['kospi'].index <= asof].iloc[-260:]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.plot(k.index, k.values, color=INK, lw=1.4)
    full = reg['kospi'][reg['kospi'].index <= asof]
    ax.plot(k.index, full.rolling(60).mean().reindex(k.index), color=ACC, lw=1, label='60일 평균')
    ax.plot(k.index, full.rolling(200).mean().reindex(k.index), color=MUTED, lw=1, ls='--', label='200일 평균')
    style_ax(ax); ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('코스피와 이동평균', fontsize=9, color=INK, loc='left')
    ax2 = ax.twinx(); b = reg['breadth'].reindex(k.index)
    ax2.fill_between(b.index, b.values, color=ACC, alpha=0.08); ax2.set_ylim(0, 100)
    ax2.tick_params(colors=MUTED, labelsize=7); ax2.spines[:].set_visible(False)
    ax2.set_ylabel('60일선 위 종목 비율(%)', fontsize=7, color=MUTED)
    return fig_b64(fig)


def chart_sectors(rows):
    r = sorted(rows, key=lambda x: x['rs3'])
    fig, ax = plt.subplots(figsize=(7.2, 0.17 * len(r) + 0.5))
    cols = [UP if x['rs3'] > 0 else DOWN for x in r]
    ax.barh([x['name'] for x in r], [x['rs3'] for x in r], color=cols, height=0.6)
    style_ax(ax); ax.grid(axis='x', color=GRID, linewidth=0.6); ax.grid(axis='y', visible=False)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title('3개월 코스피 대비 초과수익(%p)', fontsize=9, color=INK, loc='left')
    return fig_b64(fig)


def chart_exports(ex):
    s, t = ex['semi_yoy'].iloc[-24:], ex['tot_yoy'].iloc[-24:]
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.plot(s.index, s.values, color=UP, lw=1.6, label='반도체')
    ax.plot(t.index, t.values, color=INK, lw=1.2, label='전체')
    ax.axhline(0, color=MUTED, lw=0.8)
    style_ax(ax); ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('월간 수출 전년 대비 증가율(%)', fontsize=9, color=INK, loc='left')
    return fig_b64(fig)


# ---------------------------------------------------------------- HTML
CSS = """
@page { size: A4; margin: 14mm 13mm; }
body { font-family: 'Noto Sans CJK KR','Noto Sans CJK JP',sans-serif; color:#1f2328; font-size:9.6pt; line-height:1.5; }
h1 { font-size:16pt; margin:0 0 2px; } .sub { color:#6b7280; font-size:8.5pt; margin-bottom:10px; }
h2 { font-size:11.5pt; margin:16px 0 6px; padding-bottom:3px; border-bottom:1.5px solid #1f2328; }
h3 { font-size:9.8pt; margin:10px 0 4px; }
table { width:100%; border-collapse:collapse; font-size:8.6pt; margin:4px 0 6px; }
th { text-align:left; color:#6b7280; font-weight:500; border-bottom:1px solid #d1d5db; padding:3px 4px; }
td { border-bottom:1px solid #eef0f2; padding:3px 4px; font-variant-numeric: tabular-nums; }
td.r, th.r { text-align:right; }
.up { color:#c0392b; } .dn { color:#1f5fbf; } .mut { color:#9ca3af; } td.r { white-space:nowrap; }
.box { background:#f6f7f9; border-radius:6px; padding:8px 10px; margin:6px 0; }
.tag { display:inline-block; padding:0 6px; border-radius:9px; font-size:8pt; background:#eef2ee; color:#2f6f4f; }
.tag.hot { background:#fbeaea; color:#c0392b; } .tag.cold { background:#e9eff9; color:#1f5fbf; } .tag.neu { background:#f1f2f4; color:#6b7280; }
.note { color:#6b7280; font-size:7.8pt; }
img { width:100%; }
.pb { page-break-before: always; }
"""


def stage_tag(st):
    cls = {'주도': '', '초기': '', '과열 주의': 'hot', '쇠퇴': 'cold', '중립': 'neu'}[st]
    return f'<span class="tag {cls}">{st}</span>'


def onepage(asof, reg, sec, fg, mf, pk, wat):
    h = [f"<h1>오늘의 종목</h1><div class='sub'>{asof:%Y년 %m월 %d일} 장 마감 기준 · 판단 기준은 검증 전 출발값</div>"]
    k = reg['idx_rows'][0]
    n = pk['n_max'] if pk else 0
    act = {'위험선호': f'신규 매수 최대 {n}종목', '중립': f'신규 매수 최대 {n}종목, 목표 비중의 1/3씩',
           '위험회피': f'신규 매수 최대 {n}종목, 목표 비중의 1/3만'}[reg['regime']]
    fgs = f", 공포·탐욕 {fg['now']:.0f}" if not np.isnan(fg['now']) else ''
    h.append(f"<div class=box><b>시장 {reg['regime']} → {act}</b><br><span class=note>코스피 {k['close']:,.0f}({fmt_pct(k['d1'])}), 200일선 대비 {fmt_pct(k['ma200'])}, 60일선 위 종목 {reg['breadth_last']:.0f}%{fgs}</span></div>")
    ld = [x for x in sec if x['stage'] in ('주도', '초기')]
    if ld:
        h.append("<div style='margin:6px 0'><b>볼 섹터</b> " + ' · '.join(f"{x['name']}({x['stage']}, 3개월 코스피 대비 {x['rs3']:+.0f}%p)" for x in ld) + "</div>")
    else:
        h.append("<div><b>볼 섹터</b> 주도·초기 섹터 없음</div>")
    tn = {'A': '초기 섹터의 주도주', 'B': '주도 섹터의 차기주자'}
    if pk and pk['rows']:
        h.append("<table><tr><th>종목</th><th>유형</th><th>근거</th><th class=r>손절선</th></tr>")
        for x in pk['rows']:
            stop = f"{x['stop']:,.0f}<br><span class=mut>{(x['stop'] / x['close'] - 1) * 100:+.1f}%</span>"
            h.append(f"<tr><td style='white-space:nowrap'><b>{x['name']}</b><br><span class=mut>{x['sector']} · {x['close']:,.0f}원</span></td>"
                     f"<td style='white-space:nowrap'>{tn[x['type']]}</td><td>{' / '.join(x['why'])}</td><td class=r>{stop}</td></tr>")
        h.append("</table>")
    else:
        h.append("<div class=box>오늘은 기준을 모두 넘은 종목이 없습니다.</div>")
    if pk and pk.get('watch'):
        def lack(x):
            m = []
            if not x['earn']:
                m.append('실적 확인 안 됨' if pd.isna(x['ry']) else '실적 상승세 아님')
            if x['score'] < 2:
                m.append('근거 부족')
            return ', '.join(m)
        h.append("<div style='margin-top:6px'><b>관찰</b> " + ' · '.join(f"{x['name']}({tn[x['type']][:2] if False else ('초기 주도' if x['type']=='A' else '차기주자')}, {lack(x)})" for x in pk['watch']) + "</div>")
    if pk and pk.get('cut'):
        h.append("<div class=note>탈락: " + ' · '.join(f"{x['name']}({x['gate'][0]})" for x in pk['cut'][:8]) + "</div>")
    if pk and pk.get('rows'):
        pre = [f"{x['name']} {x['prelim'][0][0][4:6]}/{x['prelim'][0][0][6:]}" for x in pk['rows'] if x['prelim']]
        if pre:
            h.append("<div class=note>최근 잠정실적 공시가 있어 원문 확인 필요: " + ', '.join(pre) + "</div>")
    ch = []
    if wat:
        for e in wat['etfs']:
            if e['base'] and e['new']:
                ch.append(f"{e['name']} " + ('상위 30 진입' if e['capped'] else '신규 편입') + ': ' + ', '.join(n for n, w in e['new'][:4]))
    fk = {(r['mkt'], r['who']): r for r in mf['mk']}.get(('코스피', '외국인'))
    if fk:
        ch.append(f"코스피 외국인 오늘 {fk['d1']:+,.0f}억, 20일 {fk['d20']:+,.0f}억")
    if ch:
        h.append("<div style='margin-top:6px'><b>오늘 달라진 점</b><br>" + '<br>'.join('· ' + c for c in ch[:3]) + "</div>")
    R = PICK_RULE
    h.append("<div class=box style='margin-top:10px;font-size:8.5pt'><b>고르는 순서</b><br>"
             f"① 시장 국면으로 신규 매수 개수를 정한다(위험선호 {R['n_max']['위험선호']} / 중립 {R['n_max']['중립']} / 위험회피 {R['n_max']['위험회피']}).<br>"
             "② 주도·초기 섹터 안에서만 고른다. 섹터는 산업지수·테마 ETF의 1·3·6개월 코스피 대비 수익률과 60·120일선 위치로 판정.<br>"
             f"③ 유형: <b>초기 섹터의 주도주</b> = 섹터보다 1개월·3개월 모두 강한 종목 / <b>주도 섹터의 차기주자</b> = 섹터보다 1개월 덜 올랐지만 3개월 격차 {abs(R['b_lag_floor'])}%p 이내, 60일선 대비 {R['b_ma60_floor']}% 위(추세 유지).<br>"
             f"④ 탈락: 최근 90일 유상증자·CB·BW 결정, 영업적자이면서 매출 감소, RSI {R['rsi_hot']} 이상, (차기주자) 외국인·기관 20일 동반 순매도. 시총 {R['min_cap']:,}억·하루 거래대금 {R['min_val']}억 미만 제외.<br>"
             f"⑤ 추천: 실적 상승세(최근 분기 매출 전년비 +{R['rev_yoy']}% 이상, 직전 분기도 증가, 영업이익 흑자이고 줄지 않음, DART 정기보고서) 필수 + 수급·액티브 ETF·공시 근거 중 1개 이상. 근거 수, 시총 대비 수급 순.<br>"
             "⑥ 손절선: 최근 20거래일 최저 종가. 종가로 깨면 정리. 진입은 목표 비중의 1/3부터, 추가는 20일 신고가 돌파 때만.</div>")
    return h


def build(asof):
    px = stock_panel()
    reg = market_regime(px, asof)
    mac, glo = macro(asof)
    sec = sector_table(asof, reg['kospi'])
    mem, mem_day = sector_members(px, asof, sec)
    fg = fear_greed(px, reg, asof)
    mf = market_flows(asof)
    wat = watch_etfs(asof)
    ex = exports()
    exi = export_items()
    ev = dart_events(asof)
    bo = breakout(px, asof)
    cand = candidates(px, asof, sec, mem, wat, bo)
    pk = picks(px, asof, sec, wat, reg)
    h = []
    h += onepage(asof, reg, sec, fg, mf, pk, wat)
    h.append(f"<h1 class=pb>부록 — 근거 데이터</h1><div class='sub'>{asof:%Y년 %m월 %d일} 장 마감 기준</div>")
    # 요약
    lead = [s for s in sec if s['stage'] == '주도'][:4]
    early = [s for s in sec if s['stage'] == '초기'][:4]
    hot = [s for s in sec if s['stage'] == '과열 주의']
    k = reg['idx_rows'][0]
    h.append("<div class='box'><b>오늘의 요약</b><br>")
    h.append(f"· 시장 국면: <b>{reg['regime']}</b> — 코스피 {k['close']:,.0f}({color_pct(k['d1'])}), 200일선 대비 {fmt_pct(k['ma200'])}, 60일선 위 종목 {reg['breadth_last']:.0f}%, 변동성지수 {reg['vk_last']:.1f}(직전 50일 평균 {reg['vk_avg']:.1f})<br>")
    fgs = f"{fg['now']:.0f}({fg['label']})" if not np.isnan(fg['now']) else '확인 불가'
    fl = {(r['mkt'], r['who']): r for r in mf['mk']}
    fk = fl.get(('코스피', '외국인'))
    h.append(f"· 심리: 공포·탐욕 {fgs}" + (f" / 코스피 외국인 순매수 오늘 {fk['d1']:+,.0f}억, 20일 {fk['d20']:+,.0f}억" if fk else '') + "<br>")
    h.append(f"· 주도: {', '.join(x['name'] for x in lead) or '없음'} / 초기: {', '.join(x['name'] for x in early) or '없음'} / 과열 주의: {', '.join(x['name'] for x in hot) or '없음'}<br>")
    semi = [r for r in ex['rows'] if r['item'] == '반도체'][0]
    h.append(f"· 수출({ex['ym'][:4]}.{ex['ym'][4:]} {ex['per']}일): 반도체 전년비 {fmt_pct(semi['yoy'],0)}, 하루 평균 기준 {fmt_pct(semi['dyoy'],0)}<br>")
    h.append(f"· 공시({ev['start']:%m/%d}~{ev['end']:%m/%d}): 자사주 소각 {len(ev['burn'])}건, 공급계약 {ev['n_supply']}건, 잠정실적 {len(ev['prelim'])}건, 증자·CB·BW {len(ev['rights'])}건</div>")

    # 1. 시장 국면
    h.append("<h2>1. 시장 국면</h2>")
    h.append("<table><tr><th>지수</th><th class=r>종가</th><th class=r>전일비</th><th class=r>1개월</th><th class=r>60일선 대비</th><th class=r>200일선 대비</th></tr>")
    for r in reg['idx_rows']:
        h.append(f"<tr><td>{r['name']}</td><td class=r>{r['close']:,.2f}</td><td class=r>{color_pct(r['d1'],2)}</td><td class=r>{color_pct(r['m1'])}</td><td class=r>{color_pct(r['ma60'])}</td><td class=r>{color_pct(r['ma200'])}</td></tr>")
    h.append("</table>")
    h.append(f"<img src='{chart_market(reg, asof)}'>")
    nh = f"{reg['nh']} / {reg['nl']}" if reg['nh_ok'] else '확인 불가(기간 부족)'
    h.append(f"<table><tr><th>시장 폭·심리</th><th class=r>값</th><th>읽는 법</th></tr>"
             f"<tr><td>60일선 위 종목 비율(코스피·코스닥)</td><td class=r>{reg['breadth_last']:.1f}%</td><td class=note>50% 위면 상승 확산, 30% 아래면 위축</td></tr>"
             f"<tr><td>52주 신고가 / 신저가 종목 수</td><td class=r>{nh}</td><td class=note>신저가가 많으면 하락 확산</td></tr>"
             f"<tr><td>코스피200 변동성지수</td><td class=r>{reg['vk_last']:.2f}</td><td class=note>직전 50일 평균 {reg['vk_avg']:.2f}</td></tr>")
    for nm, v in reg['lev'].items():
        h.append(f"<tr><td>{nm} 좌수 20일 변화</td><td class=r>{color_pct(v)}</td><td class=note>개인 쏠림 참고(역지표 성격)</td></tr>")
    h.append("</table>")
    h.append("<h3>공포·탐욕 지수(자체 산출)</h3>")
    if not np.isnan(fg['now']):
        cw = f" / 개인 포지션 쏠림 {fg['crowd']:.0f}" if not np.isnan(fg['crowd']) else ''
        h.append(f"<div class=box><b>{fg['now']:.0f} · {fg['label']}</b> &nbsp; 1주 전 {fg['w1']:.0f}, 1개월 전 {fg['m1']:.0f}{cw}</div>")
        h.append("<table><tr><th>구성 요소</th><th class=r>최근 1년 중 위치</th></tr>")
        for r in sorted(fg['rows'], key=lambda x: x['crowd']):
            v = f"{r['val']:.0f}" if not np.isnan(r['val']) else '<span class=mut>기간 부족</span>'
            h.append(f"<tr><td>{r['name']}{' — 개인 쏠림' if r['crowd'] else ''}</td><td class=r>{v}</td></tr>")
        h.append("</table>")
    h.append("<div class=note>각 요소를 최근 1년(최소 100거래일) 안에서 오늘 값이 어디쯤인지 0~100으로 바꿔 평균했습니다. 0에 가까울수록 공포, 100에 가까울수록 탐욕입니다. 지수 값은 가격·변동성 다섯 요소의 평균이고, 풋/콜과 레버리지 좌수는 '개인 포지션 쏠림'으로 따로 봅니다. 하락장에서 이 값이 높으면 저가 매수가 몰리고 있다는 뜻이라 바닥 신호로 보기 이릅니다. 미국 CNN 지수의 구성을 국내 자료로 옮긴 자체 지표이고, 매매 신호로서의 효과는 아직 검증 전입니다.</div>")
    if mf['mk']:
        h.append("<h3>투자자별 순매수 (억 원, 한국투자증권)</h3><table><tr><th>시장</th><th>투자자</th><th class=r>오늘</th><th class=r>5일</th><th class=r>20일</th></tr>")
        for r in mf['mk']:
            h.append(f"<tr><td>{r['mkt']}</td><td>{r['who']}</td><td class=r>{r['d1']:+,.0f}</td><td class=r>{r['d5']:+,.0f}</td><td class=r>{r['d20']:+,.0f}</td></tr>")
        h.append("</table>")
    if mf['funds']:
        f_ = mf['funds']
        h.append(f"<div class=note>증시자금({f_['day'][4:6]}/{f_['day'][6:]}): 고객예탁금 {f_['dep']/1e4:,.1f}조 원(20일 {fmt_pct(f_['dep20'])}), 신용융자 잔고 {f_['crd']/1e4:,.1f}조 원(20일 {fmt_pct(f_['crd20'])}). 신용 잔고가 예탁금보다 빠르게 늘면 빚투 쏠림으로 봅니다.</div>")
    h.append("<h3>금리·환율 (거래소 선물, 정규시장 최다 거래 월물)</h3><table><tr><th>항목</th><th class=r>종가</th><th class=r>전일비</th><th class=r>20일</th><th>읽는 법</th></tr>")
    for r in mac:
        h.append(f"<tr><td>{r['label']}</td><td class=r>{r['last']:,.2f}</td><td class=r>{color_pct(r['d1'],2)}</td><td class=r>{color_pct(r['m1'],2)}</td><td class=note>{r['note']}</td></tr>")
    h.append("</table>")
    h.append("<h3>해외·원자재 (국내 상장 ETF의 기초지수 등락)</h3><table><tr><th>지수</th><th class=r>최근 1일</th><th class=r>20일 누적</th><th>출처 ETF</th></tr>")
    for r in glo:
        h.append(f"<tr><td>{r['label']}</td><td class=r>{color_pct(r['d1'],2)}</td><td class=r>{color_pct(r['m1'])}</td><td class=note>{r['src']}</td></tr>")
    h.append("</table><div class=note>해외 지수는 해당 ETF가 공시한 기초지수 등락이라 미국 장 마감 시점과 하루 차이가 날 수 있습니다.</div>")

    # 2. 섹터
    h.append("<h2>2. 섹터 단계</h2>")
    h.append("<table><tr><th>섹터</th><th>단계</th><th class=r>1개월</th><th class=r>3개월</th><th class=r>6개월</th><th class=r>3개월 초과</th><th class=r>6개월 초과</th><th class=r>60일선 대비</th><th class=r>ETF 좌수 20일</th></tr>")
    for s in sec:
        u = color_pct(s['units']) if s['units'] is not None else '<span class=mut>–</span>'
        h.append(f"<tr><td>{s['name']}</td><td>{stage_tag(s['stage'])}</td><td class=r>{color_pct(s['m1'])}</td><td class=r>{color_pct(s['m3'])}</td><td class=r>{color_pct(s['m6'])}</td>"
                 f"<td class=r>{color_pct(s['rs3'])}</td><td class=r>{color_pct(s['rs6'])}</td><td class=r>{color_pct(s['g60'])}</td><td class=r>{u}</td></tr>")
    h.append("</table>")
    h.append(f"<img src='{chart_sectors(sec)}'>")
    h.append("<h3>섹터별 종목</h3>")
    if mem:
        h.append("<table><tr><th>섹터</th><th>대표 종목(ETF 비중 상위)</th><th>섹터보다 덜 오른 종목(1개월)</th></tr>")
        for s in sec:
            m = mem.get(s['name'])
            if not m:
                h.append(f"<tr><td style='white-space:nowrap'>{s['name']}</td><td colspan=2 class=mut>추종 ETF가 없어 구성종목 확인 불가</td></tr>")
                continue
            top = ', '.join(f"{n} {w:.0f}%" + (f"({fmt_pct(r)})" if not np.isnan(r) else '') for n, w, r in m['top'])
            lag = ', '.join(f"{n}({fmt_pct(r)})" for n, w, r in m['lag']) or '<span class=mut>해당 없음</span>'
            h.append(f"<tr><td style='white-space:nowrap'>{s['name']}</td><td>{top}</td><td>{lag}</td></tr>")
        h.append("</table>")
        h.append(f"<div class=note>구성종목은 각 섹터 ETF의 편입 내역({mem_day[4:6]}/{mem_day[6:]} 한국투자증권 조회), 괄호는 최근 1개월 주가 등락입니다. "
                 "'덜 오른 종목'은 비중 4~15위 중 시가총액 5천억 원 이상이면서 섹터보다 1개월 덜 오른 종목으로, 가격만 본 후보이고 공시 점검은 아직 거치지 않았습니다.</div>")
    else:
        h.append("<div class=note>섹터 구성종목: 확인 불가(한투 수집 전)</div>")
    h.append("<div class=note>산업 섹터는 KRX 산업지수, 조선·방산·2차전지·전력기기·소부장은 순자산이 가장 큰 국내 테마 ETF의 순자산가치로 계산했습니다. 이번 예시판의 단계는 가격 축만 반영했고, 변화 축(수출·잠정실적·공급계약)과 섹터 내 확산도는 섹터 분류표 확정 후 붙입니다.</div>")

    # 3. 추적 ETF
    h.append("<h2>3. 액티브·전략 ETF가 담는 종목</h2>")
    if wat and wat['etfs']:
        h.append("<table><tr><th>ETF</th><th class=r>종목 수</th><th>비중 상위 5</th><th>신규 편입 / 편출 / 비중 확대</th></tr>")
        for e in wat['etfs']:
            top = ', '.join(f"{n} {w:.1f}%" for n, w in e['top'])
            if e['base']:
                a_, b_ = ('상위 30 진입', '상위 30 이탈') if e['capped'] else ('편입', '편출')
                chg = ((f'{a_}: ' + ', '.join(n for n, w in e['new'])) if e['new'] else f'{a_} 없음') + ' / ' + \
                      ((f'{b_}: ' + ', '.join(e['out'])) if e['out'] else f'{b_} 없음') + \
                      ((' / 확대: ' + ', '.join(f"{n} {d:+.1f}%p" for n, d in e['up'])) if e['up'] else '')
            else:
                chg = '<span class=mut>비교 기준 없음(수집 첫날)</span>'
            h.append(f"<tr><td style='white-space:nowrap'>{e['name']}</td><td class=r>{e['n']}</td><td>{top}</td><td>{chg}</td></tr>")
        h.append("</table>")
        base = next((e['base'] for e in wat['etfs'] if e['base']), None)
        h.append(f"<div class=note>{wat['day'][4:6]}/{wat['day'][6:]} 한국투자증권 조회 기준" + (f", 변화는 {base[4:6]}/{base[6:]} 대비" if base else '') +
                 ". 운용사가 매일 공개하는 편입 내역이라 운용역이 새로 담거나 비중을 늘린 종목이 그대로 드러납니다. TIGER 모멘텀은 지수형이라 변화는 정기 변경 때 몰립니다. 한투 조회가 ETF당 비중 상위 30종목까지만 주기 때문에, 30종목 이상 담은 ETF는 '상위 30 진입·이탈'로 표시하며 이것이 실제 신규 편입·편출과 다를 수 있습니다.</div>")
    else:
        h.append("<div class=note>확인 불가(한투 수집 전)</div>")

    # 4. 후보와 근거
    h.append("<h2>4. 후보 종목과 들어갈 이유</h2>")
    if cand:
        h.append("<table><tr><th>종목</th><th>섹터(단계)</th><th class=r>1개월</th><th class=r>RSI</th><th class=r>외국인 20일</th><th class=r>기관 20일</th><th class=r>수급/시총</th><th>근거 <span class=up>(주의)</span></th></tr>")
        for c in cand:
            sg = f"{c['sector']}({c['stage']})" if c['sector'] else '<span class=mut>–</span>'
            fmt = lambda v: f"{v:+,.0f}억" if not np.isnan(v) else '<span class=mut>–</span>'
            rv = f"{c['rsi']:.0f}" if not np.isnan(c['rsi']) else '<span class=mut>–</span>'
            h.append(f"<tr><td style='white-space:nowrap'>{c['name']}</td><td>{sg}</td><td class=r>{color_pct(c['m1'])}</td><td class=r>{rv}</td><td class=r>{fmt(c['fr'])}</td><td class=r>{fmt(c['or'])}</td>"
                     f"<td class=r>{fmt_pct(c['flow_cap'], 2)}</td><td style='font-size:8.5pt'>{', '.join(c['why'])}" + (f" <span class=up>({', '.join(c['warn'])})</span>" if c['warn'] else '') + "</td></tr>")
        h.append("</table>")
        h.append("<div class=note>후보는 (가) 주도·초기 섹터에서 덜 오른 종목, (나) 추적 ETF 2곳 이상 보유 또는 신규 편입, (다) 오늘 20일 신고가 돌파 종목을 모은 뒤 근거 수에서 주의 수를 뺀 값, 수급 순으로 늘어놓았습니다. "
                 "RSI는 14일 기준으로 50~65는 상승 흐름이되 과열 전, 70 이상은 과열로 봅니다. 수급은 최근 20거래일 순매수 금액이고 수급/시총은 외국인+기관 순매수를 시가총액으로 나눈 값입니다. 공시 관문(재무·증자·지배구조)과 매도 규칙은 아직 붙이지 않은 단계라 매수 추천이 아닙니다.</div>")
    else:
        h.append("<div class=note>확인 불가(한투 수집 전)</div>")

    # 5. 수출
    wd = ex['wd']
    h.append("<h2>5. 수출 (관세청 10일 단위 잠정치)</h2>")
    h.append(f"<div class=note>최신: {ex['ym'][:4]}년 {ex['ym'][4:]}월 {ex['per']}일 · 단위 백만 달러 · 조업일수는 거래소 개장일로 근사(이번 {wd[0]:.0f}일, 전년 같은 기간 {wd[1]:.0f}일, 전월 같은 기간 {wd[2]:.0f}일). 연초 누계와 전분기 대비는 월 확정치({ex['last_full']}까지, 분기는 {ex['q_note']}) 기준.</div>")
    h.append("<table><tr><th>품목</th><th class=r>금액</th><th class=r>전년비</th><th class=r>전년비(하루 평균)</th><th class=r>전월비</th><th class=r>전월비(하루 평균)</th><th class=r>전분기비</th><th class=r>연초 누계</th><th class=r>누계 전년비</th></tr>")
    for r in ex['rows']:
        h.append(f"<tr><td>{r['item']}</td><td class=r>{r['cur']:,.0f}</td><td class=r>{color_pct(r['yoy'],0)}</td><td class=r>{color_pct(r['dyoy'],0)}</td><td class=r>{color_pct(r['mom'],0)}</td><td class=r>{color_pct(r['dmom'],0)}</td>"
                 f"<td class=r>{color_pct(r.get('qoq', np.nan),0)}</td><td class=r>{r['ytd']:,.0f}</td><td class=r>{color_pct(r['ytd_yoy'],0)}</td></tr>")
    h.append("</table>")
    if wd[0] and wd[1] and abs(wd[0] - wd[1]) >= 2:
        h.append(f"<div class=note>조업일수 차이가 커서(이번 {wd[0]:.0f}일 vs 전년 {wd[1]:.0f}일) 단순 전년비는 왜곡돼 있습니다. 하루 평균 기준을 우선 보세요.</div>")
    h.append(f"<img src='{chart_exports(ex)}'>")
    if exi:
        h.append(f"<h3>품목별 월 수출 ({exi['ym'][:4]}년 {int(exi['ym'][4:])}월 확정, 관세청 품목별 수출입실적)</h3>")
        h.append("<table><tr><th>품목</th><th class=r>금액</th><th class=r>전년비</th><th class=r>전년비(하루 평균)</th><th class=r>전월비</th><th class=r>전월비(하루 평균)</th><th class=r>전분기비</th><th class=r>연초 누계</th><th class=r>누계 전년비</th><th class=r>누계(하루 평균)</th></tr>")
        for r in exi['rows']:
            h.append(f"<tr><td>{r['item']}</td><td class=r>{r['cur']:,.0f}</td><td class=r>{color_pct(r['yoy'],0)}</td><td class=r>{color_pct(r['dyoy'],0)}</td><td class=r>{color_pct(r['mom'],0)}</td><td class=r>{color_pct(r['dmom'],0)}</td>"
                     f"<td class=r>{color_pct(r['qoq'],0)}</td><td class=r>{r['ytd']:,.0f}</td><td class=r>{color_pct(r['ytd_yoy'],0)}</td><td class=r>{color_pct(r['dytd'],0)}</td></tr>")
        h.append("</table>")
        tr = ' / '.join(f"{r['item']} " + ' '.join(fmt_pct(x, 0) for x in r['trend']) for r in exi['rows'])
        h.append(f"<div class=note>단위 백만 달러. 전분기비는 {exi['q']} 기준. 조업일수는 거래소 개장일로 근사(이번 달 {exi['wd'][0]:.0f}일, 전년 같은 달 {exi['wd'][1]:.0f}일, 전월 {exi['wd'][2]:.0f}일). "
                 f"품목 범위: 화장품 HS 3304(기초·색조 등), 라면 1902.30-1010, 김 마른김 1212.21 + 조미김 2008.99-5010, 전력용 변압기 8504.21~23. "
                 f"올해 월별 전년비 흐름: {tr}</div>")

    # 4. 공시
    h.append(f"<h2>6. 공시 이벤트 ({ev['start']:%m/%d}~{ev['end']:%m/%d}, 유가·코스닥)</h2>")
    h.append("<h3>단일판매·공급계약 (매출액 대비 비율 상위)</h3><table><tr><th>회사</th><th>접수일</th><th class=r>계약금액(억 원)</th><th class=r>매출액 대비</th></tr>")
    for r in ev['supply'][:12]:
        amt = '–' if np.isnan(r['amt']) else f"{r['amt']:,.0f}"
        rat = '–' if np.isnan(r['ratio']) else f"{r['ratio']:.1f}%"
        h.append(f"<tr><td>{r['corp']}</td><td>{r['dt'][4:6]}/{r['dt'][6:]}</td><td class=r>{amt}</td><td class=r>{rat}</td></tr>")
    h.append("</table>")
    def names(lst, n=20):
        return ', '.join(dict.fromkeys(x['corp_name'] for x in lst[:n])) or '없음'
    h.append(f"<h3>자사주 소각 결정</h3><div>{names(ev['burn'])}</div>")
    h.append(f"<h3>영업(잠정)실적</h3><div>{names(ev['prelim'], 40)}</div>")
    h.append(f"<h3>적신호: 유상증자·CB·BW 발행 결정</h3><div>{names(ev['rights'], 40)}</div>")
    h.append("<div class=note>오너 외 임원 장내매수는 특수관계인 대조를 거쳐 다음 판부터 표로 붙입니다.</div>")

    # 5. 가격 신호
    h.append("<h2>7. 오늘 돌파 종목 (참고, 관문 미적용)</h2>")
    h.append("<div class=note>조건: 20일 신고가 + 거래대금이 직전 60일 중앙값의 2배 이상 + 시가총액 5천억 원 이상. 재무·지배구조 관문과 섹터 단계는 아직 적용하지 않았으므로 매수 후보가 아닙니다.</div>")
    h.append("<table><tr><th>종목</th><th>시장</th><th class=r>종가</th><th class=r>등락</th><th class=r>거래대금 배수</th><th class=r>시가총액(조 원)</th></tr>")
    for c, r in bo.iterrows():
        h.append(f"<tr><td>{r['ISU_NM']}</td><td>{r['MKT_NM']}</td><td class=r>{r['TDD_CLSPRC']:,.0f}</td><td class=r>{color_pct(r['FLUC_RT'],2)}</td><td class=r>{r['vr']:.1f}배</td><td class=r>{r['MKTCAP']/1e12:,.1f}</td></tr>")
    h.append("</table>")
    h.append("<h2>데이터와 한계</h2><div class=note>거래소 Open API(일별 종가), 한국투자증권 Open API, OpenDART, 관세청 10일 단위 잠정치. 예시판은 2025년 6월 이후 시세만 사용해 6개월 초과·52주 지표는 기간이 짧을 수 있습니다. 시세는 수정주가가 아니라 액면분할·병합 종목은 이동평균이 일시적으로 틀릴 수 있습니다. 투자자별 매매·ETF 구성종목·증시자금은 한국투자증권 Open API(조회 전용)로 받습니다. 모든 판정 기준은 검증 전 출발값입니다.</div>")
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(h)}</body></html>"
    return html


def to_pdf(html, path_pdf):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(); pg.set_content(html, wait_until='load')
        pg.pdf(path=path_pdf, format='A4', print_background=True, margin={'top': '14mm', 'bottom': '14mm', 'left': '13mm', 'right': '13mm'})
        b.close()


if __name__ == '__main__':
    asof = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else None
    if asof is None:
        asof = pd.Timestamp(max(os.path.basename(f)[:8] for f in glob.glob(os.path.join(DATA, 'stk', '20*', '*.csv.gz'))))
    os.makedirs(OUT, exist_ok=True)
    html = build(asof)
    hp = os.path.join(OUT, f'{asof:%Y%m%d}.html'); open(hp, 'w', encoding='utf-8').write(html)
    pp = os.path.join(OUT, f'{asof:%Y%m%d}.pdf'); to_pdf(html, pp)
    print(pp)
