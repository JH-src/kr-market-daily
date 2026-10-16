"""일별 리포트 생성 (예시판). 결과: reports/YYYYMMDD.html, .pdf

사용: python3 src/report.py [기준일 YYYYMMDD]
필요 환경변수: DART_KEY
"""
import base64, glob, gzip, io, json, os, re, sys, time, urllib.request, zipfile
from datetime import date, datetime, timedelta
import numpy as np
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
.up { color:#c0392b; } .dn { color:#1f5fbf; } .mut { color:#9ca3af; }
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


def build(asof):
    px = stock_panel()
    reg = market_regime(px, asof)
    mac, glo = macro(asof)
    sec = sector_table(asof, reg['kospi'])
    ex = exports()
    ev = dart_events(asof)
    bo = breakout(px, asof)
    h = []
    h.append(f"<h1>국내 시장 일일 점검</h1><div class='sub'>{asof:%Y년 %m월 %d일} 장 마감 기준 · 예시판(검증 전 출발값 적용)</div>")
    # 요약
    lead = [s for s in sec if s['stage'] == '주도'][:4]
    early = [s for s in sec if s['stage'] == '초기'][:4]
    hot = [s for s in sec if s['stage'] == '과열 주의']
    k = reg['idx_rows'][0]
    h.append("<div class='box'><b>오늘의 요약</b><br>")
    h.append(f"· 시장 국면: <b>{reg['regime']}</b> — 코스피 {k['close']:,.0f}({color_pct(k['d1'])}), 200일선 대비 {fmt_pct(k['ma200'])}, 60일선 위 종목 {reg['breadth_last']:.0f}%, 변동성지수 {reg['vk_last']:.1f}(직전 50일 평균 {reg['vk_avg']:.1f})<br>")
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
    h.append("<div class=note>산업 섹터는 KRX 산업지수, 조선·방산·2차전지·전력기기·소부장은 순자산이 가장 큰 국내 테마 ETF의 순자산가치로 계산했습니다. 이번 예시판의 단계는 가격 축만 반영했고, 변화 축(수출·잠정실적·공급계약)과 섹터 내 확산도는 섹터 분류표 확정 후 붙입니다.</div>")

    # 3. 수출
    wd = ex['wd']
    h.append("<h2>3. 수출 (관세청 10일 단위 잠정치)</h2>")
    h.append(f"<div class=note>최신: {ex['ym'][:4]}년 {ex['ym'][4:]}월 {ex['per']}일 · 단위 백만 달러 · 조업일수는 거래소 개장일로 근사(이번 {wd[0]:.0f}일, 전년 같은 기간 {wd[1]:.0f}일, 전월 같은 기간 {wd[2]:.0f}일). 연초 누계와 전분기 대비는 월 확정치({ex['last_full']}까지, 분기는 {ex['q_note']}) 기준.</div>")
    h.append("<table><tr><th>품목</th><th class=r>금액</th><th class=r>전년비</th><th class=r>전년비(하루 평균)</th><th class=r>전월비</th><th class=r>전월비(하루 평균)</th><th class=r>전분기비</th><th class=r>연초 누계</th><th class=r>누계 전년비</th></tr>")
    for r in ex['rows']:
        h.append(f"<tr><td>{r['item']}</td><td class=r>{r['cur']:,.0f}</td><td class=r>{color_pct(r['yoy'],0)}</td><td class=r>{color_pct(r['dyoy'],0)}</td><td class=r>{color_pct(r['mom'],0)}</td><td class=r>{color_pct(r['dmom'],0)}</td>"
                 f"<td class=r>{color_pct(r.get('qoq', np.nan),0)}</td><td class=r>{r['ytd']:,.0f}</td><td class=r>{color_pct(r['ytd_yoy'],0)}</td></tr>")
    h.append("</table>")
    if wd[0] and wd[1] and abs(wd[0] - wd[1]) >= 2:
        h.append(f"<div class=note>조업일수 차이가 커서(이번 {wd[0]:.0f}일 vs 전년 {wd[1]:.0f}일) 단순 전년비는 왜곡돼 있습니다. 하루 평균 기준을 우선 보세요.</div>")
    h.append(f"<img src='{chart_exports(ex)}'>")

    # 4. 공시
    h.append(f"<h2>4. 공시 이벤트 ({ev['start']:%m/%d}~{ev['end']:%m/%d}, 유가·코스닥)</h2>")
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
    h.append("<h2>5. 오늘 돌파 종목 (참고, 관문 미적용)</h2>")
    h.append("<div class=note>조건: 20일 신고가 + 거래대금이 직전 60일 중앙값의 2배 이상 + 시가총액 5천억 원 이상. 재무·지배구조 관문과 섹터 단계는 아직 적용하지 않았으므로 매수 후보가 아닙니다.</div>")
    h.append("<table><tr><th>종목</th><th>시장</th><th class=r>종가</th><th class=r>등락</th><th class=r>거래대금 배수</th><th class=r>시가총액(조 원)</th></tr>")
    for c, r in bo.iterrows():
        h.append(f"<tr><td>{r['ISU_NM']}</td><td>{r['MKT_NM']}</td><td class=r>{r['TDD_CLSPRC']:,.0f}</td><td class=r>{color_pct(r['FLUC_RT'],2)}</td><td class=r>{r['vr']:.1f}배</td><td class=r>{r['MKTCAP']/1e12:,.1f}</td></tr>")
    h.append("</table>")
    h.append("<h2>데이터와 한계</h2><div class=note>거래소 Open API(일별 종가), OpenDART, 관세청 10일 단위 잠정치. 예시판은 2025년 6월 이후 시세만 사용해 6개월 초과·52주 지표는 기간이 짧을 수 있습니다. 시세는 수정주가가 아니라 액면분할·병합 종목은 이동평균이 일시적으로 틀릴 수 있습니다. 투자자별 매매·공매도 자료는 없습니다. 모든 판정 기준은 검증 전 출발값입니다.</div>")
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
