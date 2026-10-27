"""공포·탐욕 지수(코스피·코스닥·나스닥) — 2026-10-27 정식판.
각 항목을 최근 250거래일 안의 위치(0~100)로 바꿔 평균한다. 0 = 극단적 공포, 100 = 극단적 탐욕.
- 코스피: 125일선 대비, 60일선 위 종목 비율, 52주 신고가-신저가(5일 평균), 코스피200 변동성지수(50일 평균 대비, 역), 코스피200 옵션 풋/콜 거래량(5일, 역), 주식-국채 20일 수익률 차
- 코스닥: 위와 같되 변동성은 지수 20일 실제 변동성(코스닥 변동성지수 없음), 풋/콜 제외(코스닥150 옵션 거래가 적고 기간이 짧음)
- 나스닥: 125일선 대비, 60일선 위 종목 비율·신고가-신저가(한투 나스닥 보통주, data/fg/us_breadth.csv), VIX(역), 주식-국채(10년 금리 변화로 환산), 하이일드 스프레드(역, 공개 자료가 있는 기간만)
자료: 거래소(data/, archive/), 한투(data/fg/us_breadth.csv), FRED(data/macro/*.csv — 없으면 내려받음).
python3 fear_greed.py → data/fg/fg_daily.csv(지수), data/fg/fg_components.csv(항목)"""
import glob, os, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data'); A = os.path.join(ROOT, 'archive')
OUT = os.path.join(D, 'fg'); os.makedirs(OUT, exist_ok=True)
START = '20100101'
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
pr = lambda s: s.rolling(250, min_periods=200).rank(pct=True) * 100


def rd(name, cols):
    fs = sorted(glob.glob(f'{A}/{name}/*.csv.gz')) + sorted(glob.glob(f'{D}/{name}/20*/*.csv.gz'))
    return pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=lambda c: c in cols) for f in fs])


def iseries(name, nm):
    d = rd(name, ['BAS_DD', 'IDX_NM', 'CLSPRC_IDX']); d = d[d['IDX_NM'].str.strip() == nm]
    s = pd.Series(num(d['CLSPRC_IDX']).values, index=pd.to_datetime(d['BAS_DD']))
    return s[~s.index.duplicated(keep='last')].sort_index().dropna()


def prices(mk):
    fs = sorted(f for f in glob.glob(f'{D}/{mk}/20*/*.csv.gz') if os.path.basename(f)[:8] >= START)
    d = pd.concat([pd.read_csv(f, dtype=str, usecols=lambda c: c in ('BAS_DD', 'ISU_CD', 'ISU_NM', 'TDD_CLSPRC')) for f in fs])
    d = d[~d['ISU_NM'].str.contains('스팩', na=False)]
    w = d.assign(p=num(d['TDD_CLSPRC'])).pivot_table(index='BAS_DD', columns='ISU_CD', values='p')
    w.index = pd.to_datetime(w.index)
    return w


def breadth(w):
    ma = w.rolling(60, min_periods=60).mean(); above = (w > ma).astype(float).where(ma.notna())
    hi, lo = w.rolling(250, min_periods=240).max(), w.rolling(250, min_periods=240).min()
    n = hi.notna().sum(axis=1)
    hl = (((w >= hi) & hi.notna()).sum(axis=1) - ((w <= lo) & lo.notna()).sum(axis=1)) / n.where(n > 100)
    return above.sum(axis=1) / above.notna().sum(axis=1).where(lambda x: x > 100) * 100, hl


def fred(sid):
    f = os.path.join(D, 'macro', f'{sid}.csv')
    try:
        os.makedirs(os.path.dirname(f), exist_ok=True)
        urllib.request.urlretrieve(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2009-01-01', f + '.tmp')
        os.replace(f + '.tmp', f)
    except Exception as e:
        print('FRED 갱신 실패(기존 파일 사용)', sid, e)
    d = pd.read_csv(f); d.columns = ['d', 'v']
    return pd.Series(pd.to_numeric(d['v'], errors='coerce').values, index=pd.to_datetime(d['d'])).dropna()


def korea():
    bd = rd('idx_bond', ['BAS_DD', 'BND_IDX_GRP_NM', 'TOT_EARNG_IDX'])
    k = bd[bd['BND_IDX_GRP_NM'].str.strip() == 'KTB 지수'].drop_duplicates('BAS_DD', keep='last')
    ktb = pd.Series(num(k['TOT_EARNG_IDX']).values, index=pd.to_datetime(k['BAS_DD'])).sort_index()
    vk = iseries('idx_drv', '코스피 200 변동성지수')
    o = rd('opt_summary', ['BAS_DD', 'PROD_NM', 'RGHT', 'VOL'])
    o = o[o['PROD_NM'].str.contains('코스피200|코스피 200', regex=True) & o['PROD_NM'].str.contains('옵션')]
    o = o.drop_duplicates(['BAS_DD', 'PROD_NM', 'RGHT'], keep='last').assign(v=lambda x: num(x['VOL']))
    pc = o.pivot_table(index='BAS_DD', columns='RGHT', values='v', aggfunc='sum'); pc.index = pd.to_datetime(pc.index)
    pcr = (pc['PUT'] / pc['CALL'].replace(0, np.nan)).sort_index()
    res = {}
    for nm, mk, idx in (('코스피', 'stk', ('idx_kospi', '코스피')), ('코스닥', 'ksq', ('idx_kosdaq', '코스닥'))):
        x = iseries(*idx); x = x[x.index >= START]
        ab, hl = breadth(prices(mk).reindex(x.index))
        C = {'모멘텀': pr(x / x.rolling(125, min_periods=125).mean() - 1), '시장 폭': pr(ab), '신고가-신저가': pr(hl.rolling(5).mean()),
             '주식-채권': pr(x.pct_change(20) - ktb.reindex(x.index).ffill().pct_change(20))}
        if nm == '코스피':
            v = vk.reindex(x.index).ffill(limit=3)
            C['변동성(역)'] = pr(-(v / v.rolling(50, min_periods=50).mean()))
            C['풋콜(역)'] = pr(-pcr.reindex(x.index).rolling(5, min_periods=3).mean())
        else:
            rv = x.pct_change().rolling(20).std()
            C['변동성(역)'] = pr(-(rv / rv.rolling(50, min_periods=50).mean()))
        res[nm] = pd.DataFrame(C)
    return res


def nasdaq():
    q = fred('NASDAQCOM'); q = q[q.index >= START]
    vix = fred('VIXCLS').reindex(q.index).ffill(limit=3); y10 = fred('DGS10').reindex(q.index).ffill(limit=3); hy = fred('BAMLH0A0HYM2').reindex(q.index).ffill(limit=3)
    C = {'모멘텀': pr(q / q.rolling(125, min_periods=125).mean() - 1), '변동성(역)': pr(-(vix / vix.rolling(50, min_periods=50).mean())),
         '주식-채권': pr(q.pct_change(20) + y10.diff(20) * 8 / 100), '하이일드(역)': pr(-hy)}
    f = os.path.join(OUT, 'us_breadth.csv')
    if os.path.exists(f):
        b = pd.read_csv(f, index_col=0, parse_dates=True).reindex(q.index)
        ok = b['n52'] > 300
        C['시장 폭'] = pr(b['above60'].where(b['n60'] > 300))
        C['신고가-신저가'] = pr(((b['new_hi'] - b['new_lo']) / b['n52']).where(ok).rolling(5, min_periods=3).mean())
    return pd.DataFrame(C)


def main():
    comp = korea(); comp['나스닥'] = nasdaq()
    need = {'코스피': 4, '코스닥': 4, '나스닥': 3}
    fg = pd.DataFrame({nm: c.mean(axis=1).where(c.notna().sum(axis=1) >= need[nm]) for nm, c in comp.items()})
    fg = fg[fg.index >= '2011-06-01'].round(1)
    fg.to_csv(os.path.join(OUT, 'fg_daily.csv'))
    pd.concat(comp, axis=1).round(1).loc['2011-06-01':].to_csv(os.path.join(OUT, 'fg_components.csv'))
    for nm, c in comp.items():
        s = fg[nm].dropna()
        print(nm, s.index.min().date(), s.index.max().date(), '최근', s.iloc[-1], '항목수', int(c.iloc[-1].notna().sum()), c.iloc[-1].round(0).to_dict())


if __name__ == '__main__':
    main()
