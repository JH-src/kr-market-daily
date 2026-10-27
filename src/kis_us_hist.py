"""나스닥 보통주 일봉(한투 해외주식, 수정주가, 조회 전용) → 미국 시장 폭·신고가-신저가 일별 시계열.
python3 kis_us_hist.py full    : 2010년부터 전체(처음 1회, 1~2시간)  → /tmp/us_close.csv.gz(원자료) + data/fg/us_breadth.csv
python3 kis_us_hist.py recent  : 최근 약 400거래일만 받아 최근 값 갱신(매일)
한계: 한투 종목 파일은 현재 상장 종목만 있어 과거로 갈수록 상장폐지 종목이 빠짐(생존 편향)."""
import csv, gzip, io, os, sys, threading, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'fg'); os.makedirs(OUT, exist_ok=True)
lock = threading.Lock(); last = [0.0]


def throttle(rate=15):
    with lock:
        w = last[0] + 1 / rate - time.time()
        if w > 0:
            time.sleep(w)
        last[0] = time.time()


def symbols():
    b = urllib.request.urlopen('https://new.real.download.dws.co.kr/common/master/nasmst.cod.zip', timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(b)); rows = z.read(z.namelist()[0]).decode('cp949', 'ignore').splitlines()
    out = []
    bad = ('WARRANT', ' UNIT', 'RIGHT', 'ACQUISITION', ' WT', 'SPAC')
    for r in rows:
        f = r.split('\t')
        if len(f) < 9 or f[8].strip() != '2':          # 2 = 주식(3 = ETF 등)
            continue
        sym, nm = f[4].strip(), f[7].upper()
        if not sym.isalpha() or any(x in nm for x in bad):
            continue
        out.append(sym)
    return sorted(set(out))


def fetch(tok, sym, start):
    rows, bymd = {}, ''
    for _ in range(60):
        r = {}
        for k in range(5):                                   # 끊김·일시 오류는 잠깐 쉬고 다시
            throttle()
            try:
                r = K.call(tok, '/uapi/overseas-price/v1/quotations/dailyprice', 'HHDFS76240000',
                           {'AUTH': '', 'EXCD': 'NAS', 'SYMB': sym, 'GUBN': '0', 'BYMD': bymd, 'MODP': '1'})
                if r.get('http_error') or r.get('rt_cd') not in (None, '0'):
                    time.sleep(1 + k); continue
                break
            except Exception:
                time.sleep(2 + 2 * k)
        o = r.get('output2') or []
        ds = [(x.get('xymd'), x.get('clos')) for x in o if x.get('xymd') and x.get('clos')]
        if not ds:
            break
        for d, c in ds:
            rows[d] = c
        mn = min(d for d, _ in ds)
        if len(ds) < 100 or mn <= start:
            break
        bymd = (pd.Timestamp(mn) - pd.Timedelta(days=1)).strftime('%Y%m%d')
    return sym, rows


def breadth(px):
    ma = px.rolling(60, min_periods=60).mean()
    above = (px > ma).astype(float).where(ma.notna())
    hi = px.rolling(250, min_periods=240).max(); lo = px.rolling(250, min_periods=240).min()
    n = hi.notna().sum(axis=1)
    return pd.DataFrame({'above60': above.sum(axis=1) / above.notna().sum(axis=1) * 100, 'n60': above.notna().sum(axis=1),
                         'new_hi': ((px >= hi) & hi.notna()).sum(axis=1), 'new_lo': ((px <= lo) & lo.notna()).sum(axis=1), 'n52': n})


def main(mode):
    syms = symbols(); print('종목', len(syms), flush=True)
    start = '20100101' if mode == 'full' else (pd.Timestamp.today() - pd.Timedelta(days=600)).strftime('%Y%m%d')
    tok = K.token()
    data = {}
    with ThreadPoolExecutor(8) as ex:
        for i, (s, r) in enumerate(ex.map(lambda s: fetch(tok, s, start), syms)):
            if r:
                data[s] = r
            if i % 300 == 0:
                print(i, len(data), flush=True)
    px = pd.DataFrame({s: pd.Series(r) for s, r in data.items()})
    px.index = pd.to_datetime(px.index); px = px.sort_index().apply(pd.to_numeric, errors='coerce')
    px = px[px.index >= pd.Timestamp(start)]
    print('기간', px.index.min().date(), px.index.max().date(), '종목', px.shape[1], flush=True)
    if mode == 'full':
        px.stack().rename('close').to_csv('/tmp/us_close.csv.gz', compression='gzip')
    b = breadth(px)
    f = os.path.join(OUT, 'us_breadth.csv')
    if mode != 'full' and os.path.exists(f):
        old = pd.read_csv(f, index_col=0, parse_dates=True)
        b = b[b['n52'] >= 0.8 * old['n52'].tail(20).median()]          # 최근 창에서 250일 신고가가 계산되는 날만 덮어씀
        b = pd.concat([old[~old.index.isin(b.index)], b]).sort_index()
    b.round(3).to_csv(f)
    print(b.tail(5), flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'recent')
