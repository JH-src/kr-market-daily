"""거래소·관세청 데이터 수집.

인증키는 저장소에 두지 않는다. 실행 전 환경변수로 넣는다.
  KRX_KEY      한국거래소 Open API 인증키
  DATA_GO_KEY  공공데이터포털 인증키(관세청 10일 수출)
  DART_KEY     OpenDART 인증키

저장 형식: data/<데이터셋>/<연도>/<YYYYMMDD>.csv.gz (거래소, 일별 원본 그대로)
          data/customs/export10d.csv (관세청, 전체 재수집해 덮어씀)
휴장일은 빈 응답이라 파일을 만들지 않고 data/_calendar/empty_days.txt에 기록한다.
"""
import csv, gzip, io, json, os, sys, time, urllib.parse, urllib.request, re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
KRX = 'https://data-dbg.krx.co.kr/svc/apis/'

# 매일 적재하는 거래소 데이터셋. 옵션·ELW는 행 수가 많아 일별 요약만 따로 만든다(원본 미적재).
KRX_SETS = {
    'stk': 'sto/stk_bydd_trd', 'ksq': 'sto/ksq_bydd_trd', 'knx': 'sto/knx_bydd_trd',
    'stk_base': 'sto/stk_isu_base_info', 'ksq_base': 'sto/ksq_isu_base_info',
    'sr': 'sto/sr_bydd_trd', 'sw': 'sto/sw_bydd_trd',
    'idx_kospi': 'idx/kospi_dd_trd', 'idx_kosdaq': 'idx/kosdaq_dd_trd', 'idx_krx': 'idx/krx_dd_trd',
    'idx_drv': 'idx/drvprod_dd_trd', 'idx_bond': 'idx/bon_dd_trd',
    'etf': 'etp/etf_bydd_trd', 'etn': 'etp/etn_bydd_trd',
    'fut': 'drv/fut_bydd_trd', 'fut_stk': 'drv/eqsfu_stk_bydd_trd', 'fut_ksq': 'drv/eqkfu_ksq_bydd_trd',
}
OPT_SETS = {'opt': 'drv/opt_bydd_trd', 'opt_stk': 'drv/eqsop_bydd_trd'}


def krx_get(ep, d, tries=4):
    key = os.environ['KRX_KEY']
    req = urllib.request.Request(KRX + ep + '?basDd=' + d, headers={'AUTH_KEY': key})
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read()).get('OutBlock_1', [])
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def save_rows(name, d, rows):
    path = os.path.join(DATA, name, d[:4], d + '.csv.gz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def opt_summary(d):
    """옵션 원본 대신 하루 요약: 상품별 콜·풋 거래량, 거래대금, 미결제약정, 거래량 가중 내재변동성."""
    out = []
    for name, ep in OPT_SETS.items():
        agg = {}
        for r in krx_get(ep, d):
            k = (r['PROD_NM'], r['RGHT_TP_NM'])
            a = agg.setdefault(k, {'vol': 0.0, 'val': 0.0, 'oi': 0.0, 'iv_w': 0.0})
            f = lambda x: float(x) if x not in ('', '-') else 0.0
            v = f(r['ACC_TRDVOL']); a['vol'] += v; a['val'] += f(r['ACC_TRDVAL']); a['oi'] += f(r['ACC_OPNINT_QTY'])
            a['iv_w'] += f(r['IMP_VOLT']) * v
        for (prod, right), a in agg.items():
            out.append({'BAS_DD': d, 'SET': name, 'PROD_NM': prod, 'RGHT': right, 'VOL': a['vol'], 'VAL': a['val'],
                        'OI': a['oi'], 'IV_VOLW': round(a['iv_w'] / a['vol'], 4) if a['vol'] else ''})
    return out


def collect_day(d):
    """하루치 수집. 휴장일이면 False."""
    first = krx_get(KRX_SETS['stk'], d)
    if not first:
        return False
    save_rows('stk', d, first)
    def one(item):
        name, ep = item
        rows = krx_get(ep, d)
        if rows:
            save_rows(name, d, rows)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, [(n, e) for n, e in KRX_SETS.items() if n != 'stk']))
    s = opt_summary(d)
    if s:
        save_rows('opt_summary', d, s)
    return True


def have(d):
    return os.path.exists(os.path.join(DATA, 'stk', d[:4], d + '.csv.gz'))


def empty_days():
    p = os.path.join(DATA, '_calendar', 'empty_days.txt')
    return set(open(p).read().split()) if os.path.exists(p) else set()


def mark_empty(d):
    p = os.path.join(DATA, '_calendar', 'empty_days.txt')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a') as f:
        f.write(d + '\n')


def krx_range(start, end, budget_sec=None):
    t0 = time.time(); skip = empty_days(); n = 0
    d = start
    while d <= end:
        s = d.strftime('%Y%m%d')
        if d.weekday() < 5 and not have(s) and s not in skip:
            if collect_day(s):
                n += 1
            elif d < date.today():
                mark_empty(s)
            if budget_sec and time.time() - t0 > budget_sec:
                print('budget reached at', s); break
        d += timedelta(days=1)
    print('collected days:', n)


def customs():
    key = os.environ['DATA_GO_KEY']
    names = ['전체', '반도체', '철강제품', '승용차', '석유제품', '무선통신기기', '선박', '자동차부품', '컴퓨터주변기기', '정밀기기', '가전제품']
    rows = []
    y = date.today().year
    for a, b in [(f'{y-9}01', f'{y-5}12'), (f'{y-4}01', f'{y}12')]:
        u = 'https://apis.data.go.kr/1220000/prlstMmUtPrviExpAcrs/getPrlstMmUtPrviExpAcrs?' + urllib.parse.urlencode(
            {'serviceKey': key, 'strtYymm': a, 'endYymm': b})
        for i in range(4):
            try:
                x = urllib.request.urlopen(u, timeout=60).read().decode('utf-8', 'ignore'); break
            except Exception:
                if i == 3:
                    raise
                time.sleep(5 * (i + 1))
        for it in re.findall(r'<item>(.*?)</item>', x, re.S):
            g = lambda t: (re.search(f'<{t}>(.*?)</{t}>', it) or [None, ''])[1].strip()
            r = {'yymm': g('priodMon'), 'period': g('priodDt')}
            for i, n in enumerate(names):
                r[n] = g('itemUsdAmt%02d' % i).replace(',', '')
            rows.append(r)
    p = os.path.join(DATA, 'customs', 'export10d.csv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('customs rows:', len(rows))


def customs_items(years=10):
    """관세청 품목별 수출입실적(월). config/export_items.json의 품목을 HS 코드로 받아 합산 -> data/customs/items_monthly.csv"""
    key = os.environ['DATA_GO_KEY']
    cfg = {k: v for k, v in json.load(open(os.path.join(ROOT, 'config', 'export_items.json'), encoding='utf-8')).items() if not k.startswith('_')}
    y = date.today().year
    acc = {}
    for item, codes in cfg.items():
        for code in codes:
            for yy in range(y - years + 1, y + 1):
                u = 'https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList?' + urllib.parse.urlencode(
                    {'serviceKey': key, 'strtYymm': f'{yy}01', 'endYymm': f'{yy}12', 'hsSgn': code})
                x = ''
                for i in range(5):
                    try:
                        x = urllib.request.urlopen(u, timeout=120).read().decode('utf-8', 'ignore'); break
                    except Exception:
                        time.sleep(5 * (i + 1))
                if '<resultCode>00' not in x:
                    print('품목 실패', item, code, yy, x[:200]); continue
                for it in re.findall(r'<item>(.*?)</item>', x, re.S):
                    g = lambda t: (re.search(f'<{t}>(.*?)</{t}>', it) or [None, ''])[1].strip()
                    hs, ym = g('hsCode'), g('year').replace('.', '')
                    if not hs.startswith(code) or len(ym) != 6:
                        continue   # 총계 행 제외
                    k = (ym, item)
                    acc[k] = acc.get(k, 0) + float(g('expDlr') or 0)
    rows = [{'yymm': ym, 'item': it, 'exp_usd': f'{v:.0f}'} for (ym, it), v in sorted(acc.items())]
    p = os.path.join(DATA, 'customs', 'items_monthly.csv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['yymm', 'item', 'exp_usd']); w.writeheader(); w.writerows(rows)
    print('items rows:', len(rows))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'customs':
        customs()
    elif cmd == 'customs_items':
        customs_items()
    elif cmd == 'krx':
        s = date.fromisoformat(sys.argv[2]); e = date.fromisoformat(sys.argv[3])
        budget = float(sys.argv[4]) if len(sys.argv) > 4 else None
        krx_range(s, e, budget)
