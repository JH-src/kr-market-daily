"""DART 실적·공시 점검(원페이지 추천용).

- quarterly(asof, codes): 후보 종목의 최근 분기 매출·영업이익(3개월, 연결 우선) — 다중회사 주요계정 API, 100개사씩, 캐시
- corp_events(codes, asof): 후보 종목별 최근 90일 공시에서 증자·CB·BW(관문), 공급계약·소각·잠정실적(근거)
"""
import csv, io, json, os, re, time, urllib.request, zipfile
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'dart')
RC = {'11013': 1, '11012': 2, '11014': 3, '11011': 4}   # 보고서 코드 -> 분기


def _get(url, tries=4):
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read())
        except Exception:
            time.sleep(3 * (i + 1))
    return {}


def corp_map():
    p = os.path.join(OUT, 'corpcode.csv')
    if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 7 * 86400:
        z = urllib.request.urlopen(f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={os.environ['DART_KEY']}", timeout=120).read()
        x = zipfile.ZipFile(io.BytesIO(z)).read('CORPCODE.xml').decode('utf-8')
        rows = []
        for b in re.findall(r'<list>(.*?)</list>', x, re.S):
            sc = re.search(r'<stock_code>(.*?)</stock_code>', b).group(1).strip()
            if sc:
                rows.append({'stock_code': sc, 'corp_code': re.search(r'<corp_code>(.*?)</corp_code>', b).group(1),
                             'corp_name': re.search(r'<corp_name>(.*?)</corp_name>', b).group(1)})
        os.makedirs(OUT, exist_ok=True)
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, ['stock_code', 'corp_code', 'corp_name']); w.writeheader(); w.writerows(rows)
    return {r['stock_code']: r['corp_code'] for r in csv.DictReader(open(p, encoding='utf-8'))}


def _num(s):
    s = (s or '').replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def fetch_period(year, rc, cmap, codes, refresh=False):
    """한 분기(보고서)의 매출·영업이익. 필요한 종목만 100개씩 받아 캐시에 덧붙인다. 연결(CFS) 우선, 없으면 별도(OFS).
    한 번 조회했는데 자료가 없던 종목은 '없음'으로 기록해 다시 부르지 않는다(refresh=True면 다시)."""
    p = os.path.join(OUT, 'fin', f'{year}_{rc}.csv')
    cols = ['stock_code', 'fs', 'rcept_no', 'rev', 'rev_py', 'op', 'op_py']
    have = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding='utf-8')):
            have[r['stock_code']] = r
    need = [c for c in codes if c in cmap and (refresh or c not in have)]
    if need:
        key = os.environ['DART_KEY']
        from concurrent.futures import ThreadPoolExecutor
        parts = [need[i:i + 100] for i in range(0, len(need), 100)]
        get = lambda part: (part, _get(f"https://opendart.fss.or.kr/api/fnlttMultiAcnt.json?crtfc_key={key}&corp_code={','.join(cmap[c] for c in part)}&bsns_year={year}&reprt_code={rc}"))
        with ThreadPoolExecutor(int(os.environ.get('DART_THREADS', '1'))) as ex:
            results = list(ex.map(get, parts))
        for part, d in results:
            if d.get('status') not in ('000', '013'):
                print('DART 실적 조회 실패', year, rc, d.get('status'), d.get('message')); continue
            res = {}
            for r in d.get('list', []):
                if r.get('account_nm') not in ('매출액', '영업이익') or r.get('sj_div') != 'IS':
                    continue
                e = res.setdefault((r['stock_code'], r['fs_div']), {'stock_code': r['stock_code'], 'fs': r['fs_div'], 'rcept_no': r.get('rcept_no', '')})
                tag = 'rev' if r['account_nm'] == '매출액' else 'op'
                e[tag] = _num(r.get('thstrm_amount'))        # 해당 분기 3개월(사업보고서는 연간)
                e[tag + '_py'] = _num(r.get('frmtrm_amount'))    # 전년 같은 분기
            for c in part:
                have[c] = res.get((c, 'CFS')) or res.get((c, 'OFS')) or {'stock_code': c, 'fs': '없음'}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader(); w.writerows(have.values())
    return p


def periods_before(asof, n=3):
    """asof 시점에 제출 기한이 지난 최근 분기 n개 (1분기 5/15, 반기 8/14, 3분기 11/14, 사업 3/31 기준)."""
    out = []
    y = asof.year
    cands = []
    for yy in (y, y - 1, y - 2):
        cands += [(yy, '11014', f'{yy}1114'), (yy, '11012', f'{yy}0814'), (yy, '11013', f'{yy}0515'), (yy - 1, '11011', f'{yy}0331')]
    for yy, rc, due in sorted(cands, key=lambda x: x[2], reverse=True):
        if due <= asof.strftime('%Y%m%d'):
            out.append((yy, rc))
        if len(out) == n:
            break
    return out


def quarterly(asof, codes):
    cmap = corp_map()
    ps = periods_before(asof, 3)
    return ps, {k: fetch_period(k[0], k[1], cmap, codes) for k in ps}


def corp_events(codes, asof):
    """후보 종목별 최근 90일 공시 요약."""
    key = os.environ['DART_KEY']
    cmap = corp_map()
    a = asof.strftime('%Y%m%d')
    b90 = (asof - timedelta(days=90)).strftime('%Y%m%d')
    b30 = (asof - timedelta(days=30)).strftime('%Y%m%d')
    out = {}
    for sc in codes:
        cc = cmap.get(sc)
        if not cc:
            continue
        d = _get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&corp_code={cc}&bgn_de={b90}&end_de={a}&page_count=100")
        e = {'rights': [], 'supply': [], 'burn': [], 'prelim': []}
        for x in d.get('list', []):
            nm, dt = x['report_nm'], x['rcept_dt']
            if re.search(r'유상증자결정|전환사채권발행결정|신주인수권부사채권발행결정', nm) and '철회' not in nm:
                e['rights'].append((dt, nm.strip()))
            elif dt >= b30 and '단일판매' in nm and '해지' not in nm:
                e['supply'].append((dt, nm.strip()))
            elif dt >= b30 and '주식소각결정' in nm:
                e['burn'].append((dt, nm.strip()))
            elif dt >= b30 and re.search(r'영업\(잠정\)실적', nm):
                e['prelim'].append((dt, nm.strip(), x['rcept_no']))
        out[sc] = e
        time.sleep(0.1)
    return out


EXT = {('IS', '당기순이익(손실)'): 'ni', ('IS', '당기순이익'): 'ni', ('BS', '자본총계'): 'eq', ('BS', '부채총계'): 'debt'}


def fetch_ext(year, rc, cmap, codes):
    """밸류·재무 건전성용: 당기순이익(해당 분기 3개월), 자본총계·부채총계(분기 말) -> data/dart/fin_ext/"""
    p = os.path.join(OUT, 'fin_ext', f'{year}_{rc}.csv')
    if os.path.exists(p):
        return p
    key = os.environ['DART_KEY']
    from concurrent.futures import ThreadPoolExecutor
    need = [c for c in codes if c in cmap]
    parts = [need[i:i + 100] for i in range(0, len(need), 100)]
    get = lambda part: (part, _get(f"https://opendart.fss.or.kr/api/fnlttMultiAcnt.json?crtfc_key={key}&corp_code={','.join(cmap[c] for c in part)}&bsns_year={year}&reprt_code={rc}"))
    with ThreadPoolExecutor(int(os.environ.get('DART_THREADS', '4'))) as ex:
        results = list(ex.map(get, parts))
    res = {}
    for part, d in results:
        for r in d.get('list', []):
            k = EXT.get((r.get('sj_div'), (r.get('account_nm') or '').replace(' ', '')))
            if not k:
                continue
            e = res.setdefault((r['stock_code'], r['fs_div']), {'stock_code': r['stock_code'], 'fs': r['fs_div'], 'rcept_no': r.get('rcept_no', '')})
            e[k] = _num(r.get('thstrm_amount'))
    best = {}
    for (sc, fs), e in res.items():
        if sc not in best or fs == 'CFS':
            best[sc] = e
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['stock_code', 'fs', 'rcept_no', 'ni', 'eq', 'debt'], extrasaction='ignore')
        w.writeheader(); w.writerows(best.values())
    return p
