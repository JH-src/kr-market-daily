"""한국투자증권 Open API 일별 수집(GitHub Actions에서 실행, 조회 전용).

1) 섹터 ETF·추적 ETF 구성종목·비중 -> data/kis/holdings/YYYYMMDD.csv (추적 ETF는 sector 칸이 '추적:<이름>')
4) 위 구성종목 전체의 종목별 투자자 순매수(최근 30거래일, 백만원) -> data/kis/flows/YYYY.csv
2) 코스피·코스닥 투자자별 순매수(일별) -> data/kis/mkt_investor_<시장>.csv (날짜 기준 누적)
3) 증시자금(예탁금·신용잔고 등) -> data/kis/mktfunds.csv (날짜 기준 누적)
"""
import csv, json, os, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'kis')
Q = '/uapi/domestic-stock/v1/quotations/'


def upsert(path, rows, key):
    old = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding='utf-8')):
            old[r[key]] = r
    for r in rows:
        old[r[key]] = r
    if not old:
        return
    cols = list(next(iter(old.values())).keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore')
        w.writeheader()
        for k in sorted(old):
            w.writerow(old[k])


def components(tok, code):
    """ETF 구성종목. 응답이 이어지면(tr_cont F/M) 다음 쪽까지 받는다."""
    out, cont = [], ''
    for _ in range(10):
        q = urllib.parse.urlencode({'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': code, 'FID_COND_SCR_DIV_CODE': '11216'})
        req = urllib.request.Request(f'{K.BASE}/uapi/etfetn/v1/quotations/inquire-component-stock-price?{q}', headers={
            'content-type': 'application/json; charset=utf-8', 'authorization': f'Bearer {tok}',
            'appkey': K.KEY, 'appsecret': K.SEC, 'tr_id': 'FHKST121600C0', 'custtype': 'P', 'tr_cont': cont})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            print('구성종목 오류', code, e); break
        r = json.loads(resp.read())
        out += r.get('output2') or []
        if resp.headers.get('tr_cont') in ('F', 'M') and r.get('output2'):
            cont = 'N'; time.sleep(0.2)
        else:
            break
    return out


FLOW_COLS = {'stck_bsop_date': 'date', 'stck_clpr': 'close', 'acml_tr_pbmn': 'value',
             'frgn_ntby_tr_pbmn': 'frgn', 'orgn_ntby_tr_pbmn': 'orgn', 'prsn_ntby_tr_pbmn': 'prsn',
             'ivtr_ntby_tr_pbmn': 'ivtr', 'pe_fund_ntby_tr_pbmn': 'pefund', 'mrbn_ntby_tr_pbmn': 'pension'}


def flows(tok, codes, today):
    rows = []
    for c in sorted(codes):
        r = K.call(tok, Q + 'investor-trade-by-stock-daily', 'FHPTJ04160001',
                   {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c, 'FID_INPUT_DATE_1': today, 'FID_ORG_ADJ_PRC': '', 'FID_ETC_CLS_CODE': ''})
        for x in r.get('output2') or []:
            if x.get('stck_bsop_date'):
                d = {v: x.get(k, '') for k, v in FLOW_COLS.items()}
                d['code'] = c; d['key'] = d['date'] + c
                rows.append(d)
        time.sleep(0.07)
    by_year = {}
    for d in rows:
        by_year.setdefault(d['date'][:4], []).append(d)
    os.makedirs(os.path.join(OUT, 'flows'), exist_ok=True)
    for y, rs in by_year.items():
        upsert(os.path.join(OUT, 'flows', y + '.csv'), rs, 'key')
    return len(rows)


def main():
    tok = K.token()
    today = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d')
    etfs = {k: v for k, v in json.load(open(os.path.join(ROOT, 'config', 'sector_etf.json'), encoding='utf-8')).items() if not k.startswith('_')}
    watch = {k: v for k, v in json.load(open(os.path.join(ROOT, 'config', 'watch_etf.json'), encoding='utf-8')).items() if not k.startswith('_')}
    etfs.update({'추적:' + k: v for k, v in watch.items()})
    rows = []
    for sec, code in etfs.items():
        comp = components(tok, code)
        for x in comp:
            rows.append({'date': today, 'sector': sec, 'etf': code, 'code': x.get('stck_shrn_iscd', ''),
                         'name': x.get('hts_kor_isnm', ''), 'weight': x.get('etf_cnfg_issu_rlim', ''),
                         'mktcap_eok': x.get('hts_avls', '')})
        if not comp:
            print('구성종목 실패', sec, code)
        time.sleep(0.3)
    os.makedirs(os.path.join(OUT, 'holdings'), exist_ok=True)
    if rows:
        with open(os.path.join(OUT, 'holdings', today + '.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    for mkt, iscd, cls in [('kospi', '0001', 'KSP'), ('kosdaq', '1001', 'KSQ')]:
        r = K.call(tok, Q + 'inquire-investor-daily-by-market', 'FHPTJ04040000',
                   {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': iscd, 'FID_INPUT_DATE_1': today,
                    'FID_INPUT_ISCD_1': cls, 'FID_INPUT_DATE_2': today, 'FID_INPUT_ISCD_2': iscd})
        upsert(os.path.join(OUT, f'mkt_investor_{mkt}.csv'), r.get('output') or [], 'stck_bsop_date')
        time.sleep(0.3)

    r = K.call(tok, Q + 'mktfunds', 'FHKST649100C0', {'FID_INPUT_DATE_1': ''})
    upsert(os.path.join(OUT, 'mktfunds.csv'), r.get('output') or [], 'bsop_date')
    codes = {x['code'] for x in rows if x['code'].isdigit() and len(x['code']) == 6}
    codes |= set(watch.values())          # 추적 ETF 자체의 투자자별 매매도
    n = flows(tok, codes, today)
    print('done', today, len(rows), len(codes), n)


if __name__ == '__main__':
    main()
