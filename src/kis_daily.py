"""한국투자증권 Open API 일별 수집(GitHub Actions에서 실행, 조회 전용).

1) 섹터 ETF 구성종목·비중 -> data/kis/holdings/YYYYMMDD.csv
2) 코스피·코스닥 투자자별 순매수(일별) -> data/kis/mkt_investor_<시장>.csv (날짜 기준 누적)
3) 증시자금(예탁금·신용잔고 등) -> data/kis/mktfunds.csv (날짜 기준 누적)
"""
import csv, json, os, time
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


def main():
    tok = K.token()
    today = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d')
    etfs = {k: v for k, v in json.load(open(os.path.join(ROOT, 'config', 'sector_etf.json'), encoding='utf-8')).items() if not k.startswith('_')}
    rows = []
    for sec, code in etfs.items():
        r = K.call(tok, '/uapi/etfetn/v1/quotations/inquire-component-stock-price', 'FHKST121600C0',
                   {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': code, 'FID_COND_SCR_DIV_CODE': '11216'})
        for x in r.get('output2') or []:
            rows.append({'date': today, 'sector': sec, 'etf': code, 'code': x.get('stck_shrn_iscd', ''),
                         'name': x.get('hts_kor_isnm', ''), 'weight': x.get('etf_cnfg_issu_rlim', ''),
                         'mktcap_eok': x.get('hts_avls', '')})
        if not r.get('output2'):
            print('구성종목 실패', sec, code, r.get('msg1') or r.get('body'))
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
    print('done', today, len(rows))


if __name__ == '__main__':
    main()
