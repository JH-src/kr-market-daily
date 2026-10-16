"""국내 주식형 ETF 전체의 구성종목(비중 상위 30)을 받아 테마 분류에 쓴다(조회 전용).
결과: data/kis/etf_all/<YYYYMMDD>.csv (etf, etf_name, code, name, weight)"""
import csv, os, time
import kis_probe as K
from kis_daily import components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'kis', 'etf_all')
os.makedirs(OUT, exist_ok=True)
etfs = [l.rstrip('\n').split(',', 1) for l in open(os.path.join(ROOT, 'config', 'etf_all_equity.csv'), encoding='utf-8')]
tok = K.token()
today = time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600))
rows = []
for code, name in etfs:
    for x in components(tok, code):
        rows.append({'etf': code, 'etf_name': name, 'code': x.get('stck_shrn_iscd', ''), 'name': x.get('hts_kor_isnm', ''), 'weight': x.get('etf_cnfg_issu_rlim', '')})
    time.sleep(0.08)
with open(os.path.join(OUT, today + '.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, ['etf', 'etf_name', 'code', 'name', 'weight']); w.writeheader(); w.writerows(rows)
print('etf_all', len(etfs), len(rows))
