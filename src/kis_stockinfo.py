"""종목별 업종 분류(거래소 지수업종 대·중·소, 표준산업분류명)를 한투에서 받는다(조회 전용).
결과: data/kis/stockinfo.csv"""
import csv, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
codes = [l.split(',')[0] for l in open(os.path.join(ROOT, 'config', 'universe_market.csv'), encoding='utf-8')]
tok = K.token()
lock = threading.Lock(); last = [0.0]
F = ['prdt_abrv_name', 'std_idst_clsf_cd', 'std_idst_clsf_cd_name', 'idx_bztp_lcls_cd_name', 'idx_bztp_mcls_cd_name', 'idx_bztp_scls_cd_name', 'mket_id_cd', 'lstg_dt']


def one(c):
    with lock:
        w = last[0] + 1 / 15 - time.time()
        if w > 0:
            time.sleep(w)
        last[0] = time.time()
    r = K.call(tok, '/uapi/domestic-stock/v1/quotations/search-stock-info', 'CTPF1002R', {'PRDT_TYPE_CD': '300', 'PDNO': c})
    o = r.get('output') or {}
    return {'code': c, **{k: o.get(k, '') for k in F}}


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(one, codes))
with open(os.path.join(ROOT, 'data', 'kis', 'stockinfo.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, ['code'] + F); w.writeheader(); w.writerows(rows)
print('stockinfo', len(rows), sum(1 for r in rows if r['idx_bztp_scls_cd_name']))
