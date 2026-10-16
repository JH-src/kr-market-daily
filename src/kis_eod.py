"""장 마감 후 한투로 전 종목 당일 시세를 받아 거래소 일별 파일과 같은 형식으로 저장(조회 전용).
결과: data/kis/eod/<YYYYMMDD>.csv (BAS_DD, ISU_CD, MKT_NM, TDD_CLSPRC, FLUC_RT, TDD_OPNPRC, TDD_HGPRC, TDD_LWPRC, ACC_TRDVOL, ACC_TRDVAL, MKTCAP, LIST_SHRS)
      data/kis/eod/<YYYYMMDD>_idx.json (코스피·코스닥 지수)
"""
import csv, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'kis', 'eod')
lock = threading.Lock(); last = [0.0]


def throttle(rate=15):
    with lock:
        w = last[0] + 1 / rate - time.time()
        if w > 0:
            time.sleep(w)
        last[0] = time.time()


def main():
    os.makedirs(OUT, exist_ok=True)
    mk = {}
    for line in open(os.path.join(ROOT, 'config', 'universe_market.csv'), encoding='utf-8'):
        c, m = line.strip().split(',')[:2]
        mk[c] = m
    tok = K.token()
    today = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600))

    def one(c):
        for _ in range(3):
            throttle()
            r = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-price', 'FHKST01010100', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c})
            o = r.get('output') or {}
            if o.get('stck_prpr'):
                # 현재가는 장 마감 뒤 시간외 단일가로 바뀌므로, 종가는 일봉(정규장 종가)에서 따로 받는다
                throttle()
                d = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice', 'FHKST03010100',
                           {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c, 'FID_INPUT_DATE_1': today, 'FID_INPUT_DATE_2': today,
                            'FID_PERIOD_DIV_CODE': 'D', 'FID_ORG_ADJ_PRC': '1'})
                bar = next((x for x in (d.get('output2') or []) if x.get('stck_bsop_date') == today), None)
                if not bar:
                    return None
                px, sh = float(bar['stck_clpr']), float(o.get('lstn_stcn') or 0)
                return {'BAS_DD': today, 'ISU_CD': c, 'MKT_NM': mk.get(c, ''), 'TDD_CLSPRC': bar['stck_clpr'], 'FLUC_RT': bar.get('prdy_ctrt', o.get('prdy_ctrt', '')),
                        'TDD_OPNPRC': bar.get('stck_oprc', ''), 'TDD_HGPRC': bar.get('stck_hgpr', ''), 'TDD_LWPRC': bar.get('stck_lwpr', ''),
                        'ACC_TRDVOL': bar.get('acml_vol', ''), 'ACC_TRDVAL': bar.get('acml_tr_pbmn', ''), 'MKTCAP': f'{px * sh:.0f}', 'LIST_SHRS': o.get('lstn_stcn', '')}
            time.sleep(1)
        return None

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, list(mk)) if r]
    with open(os.path.join(OUT, today + '.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    idx = {}
    for nm, code in [('코스피', '0001'), ('코스닥', '1001')]:
        r = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-index-price', 'FHPUP02100000', {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': code})
        o = r.get('output') or {}
        idx[nm] = {'close': o.get('bstp_nmix_prpr'), 'chg': o.get('bstp_nmix_prdy_ctrt'), 'open': o.get('bstp_nmix_oprc'),
                   'high': o.get('bstp_nmix_hgpr'), 'low': o.get('bstp_nmix_lwpr')}
    json.dump(idx, open(os.path.join(OUT, today + '_idx.json'), 'w'), ensure_ascii=False)
    print('eod', today, len(rows), len(mk), round(time.time() - t0), idx)


if __name__ == '__main__':
    main()
