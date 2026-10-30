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
    now = time.gmtime(time.time() + 9 * 3600)
    if len(sys.argv) > 1:
        today = sys.argv[1]
    elif (now.tm_hour, now.tm_min) < (15, 35):
        # 예약 실행이 늦어져 자정을 넘기면 전날 시세가 오늘 날짜로 저장되던 문제(2026-10-30 발견) 방지
        print('장 마감 전(또는 자정 이후 지연 실행)이라 건너뜀'); return
    else:
        today = time.strftime('%Y%m%d', now)

    def one(c):
        for _ in range(3):
            throttle()
            r = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-price', 'FHKST01010100', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c})
            o = r.get('output') or {}
            if o.get('stck_prpr'):
                # 정규장 종가: 장 마감 뒤 현재가·일봉은 시간외 단일가(16~18시)로 바뀌므로 15:30 분봉 값을 우선 쓴다.
                close, src = None, ''
                throttle()
                mb = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice', 'FHKST03010200',
                            {'FID_ETC_CLS_CODE': '', 'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c, 'FID_INPUT_HOUR_1': '153000', 'FID_PW_DATA_INCU_YN': 'N'})
                bars = [x for x in (mb.get('output2') or []) if x.get('stck_bsop_date') == today and x.get('stck_cntg_hour', '') <= '153000']
                if bars:
                    close, src = max(bars, key=lambda x: x['stck_cntg_hour'])['stck_prpr'], 'min1530'
                elif time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600)) == today and time.gmtime(time.time() + 9 * 3600).tm_hour < 16:
                    close, src = o['stck_prpr'], 'prpr'
                if close is None:
                    return None
                px, sh = float(close), float(o.get('lstn_stcn') or 0)
                return {'BAS_DD': today, 'ISU_CD': c, 'MKT_NM': mk.get(c, ''), 'TDD_CLSPRC': close, 'FLUC_RT': '',
                        'TDD_OPNPRC': o.get('stck_oprc', ''), 'TDD_HGPRC': o.get('stck_hgpr', ''), 'TDD_LWPRC': o.get('stck_lwpr', ''),
                        'ACC_TRDVOL': o.get('acml_vol', ''), 'ACC_TRDVAL': o.get('acml_tr_pbmn', ''), 'MKTCAP': f'{px * sh:.0f}',
                        'LIST_SHRS': o.get('lstn_stcn', ''), 'SRC': src, 'PRPR_NOW': o['stck_prpr']}
            time.sleep(1)
        return None

    fp = os.path.join(OUT, today + '.csv')
    if os.path.exists(fp) and sum(1 for _ in open(fp)) - 1 >= 0.95 * len(mk):
        print('이미 수집됨', today); return
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
