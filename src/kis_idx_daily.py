"""한투 업종지수 일봉(코스피 0001, 코스닥 1001) — 최근 거래량·거래대금 추이용(조회 전용).
결과: data/kis/idx_daily/<코드>.csv (응답 필드 그대로)"""
import csv, os, time
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'kis', 'idx_daily')
os.makedirs(OUT, exist_ok=True)
tok = K.token()
end = time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600))
start = time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600 - 75 * 86400))
for code in ('0001', '1001'):
    r = K.call(tok, K.Q + 'inquire-daily-indexchartprice', 'FHKUP03500100',
               {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': code, 'FID_INPUT_DATE_1': start, 'FID_INPUT_DATE_2': end, 'FID_PERIOD_DIV_CODE': 'D'})
    rows = r.get('output2') or []
    print(code, r.get('rt_cd'), r.get('msg1'), len(rows))
    if rows:
        with open(os.path.join(OUT, f'{code}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    time.sleep(1)
