"""한투 업종지수 일봉(코스피 0001, 코스닥 1001) — 최근 거래량·거래대금 추이용(조회 전용).
결과: data/kis/idx_daily/<코드>.csv (응답 필드 그대로)"""
import csv, os, time
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'kis', 'idx_daily')
os.makedirs(OUT, exist_ok=True)
tok = K.token()
import datetime as dt
today = dt.date.fromtimestamp(time.time() + 9 * 3600)
first = dt.date(today.year, 1, 1)                    # 올해 1월 1일부터(구간당 최대 100건이라 60일씩 나눠 받음)
for code in ('0001', '1001'):
    allr, a = {}, first
    while a <= today:
        b = min(a + dt.timedelta(days=59), today)
        r = K.call(tok, K.Q + 'inquire-daily-indexchartprice', 'FHKUP03500100',
                   {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': code, 'FID_INPUT_DATE_1': a.strftime('%Y%m%d'), 'FID_INPUT_DATE_2': b.strftime('%Y%m%d'), 'FID_PERIOD_DIV_CODE': 'D'})
        for x in r.get('output2') or []:
            if x.get('stck_bsop_date') and a.strftime('%Y%m%d') <= x['stck_bsop_date'] <= b.strftime('%Y%m%d'):
                allr[x['stck_bsop_date']] = x
        print(code, a, b, r.get('rt_cd'), r.get('msg1'), len(r.get('output2') or []))
        a = b + dt.timedelta(days=1); time.sleep(1)
    rows = [allr[k] for k in sorted(allr)]
    if rows:
        with open(os.path.join(OUT, f'{code}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
