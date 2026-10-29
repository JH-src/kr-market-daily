"""한투 해외 일봉(환율): 원/달러(FX@KRW) 최근 2년 이상(조회 전용). 결과: data/kis/fx/usdkrw.csv"""
import csv, os, time
import kis_probe as K
import datetime as dt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'kis', 'fx'); os.makedirs(OUT, exist_ok=True)
tok = K.token(); rows = {}
end = dt.date.today(); a = dt.date(2024, 1, 1)
while a <= end:
    b = min(a + dt.timedelta(days=120), end)
    r = K.call(tok, '/uapi/overseas-price/v1/quotations/inquire-daily-chartprice', 'FHKST03030100',
               {'FID_COND_MRKT_DIV_CODE': 'X', 'FID_INPUT_ISCD': 'FX@KRW', 'FID_INPUT_DATE_1': a.strftime('%Y%m%d'), 'FID_INPUT_DATE_2': b.strftime('%Y%m%d'), 'FID_PERIOD_DIV_CODE': 'D'})
    o = r.get('output2') or []
    print(a, b, r.get('rt_cd'), r.get('msg1'), len(o), o[0] if o else '')
    for x in o:
        if x.get('stck_bsop_date'):
            rows[x['stck_bsop_date']] = x
    a = b + dt.timedelta(days=1); time.sleep(0.5)
if rows:
    ks = sorted(rows); f = list(rows[ks[0]])
    with open(os.path.join(OUT, 'usdkrw.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=f); w.writeheader(); [w.writerow(rows[k]) for k in ks]
