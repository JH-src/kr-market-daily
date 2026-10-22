"""증시자금(예탁금·신용잔고 등) 과거 구간 받기(조회 전용). 기준일을 바꿔가며 받아 data/kis/mktfunds.csv에 합침."""
import time
import kis_probe as K
import kis_daily as KD
import os

tok = K.token()
for d in ('20260131', '20260301', '20260401', '20260501', '20260601'):
    r = K.call(tok, K.Q + 'mktfunds', 'FHKST649100C0', {'FID_INPUT_DATE_1': d})
    rows = r.get('output') or []
    print(d, r.get('rt_cd'), r.get('msg1'), len(rows), rows[0]['bsop_date'] if rows else '', rows[-1]['bsop_date'] if rows else '')
    if rows:
        KD.upsert(os.path.join(KD.OUT, 'mktfunds.csv'), rows, 'bsop_date')
    time.sleep(1)
