"""전 종목 외국인 보유비율(오늘 값)·상장주식수 수집(조회 전용). 과거 보유비율은 이 값에서 외국인 순매수를 거꾸로 빼서 다시 만든다.
결과: data/kis/frgn_hold/<날짜>.csv (code, frgn_ehrt, lstn_stcn, close, w52_hgpr_date)"""
import csv, os, sys, time
from datetime import datetime, timedelta, timezone
import kis_probe as K

OUT = os.path.join(K.OUT, '..', 'frgn_hold')
os.makedirs(OUT, exist_ok=True)


def main():
    tok = None
    for i in range(6):
        try:
            tok = K.token(); break
        except Exception as e:
            print('토큰 재시도', i, e); time.sleep(65)
    today = (datetime.now(timezone.utc) + timedelta(hours=9)).strftime('%Y%m%d')
    codes = [c.strip() for c in open(os.path.join(K.OUT, '..', '..', '..', 'config', 'universe_kis.txt')) if c.strip()]
    rows, fail = [], 0
    for n, c in enumerate(codes):
        r = K.call(tok, K.Q + 'inquire-price', 'FHKST01010100', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c})
        o = r.get('output') if isinstance(r, dict) else None
        if not o or r.get('rt_cd') != '0':
            fail += 1
            if fail > 50 and fail > n * 0.5:
                print('실패 과다, 중단', n, fail); break
            time.sleep(1); continue
        rows.append({'date': today, 'code': c, 'frgn_ehrt': o.get('hts_frgn_ehrt', ''), 'lstn_stcn': o.get('lstn_stcn', ''),
                     'close': o.get('stck_prpr', ''), 'frgn_ntby_qty': o.get('frgn_ntby_qty', ''), 'vol_tnrt': o.get('vol_tnrt', '')})
        time.sleep(0.07)
    p = os.path.join(OUT, f'{today}.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'code', 'frgn_ehrt', 'lstn_stcn', 'close', 'frgn_ntby_qty', 'vol_tnrt']); w.writeheader(); w.writerows(rows)
    print('저장', p, len(rows), '실패', fail)


if __name__ == '__main__':
    main()
