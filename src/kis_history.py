"""한투 종목별 투자자 순매수 과거 적재(GitHub Actions, 이어받기 가능).

종목마다 30거래일씩 거슬러 올라가며 2015년까지(자료가 끊기면 거기까지) 받아 archive/kis_flows/<종목>.csv.gz 로 저장.
완료한 종목은 archive/kis_flows/_done.txt 에 기록. 초당 호출은 15회 이하로 제한.
"""
import csv, gzip, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import kis_probe as K
from kis_daily import FLOW_COLS, Q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'archive', 'kis_flows')
STOP = '20150101'
lock = threading.Lock()
last = [0.0]


def throttle(rate=15):
    with lock:
        wait = last[0] + 1 / rate - time.time()
        if wait > 0:
            time.sleep(wait)
        last[0] = time.time()


def one(tok, code, today):
    rows, d, err = {}, today, 0
    for _ in range(150):
        throttle()
        r = K.call(tok, Q + 'investor-trade-by-stock-daily', 'FHPTJ04160001',
                   {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': code, 'FID_INPUT_DATE_1': d, 'FID_ORG_ADJ_PRC': '', 'FID_ETC_CLS_CODE': ''})
        if r.get('http_error') or (r.get('rt_cd') not in (None, '0')):
            # 초당 건수 초과 등 오류 응답은 빈 결과가 아님: 잠시 쉬고 다시
            err += 1
            if err > 6:
                break
            time.sleep(2); continue
        out = [x for x in (r.get('output2') or []) if x.get('stck_bsop_date')]
        if not out:
            break
        for x in out:
            rows[x['stck_bsop_date']] = {v: x.get(k, '') for k, v in FLOW_COLS.items()}
        oldest = min(x['stck_bsop_date'] for x in out)
        if oldest <= STOP or len(out) < 2:
            break
        y, m, dd = int(oldest[:4]), int(oldest[4:6]), int(oldest[6:])
        import datetime as _d
        d = (_d.date(y, m, dd) - _d.timedelta(days=1)).strftime('%Y%m%d')
    if rows:
        with gzip.open(os.path.join(OUT, code + '.csv.gz'), 'wt', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, list(FLOW_COLS.values())); w.writeheader()
            for k in sorted(rows):
                w.writerow(rows[k])
    return code, len(rows)


def main(budget):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    dp = os.path.join(OUT, '_done.txt')
    done = set(open(dp).read().split()) if os.path.exists(dp) else set()
    codes = [c.strip() for c in open(os.path.join(ROOT, 'config', 'universe_kis.txt')) if c.strip() and c.strip() not in done]
    # 선정 종목은 먼저
    lp = os.path.join(ROOT, 'data', 'picks', 'log.csv')
    pri = set(l.split(',')[1] for l in open(lp).read().split()[1:]) if os.path.exists(lp) else set()
    codes = [c for c in codes if c in pri] + [c for c in codes if c not in pri]
    print('남은 종목', len(codes), '우선', len(pri & set(codes)), flush=True)
    tok = K.token()
    today = time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600))
    with ThreadPoolExecutor(6) as ex:
        futs = {}
        it = iter(codes)
        for c in list(codes[:12]):
            futs[ex.submit(one, tok, c, today)] = c
        nxt = 12
        n = 0
        while futs:
            for f in list(futs):
                if f.done():
                    c = futs.pop(f)
                    try:
                        code, k = f.result()
                        if k > 0:   # 받은 행이 있을 때만 완료 처리(오류로 빈 결과면 다음에 다시)
                            with open(dp, 'a') as fh:
                                fh.write(code + '\n')
                        n += 1
                        if n % 50 == 0:
                            print(n, code, k, round(time.time() - t0), flush=True)
                    except Exception as e:
                        print('실패', c, e, flush=True)
                    if nxt < len(codes) and time.time() - t0 < budget:
                        futs[ex.submit(one, tok, codes[nxt], today)] = codes[nxt]; nxt += 1
            time.sleep(0.2)
    left = len(codes) - n
    print('완료' if left == 0 else f'남음 {left}', flush=True)


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 18000)
