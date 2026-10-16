"""핵심 데이터셋만 빠르게 과거 적재(예시·검증용). 날짜 여러 개를 동시에 받는다."""
import sys, os, time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import collect as C

CORE = ['stk', 'ksq', 'idx_kospi', 'idx_kosdaq', 'idx_krx', 'idx_drv', 'idx_bond', 'etf', 'fut']


def one(d):
    s = d.strftime('%Y%m%d')
    if os.path.exists(os.path.join(C.DATA, 'stk', s[:4], s + '.csv.gz')):
        return s, 'have'
    rows = C.krx_get(C.KRX_SETS['stk'], s)
    if not rows:
        return s, 'empty'
    C.save_rows('stk', s, rows)
    for n in CORE[1:]:
        r = C.krx_get(C.KRX_SETS[n], s)
        if r:
            C.save_rows(n, s, r)
    try:
        sm = C.opt_summary(s)
        if sm:
            C.save_rows('opt_summary', s, sm)
    except Exception as e:
        print('opt fail', s, e)
    return s, 'ok'


if __name__ == '__main__':
    s, e = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    days = [s + timedelta(i) for i in range((e - s).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    t0 = time.time(); empties = []
    with ThreadPoolExecutor(3) as ex:
        for i, (d, st) in enumerate(ex.map(one, days)):
            if st == 'empty':
                empties.append(d)
            if i % 20 == 0:
                print(i, len(days), d, st, round(time.time() - t0), flush=True)
    os.makedirs(os.path.join(C.DATA, '_calendar'), exist_ok=True)
    with open(os.path.join(C.DATA, '_calendar', 'empty_days.txt'), 'a') as f:
        for d in empties:
            if d < date.today().strftime('%Y%m%d'):
                f.write(d + '\n')
    print('done', round(time.time() - t0))
