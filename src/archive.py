"""16년치 거래소 원자료 적재(이어받기 가능) + 연도별 압축 보관.

python3 src/archive.py <시작연도> <끝연도> <초 예산> [단계 1|2]
- 최근 연도부터 거꾸로, 하루치 파일(data/<set>/<YYYY>/<YYYYMMDD>.csv.gz)이 있으면 건너뜀
- 한 해가 끝나면 archive/<set>/<YYYY>.csv.gz 로 합쳐 저장(저장소 보관용)
- 예산 시간이 다 되면 그 자리에서 멈춤(다시 실행하면 이어서)
"""
import gzip, glob, os, sys, time, csv
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import collect as C
import backfill_core as B

ROOT = C.ROOT
SETS = B.CORE + ['opt_summary']
# 1단계: 시뮬레이션 필수 자료만(빠르게). 2단계에서 나머지(채권지수·선물·옵션 요약)를 채운다.
PHASE1 = ['stk', 'ksq', 'idx_kospi', 'idx_kosdaq', 'idx_krx', 'idx_drv', 'etf']
PHASE2 = ['idx_bond', 'fut', 'opt_summary']


def one_day(d, sets):
    s = d.strftime('%Y%m%d')
    path = lambda n: os.path.join(C.DATA, n, s[:4], s + '.csv.gz')
    if 'stk' in sets and not os.path.exists(path('stk')):
        rows = C.krx_get(C.KRX_SETS['stk'], s)
        if not rows:
            return s, 'empty'
        C.save_rows('stk', s, rows)
    for n in sets:
        if n == 'stk' or os.path.exists(path(n)):
            continue
        if n == 'opt_summary':
            try:
                sm = C.opt_summary(s)
                if sm:
                    C.save_rows('opt_summary', s, sm)
            except Exception as e:
                print('opt fail', s, e)
            continue
        r = C.krx_get(C.KRX_SETS[n], s)
        if r:
            C.save_rows(n, s, r)
    return s, 'ok'
LOG = os.path.join(ROOT, 'data', '_calendar', 'archive_log.txt')


def empty_days():
    p = os.path.join(C.DATA, '_calendar', 'empty_days.txt')
    return set(open(p).read().split()) if os.path.exists(p) else set()


def consolidate(y, sets=None):
    for s in (sets or SETS):
        fs = sorted(glob.glob(os.path.join(C.DATA, s, str(y), f'{y}*.csv.gz')))
        if not fs:
            continue
        out = os.path.join(ROOT, 'archive', s, f'{y}.csv.gz')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        header = None
        with gzip.open(out, 'wt', encoding='utf-8', newline='') as g:
            for f in fs:
                with gzip.open(f, 'rt', encoding='utf-8') as h:
                    first = h.readline()
                    if header is None:
                        header = first; g.write(first)
                    elif first != header:
                        # 열 구성이 달라진 날은 열 이름 기준으로 맞춘다
                        rd = csv.DictReader([first] + h.readlines())
                        cols = next(csv.reader([header]))
                        w = csv.DictWriter(g, cols, extrasaction='ignore')
                        for r in rd:
                            w.writerow(r)
                        continue
                    g.write(h.read())


def run(y0, y1, budget, sets):
    t0 = time.time()
    emp = empty_days()
    for y in range(y1, y0 - 1, -1):
        days = [date(y, 1, 1) + timedelta(i) for i in range(366)]
        days = [d for d in days if d.year == y and d.weekday() < 5 and d < date.today()]
        todo = [d for d in days if d.strftime('%Y%m%d') not in emp
                and not all(os.path.exists(os.path.join(C.DATA, n, str(y), d.strftime('%Y%m%d') + '.csv.gz')) for n in sets if n != 'opt_summary')]
        print(y, '남은 날', len(todo), flush=True)
        new_empty = []
        with ThreadPoolExecutor(5) as ex:
            futs = []
            for d in todo:
                futs.append(ex.submit(one_day, d, sets))
            for i, f in enumerate(futs):
                if time.time() - t0 > budget:
                    for x in futs:
                        x.cancel()
                    break
                try:
                    s, st = f.result()
                except Exception as e:
                    print('실패', e, flush=True); continue
                if st == 'empty':
                    new_empty.append(s)
                if i % 25 == 0:
                    print(y, i, len(todo), s, st, round(time.time() - t0), flush=True)
        with open(os.path.join(C.DATA, '_calendar', 'empty_days.txt'), 'a') as fh:
            for s in new_empty:
                fh.write(s + '\n')
        emp |= set(new_empty)
        if time.time() - t0 > budget:
            print('예산 소진, 멈춤', y, flush=True)
            return False
        consolidate(y, sets)
        with open(LOG, 'a') as fh:
            fh.write(f'{y} done {time.strftime("%Y-%m-%d %H:%M")}\n')
        print(y, '완료·보관', flush=True)
    return True


if __name__ == '__main__':
    phase = sys.argv[4] if len(sys.argv) > 4 else '1'
    done = run(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), PHASE1 if phase == '1' else PHASE2)
    print('ALL_DONE' if done else 'PAUSED')
