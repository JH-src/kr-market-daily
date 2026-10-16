"""16년치 거래소 원자료 적재(이어받기 가능) + 연도별 압축 보관.

python3 src/archive.py <시작연도> <끝연도> <초 예산>
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
LOG = os.path.join(ROOT, 'data', '_calendar', 'archive_log.txt')


def empty_days():
    p = os.path.join(C.DATA, '_calendar', 'empty_days.txt')
    return set(open(p).read().split()) if os.path.exists(p) else set()


def consolidate(y):
    for s in SETS:
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


def run(y0, y1, budget):
    t0 = time.time()
    emp = empty_days()
    for y in range(y1, y0 - 1, -1):
        days = [date(y, 1, 1) + timedelta(i) for i in range(366)]
        days = [d for d in days if d.year == y and d.weekday() < 5 and d < date.today()]
        todo = [d for d in days if d.strftime('%Y%m%d') not in emp
                and not os.path.exists(os.path.join(C.DATA, 'stk', str(y), d.strftime('%Y%m%d') + '.csv.gz'))]
        print(y, '남은 날', len(todo), flush=True)
        new_empty = []
        with ThreadPoolExecutor(4) as ex:
            futs = []
            for d in todo:
                futs.append(ex.submit(B.one, d))
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
        consolidate(y)
        with open(LOG, 'a') as fh:
            fh.write(f'{y} done {time.strftime("%Y-%m-%d %H:%M")}\n')
        print(y, '완료·보관', flush=True)
    return True


if __name__ == '__main__':
    done = run(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
    print('ALL_DONE' if done else 'PAUSED')
