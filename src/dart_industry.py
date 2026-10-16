"""상장사(상장폐지 포함, DART 종목코드 보유) 산업분류(KSIC) -> data/dart/industry.csv (이어받기)"""
import csv, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D

p = os.path.join(D.OUT, 'industry.csv')
have = set()
if os.path.exists(p):
    have = {r['stock_code'] for r in csv.DictReader(open(p, encoding='utf-8'))}
rows = list(csv.DictReader(open(os.path.join(D.OUT, 'corpcode.csv'), encoding='utf-8')))
f = open(p, 'a', newline='', encoding='utf-8')
w = csv.DictWriter(f, ['stock_code', 'corp_code', 'corp_name', 'induty_code', 'corp_cls', 'est_dt'])
if not have:
    w.writeheader()
n = 0
for r in rows:
    if r['stock_code'] in have:
        continue
    d = D._get(f"https://opendart.fss.or.kr/api/company.json?crtfc_key={os.environ['DART_KEY']}&corp_code={r['corp_code']}")
    if d.get('status') != '000':
        print('실패', r['stock_code'], d.get('status'), d.get('message'), flush=True)
        if d.get('status') == '020':
            break   # 한도 초과
        continue
    w.writerow({'stock_code': r['stock_code'], 'corp_code': r['corp_code'], 'corp_name': d.get('corp_name', ''),
                'induty_code': d.get('induty_code', ''), 'corp_cls': d.get('corp_cls', ''), 'est_dt': d.get('est_dt', '')})
    n += 1
    if n % 200 == 0:
        f.flush(); print(n, flush=True)
    time.sleep(0.05)
f.close()
print('ALL_DONE', n)
