"""주식등의 대량보유상황보고서(5% 지분공시) 목록 2015년~ -> data/dart/major_holders.csv (이어받기)
보고자(flr_nm)로 외국계 기관·국민연금 등 구분은 시뮬레이션 쪽에서 한다."""
import csv, os, sys, time
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D

p = os.path.join(D.OUT, 'major_holders.csv')
dp = os.path.join(D.OUT, 'major_holders_done.txt')
done = set(open(dp).read().split()) if os.path.exists(dp) else set()
seen = {r['rcept_no'] for r in csv.DictReader(open(p, encoding='utf-8'))} if os.path.exists(p) else set()
f = open(p, 'a', newline='', encoding='utf-8')
w = csv.DictWriter(f, ['rcept_dt', 'stock_code', 'corp_name', 'flr_nm', 'report_nm', 'rcept_no'])
if not seen:
    w.writeheader()
key = os.environ['DART_KEY']
s = date(2015, 1, 1)
while s < date.today():
    e = min(s + timedelta(days=80), date.today())
    tag = s.strftime('%Y%m%d')
    if tag not in done:
        page, ok = 1, True
        while True:
            d = D._get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&bgn_de={s:%Y%m%d}&end_de={e:%Y%m%d}&pblntf_detail_ty=D001&page_no={page}&page_count=100")
            if d.get('status') not in ('000', '013'):
                print('실패', s, page, d.get('status'), d.get('message'), flush=True); ok = False; break
            for x in d.get('list', []):
                if x.get('stock_code') and x['rcept_no'] not in seen:
                    w.writerow({k: x.get(k, '') for k in ['rcept_dt', 'stock_code', 'corp_name', 'flr_nm', 'report_nm', 'rcept_no']})
                    seen.add(x['rcept_no'])
            if page >= int(d.get('total_page', 1) or 1):
                break
            page += 1
            time.sleep(0.1)
        f.flush()
        if ok:
            open(dp, 'a').write(tag + '\n')
        print('major', s, len(seen), flush=True)
    s = e + timedelta(days=1)
print('ALL_DONE')
