"""시뮬레이션용 DART 과거 자료(이어받기 가능).

1) 2015년~ 전 상장사 분기 매출·영업이익 -> data/dart/fin/<연도>_<보고서>.csv (rcept_no 앞 8자리 = 공시일, 그 이후부터만 사용)
2) 2015년~ 주요사항보고서 중 유상증자·CB·BW·자사주 소각 결정 -> data/dart/events_major.csv
"""
import csv, os, re, sys, time
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D


def fin_all(y0=2015):
    cmap = D.corp_map()
    codes = sorted(cmap)
    today = date.today()
    for y in range(y0, today.year + 1):
        for rc, due in [('11013', '0515'), ('11012', '0814'), ('11014', '1114'), ('11011', None)]:
            due_d = date(y + 1, 3, 31) if rc == '11011' else date(y, int(due[:2]), int(due[2:]))
            if due_d > today:
                continue
            p = D.fetch_period(y, rc, cmap, codes)
            print('fin', y, rc, sum(1 for _ in open(p)) - 1, flush=True)


def events(y0=2015):
    key = os.environ['DART_KEY']
    p = os.path.join(D.OUT, 'events_major.csv')
    seen = set()
    if os.path.exists(p):
        seen = {r['rcept_no'] for r in csv.DictReader(open(p, encoding='utf-8'))}
    done_p = os.path.join(D.OUT, 'events_major_done.txt')
    done = set(open(done_p).read().split()) if os.path.exists(done_p) else set()
    f = open(p, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, ['rcept_dt', 'stock_code', 'corp_name', 'kind', 'report_nm', 'rcept_no'])
    if not seen:
        w.writeheader()
    s = date(y0, 1, 1)
    while s < date.today():
        e = min(s + timedelta(days=80), date.today())
        tag = s.strftime('%Y%m%d')
        if tag not in done:
            page, ok = 1, True
            while True:
                d = D._get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&bgn_de={s:%Y%m%d}&end_de={e:%Y%m%d}&pblntf_ty=B&page_no={page}&page_count=100")
                if d.get('status') not in ('000', '013'):
                    print('목록 실패', s, page, d.get('status'), d.get('message'), flush=True); ok = False; break
                for x in d.get('list', []):
                    nm = x['report_nm']
                    if not x.get('stock_code') or x['rcept_no'] in seen:
                        continue
                    kind = ('rights' if re.search(r'유상증자결정|전환사채권발행결정|신주인수권부사채권발행결정', nm) else
                            'burn' if '주식소각결정' in nm else None)
                    if kind:
                        w.writerow({'rcept_dt': x['rcept_dt'], 'stock_code': x['stock_code'], 'corp_name': x['corp_name'],
                                    'kind': kind, 'report_nm': nm.strip(), 'rcept_no': x['rcept_no']})
                        seen.add(x['rcept_no'])
                if page >= int(d.get('total_page', 1) or 1):
                    break
                page += 1
                time.sleep(0.1)
            f.flush()
            if ok:
                with open(done_p, 'a') as fh:
                    fh.write(tag + '\n')
            print('events', s, e, len(seen), flush=True)
        s = e + timedelta(days=1)
    f.close()


if __name__ == '__main__':
    events()
    fin_all()
    print('ALL_DONE')
