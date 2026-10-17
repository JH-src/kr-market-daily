"""요소 시험용 추가 공시 이벤트(2015년~, 이어받기): 자기주식 취득·신탁 결정(B), 단일판매·공급계약 체결(I001), 최대주주 등 장내매수(지분공시 D 아님) 제외.
결과 data/dart/events_extra.csv (rcept_dt, stock_code, corp_name, kind, report_nm, rcept_no)."""
import csv, os, re, sys, time
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D

RULES = {
    'B': [('buyback', r'자기주식취득결정|자기주식취득신탁계약체결결정'), ('burn', r'주식소각결정')],
    'I001': [('supply', r'단일판매.?공급계약체결'), ('supply_cancel', r'단일판매.?공급계약해지')],
}


def main(y0=2015):
    key = os.environ['DART_KEY']
    p = os.path.join(D.OUT, 'events_extra.csv')
    seen = {r['rcept_no'] for r in csv.DictReader(open(p, encoding='utf-8'))} if os.path.exists(p) else set()
    dp = os.path.join(D.OUT, 'events_extra_done.txt')
    done = set(open(dp).read().split()) if os.path.exists(dp) else set()
    f = open(p, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, ['rcept_dt', 'stock_code', 'corp_name', 'kind', 'report_nm', 'rcept_no'])
    if not seen:
        w.writeheader()
    for ty, rules in RULES.items():
        q = f"pblntf_ty={ty}" if len(ty) == 1 else f"pblntf_detail_ty={ty}"
        s = date(y0, 1, 1)
        while s < date.today():
            e = min(s + timedelta(days=80), date.today())
            tag = f"{ty}_{s:%Y%m%d}"
            if tag not in done:
                page, ok = 1, True
                while True:
                    d = D._get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&bgn_de={s:%Y%m%d}&end_de={e:%Y%m%d}&{q}&page_no={page}&page_count=100")
                    if d.get('status') not in ('000', '013'):
                        print('목록 실패', tag, page, d.get('status'), d.get('message'), flush=True); ok = False; break
                    for x in d.get('list', []):
                        if not x.get('stock_code') or x['rcept_no'] in seen:
                            continue
                        nm = x['report_nm']
                        for kind, pat in rules:
                            if re.search(pat, nm):
                                w.writerow({'rcept_dt': x['rcept_dt'], 'stock_code': x['stock_code'], 'corp_name': x['corp_name'],
                                            'kind': kind, 'report_nm': nm.strip(), 'rcept_no': x['rcept_no']})
                                seen.add(x['rcept_no']); break
                    if page >= int(d.get('total_page', 1) or 1):
                        break
                    page += 1
                f.flush()
                if ok:
                    open(dp, 'a').write(tag + '\n')
                print(tag, len(seen), flush=True)
            s = e + timedelta(days=1)
    f.close(); print('ALL_DONE')


if __name__ == '__main__':
    main()
