"""요소 시험용 추가 공시 이벤트(2015년~, 이어받기): 자기주식 취득·신탁 결정(B), 단일판매·공급계약 체결(I001), 최대주주 등 장내매수(지분공시 D 아님) 제외.
결과 data/dart/events_extra.csv (rcept_dt, stock_code, corp_name, kind, report_nm, rcept_no)."""
import csv, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D

RULES = {
    'B': [('buyback', r'자기주식취득결정|자기주식취득신탁계약체결결정'), ('burn', r'주식소각결정')],
    'I001': [('supply', r'단일판매.?공급계약체결'), ('supply_cancel', r'단일판매.?공급계약해지')],
}


def window(key, ty, rules, s, e):
    q = f"pblntf_ty={ty}" if len(ty) == 1 else f"pblntf_detail_ty={ty}"
    rows, page = [], 1
    while True:
        d = D._get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&bgn_de={s:%Y%m%d}&end_de={e:%Y%m%d}&{q}&page_no={page}&page_count=100")
        if d.get('status') not in ('000', '013'):
            return None
        for x in d.get('list', []):
            if not x.get('stock_code'):
                continue
            for kind, pat in rules:
                if re.search(pat, x['report_nm']):
                    rows.append({'rcept_dt': x['rcept_dt'], 'stock_code': x['stock_code'], 'corp_name': x['corp_name'],
                                 'kind': kind, 'report_nm': x['report_nm'].strip(), 'rcept_no': x['rcept_no']}); break
        if page >= int(d.get('total_page', 1) or 1):
            return rows
        page += 1


def main(y0=2015):
    """창(80일)마다 따로 받아 창별 파일로 저장(병렬·이어받기), 끝에 하나로 합침."""
    key = os.environ['DART_KEY']
    tmp = os.path.join(D.OUT, 'events_extra_parts'); os.makedirs(tmp, exist_ok=True)
    jobs = []
    for ty, rules in RULES.items():
        s = date(y0, 1, 1)
        while s < date.today():
            e = min(s + timedelta(days=80), date.today())
            if not os.path.exists(os.path.join(tmp, f"{ty}_{s:%Y%m%d}.csv")):
                jobs.append((ty, rules, s, e))
            s = e + timedelta(days=1)
    print('남은 창', len(jobs), flush=True)
    def one(j):
        ty, rules, s, e = j
        rows = window(key, ty, rules, s, e)
        if rows is None:
            print('실패', ty, s, flush=True); return
        with open(os.path.join(tmp, f"{ty}_{s:%Y%m%d}.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, ['rcept_dt', 'stock_code', 'corp_name', 'kind', 'report_nm', 'rcept_no']); w.writeheader(); w.writerows(rows)
        print(ty, s, len(rows), flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, jobs))
    import glob
    import pandas as pd
    parts = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(os.path.join(tmp, '*.csv')))]
    out = pd.concat(parts).drop_duplicates('rcept_no').sort_values('rcept_dt')
    out.to_csv(os.path.join(D.OUT, 'events_extra.csv'), index=False)
    print('ALL_DONE', len(out), out['kind'].value_counts().to_dict())


if __name__ == '__main__':
    main()
