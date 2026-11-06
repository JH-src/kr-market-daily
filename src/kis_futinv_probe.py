"""선물 투자자별 매매동향(일별) 조회 방법 탐색 2차(조회 전용) -> data/kis/probe/futinv_probe2.json"""
import json, os, time
import kis_probe as K
Q = '/uapi/domestic-stock/v1/quotations/'
tok = K.token(); out = {}
d = '20261106'
tries = []
for cond in ['F', 'JF', 'W', 'U']:
    for isc, i1 in [('F001', 'K2I'), ('101', 'K2I'), ('K2I', 'K2I'), ('F001', 'F001'), ('0001', 'K2I')]:
        tries.append(('daily', cond, isc, i1))
for t, cond, isc, i1 in tries:
    r = K.call(tok, Q + 'inquire-investor-daily-by-market', 'FHPTJ04040000',
               {'FID_COND_MRKT_DIV_CODE': cond, 'FID_INPUT_ISCD': isc, 'FID_INPUT_DATE_1': d,
                'FID_INPUT_ISCD_1': i1, 'FID_INPUT_DATE_2': d, 'FID_INPUT_ISCD_2': isc})
    o = r.get('output') or []
    nz = [x for x in o if x.get('frgn_ntby_qty') not in ('0', None)]
    out[f'{cond}_{isc}_{i1}'] = {'rt': r.get('rt_cd'), 'msg': r.get('msg1'), 'n': len(o), 'nz': len(nz), 'head': (nz or o)[:2], 'err': r.get('http_error')}
    time.sleep(0.4)
# 시간대별(당일) 시장 투자자: 과거 날짜 지정 가능 여부
for extra in [{'FID_INPUT_DATE_1': '20261030'}, {}]:
    p = {'FID_INPUT_ISCD': 'K2I', 'FID_INPUT_ISCD_2': 'F001'}; p.update(extra)
    r = K.call(tok, Q + 'inquire-investor-time-by-market', 'FHPTJ04030000', p)
    out['time_' + str(extra)] = {'rt': r.get('rt_cd'), 'msg': r.get('msg1'), 'head': (r.get('output') or [])[:1]}
json.dump(out, open('../data/kis/probe/futinv_probe2.json', 'w'), ensure_ascii=False, indent=1)
print({k: (v.get('rt'), v.get('n'), v.get('nz')) for k, v in out.items()})
