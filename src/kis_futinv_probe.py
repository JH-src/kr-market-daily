"""선물 투자자별 매매동향 조회 방법 탐색(조회 전용) -> data/kis/probe/futinv_probe.json"""
import json, os, time
import kis_probe as K
Q = '/uapi/domestic-stock/v1/quotations/'
tok = K.token(); out = {}
d1, d2 = '20260901', '20261106'
for i1 in ['K2I', 'F001', 'KFU', 'KSP']:
    for isc in ['F001', '0001', '101']:
        r = K.call(tok, Q + 'inquire-investor-daily-by-market', 'FHPTJ04040000',
                   {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': isc, 'FID_INPUT_DATE_1': d2,
                    'FID_INPUT_ISCD_1': i1, 'FID_INPUT_DATE_2': d2, 'FID_INPUT_ISCD_2': isc})
        o = r.get('output') or []
        out[f'daily_{i1}_{isc}'] = {'rt': r.get('rt_cd'), 'msg': r.get('msg1'), 'n': len(o), 'head': o[:2], 'err': r.get('http_error') or r.get('net_error')}
        time.sleep(0.4)
for i1, i2 in [('999', 'F001'), ('999', 'K2I'), ('K2I', 'F001'), ('KSP', '0001')]:
    r = K.call(tok, Q + 'inquire-investor-time-by-market', 'FHPTJ04030000', {'FID_INPUT_ISCD': i1, 'FID_INPUT_ISCD_2': i2})
    o = r.get('output') or []
    out[f'time_{i1}_{i2}'] = {'rt': r.get('rt_cd'), 'msg': r.get('msg1'), 'n': len(o) if isinstance(o, list) else 1, 'head': o[:2] if isinstance(o, list) else o}
    time.sleep(0.4)
os.makedirs('../data/kis/probe', exist_ok=True)
json.dump(out, open('../data/kis/probe/futinv_probe.json', 'w'), ensure_ascii=False, indent=1)
print({k: (v['rt'], v['msg'], v['n']) for k, v in out.items()})
