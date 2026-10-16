"""장 마감 직후 한투만으로 전 종목 시세를 받을 수 있는지 시간 측정(조회 전용).
결과: data/kis/probe/speed.json"""
import json, os, time
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
codes = [c.strip() for c in open(os.path.join(ROOT, 'config', 'universe_kis.txt')) if c.strip()]
tok = K.token()
t0 = time.time(); ok = 0; sample = None; errs = 0
for c in codes[:400]:
    r = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-price', 'FHKST01010100', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': c})
    o = r.get('output') or {}
    if o.get('stck_prpr'):
        ok += 1; sample = sample or {k: o.get(k) for k in ['stck_prpr', 'prdy_ctrt', 'acml_vol', 'acml_tr_pbmn', 'hts_avls', 'lstn_stcn', 'per', 'pbr', 'bstp_kor_isnm']}
    else:
        errs += 1
    time.sleep(0.055)
el = time.time() - t0
idx = {}
for nm, code in [('코스피', '0001'), ('코스닥', '1001'), ('VKOSPI', '0004')]:
    r = K.call(tok, '/uapi/domestic-stock/v1/quotations/inquire-index-price', 'FHPUP02100000', {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': code})
    idx[nm] = {k: (r.get('output') or {}).get(k) for k in ['bstp_nmix_prpr', 'bstp_nmix_prdy_ctrt']} if isinstance(r.get('output'), dict) else r.get('msg1')
out = {'calls': 400, 'ok': ok, 'errors': errs, 'seconds': round(el, 1), 'per_call': round(el / 400, 3),
       'est_all_minutes': round(el / 400 * len(codes) / 60, 1), 'n_codes': len(codes), 'sample': sample, 'index': idx,
       'run_at_kst': time.strftime('%Y-%m-%d %H:%M', time.gmtime(time.time() + 9 * 3600))}
os.makedirs(os.path.join(ROOT, 'data', 'kis', 'probe'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'data', 'kis', 'probe', 'speed.json'), 'w'), ensure_ascii=False, indent=1)
print(out)
