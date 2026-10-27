"""한투 해외주식 일봉 조회 가능 기간 점검(조회 전용). 결과: data/kis/probe/us_probe.json"""
import io, json, os, time, urllib.request, zipfile
import kis_probe as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
# 나스닥 종목 마스터(한투 공개 파일)
try:
    b = urllib.request.urlopen('https://new.real.download.dws.co.kr/common/master/nasmst.cod.zip', timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(b)); t = z.read(z.namelist()[0]).decode('cp949', 'ignore').splitlines()
    out['master_rows'] = len(t); out['master_sample'] = t[:3]
except Exception as e:
    out['master_error'] = str(e)
tok = K.token()
for sym in ('AAPL', 'MSFT', 'QQQ'):
    for bymd in ('19950103', '20050103', '20120103', '20200102', ''):
        excd = 'NAS'
        r = K.call(tok, '/uapi/overseas-price/v1/quotations/dailyprice', 'HHDFS76240000',
                   {'AUTH': '', 'EXCD': excd, 'SYMB': sym, 'GUBN': '0', 'BYMD': bymd, 'MODP': '1'})
        rows = r.get('output2') or []
        ds = [x.get('xymd') for x in rows if x.get('xymd')]
        out[f'{sym}_{bymd or "today"}'] = {'rt': r.get('rt_cd'), 'msg': r.get('msg1'), 'n': len(ds), 'first': min(ds) if ds else None, 'last': max(ds) if ds else None}
        time.sleep(0.3)
os.makedirs(os.path.join(ROOT, 'data', 'kis', 'probe'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'data', 'kis', 'probe', 'us_probe.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
