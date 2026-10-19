"""시트 신종·후순위 종목의 한투 채권 기본조회·발행정보 원문 저장(조회 전용). 결과: data/kis/bond/raw.json"""
import json, os, time
import kis_probe as K
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
isins = [l.strip() for l in open(os.path.join(ROOT, 'config', 'bond_isins.txt')) if l.strip()]
tok = K.token()
EP = [('info', '/uapi/domestic-bond/v1/quotations/search-bond-info', 'CTPF1114R'),
      ('issue', '/uapi/domestic-bond/v1/quotations/issue-info', 'CTPF1101R')]
out = {}
for i, c in enumerate(isins):
    out[c] = {}
    for k, p, t in EP:
        try:
            out[c][k] = K.call(tok, p, t, {'PDNO': c, 'PRDT_TYPE_CD': '302'})
        except Exception as e:
            out[c][k] = {'err': str(e)[:300]}
        time.sleep(0.08)
    if i == 0:
        print(json.dumps(out[c], ensure_ascii=False)[:3000])
os.makedirs(os.path.join(ROOT, 'data', 'kis', 'bond'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'data', 'kis', 'bond', 'raw.json'), 'w', encoding='utf-8'), ensure_ascii=False)
print('bond', len(out))
