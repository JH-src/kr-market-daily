"""외국인 보유비율 과거 자료를 어디서 받을 수 있는지 점검(Actions에서 실행). 결과: data/kis/probe/frgn_*.json"""
import json, os, urllib.request, urllib.parse
from kis_probe import token, call, Q, OUT

def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, f'frgn_{name}.json'), 'w'), ensure_ascii=False, indent=1)

import time
tok = None
for i in range(6):
    try:
        tok = token(); break
    except Exception as e:
        print('토큰 재시도', i, e); time.sleep(65)
# 1) 한투 현재가: 외국인 보유수량·소진율(오늘 값)
save('한투_현재가', call(tok, Q + 'inquire-price', 'FHKST01010100', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930'}))
# 2) 한투 일자별 시세: 소진율이 날짜별로 오는지, 몇 일치인지
save('한투_일자별', call(tok, Q + 'inquire-daily-price', 'FHKST01010400', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930', 'FID_PERIOD_DIV_CODE': 'D', 'FID_ORG_ADJ_PRC': '0'}))
save('한투_일자별_월', call(tok, Q + 'inquire-daily-price', 'FHKST01010400', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930', 'FID_PERIOD_DIV_CODE': 'M', 'FID_ORG_ADJ_PRC': '0'}))
# 3) 거래소 정보데이터시스템: 외국인 보유량 개별종목 추이(과거 전체)
def krx(bld, params):
    body = urllib.parse.urlencode(dict(bld=bld, **params)).encode()
    req = urllib.request.Request('https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd', data=body, headers={
        'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'})
    try:
        r = urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'ignore')
        try:
            j = json.loads(r); [j.__setitem__(k, v[:5]) for k, v in list(j.items()) if isinstance(v, list)]; return j
        except Exception:
            return {'raw': r[:800]}
    except Exception as e:
        return {'error': str(e)[:300]}
save('거래소_종목추이', krx('dbms/MDC/STAT/standard/MDCSTAT03702', {'searchType': '2', 'mktId': 'ALL', 'isuCd': 'KR7005930003', 'strtDd': '20150102', 'endDd': '20150130', 'share': '1', 'csvxls_isNo': 'false'}))
save('거래소_전종목', krx('dbms/MDC/STAT/standard/MDCSTAT03701', {'searchType': '1', 'mktId': 'STK', 'trdDd': '20150130', 'share': '1', 'csvxls_isNo': 'false'}))
print('done')
