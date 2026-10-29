"""특정 종목의 당일 큰 체결 확인(조회 전용): 현재가·시간외·시간대별 체결·투자자별. 결과: data/kis/probe/block_<코드>_<날짜>.json
python3 kis_block_check.py 034730"""
import json, os, sys, time
import kis_probe as K
code = sys.argv[1] if len(sys.argv) > 1 else '034730'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = K.token(); Q = K.Q; out = {}
def c(name, path, tr, params):
    r = K.call(tok, Q + path, tr, params); out[name] = r; time.sleep(0.4)
    print(name, r.get('rt_cd'), r.get('msg1'))
p = {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': code}
c('현재가', 'inquire-price', 'FHKST01010100', p)
c('투자자', 'inquire-investor', 'FHKST01010900', p)
c('회원사', 'inquire-member', 'FHKST01010600', p)
c('시간외현재가', 'inquire-overtime-price', 'FHPST02300000', p)
c('시간외일자별', 'inquire-daily-overtimeprice', 'FHPST02320000', p)
c('일별', 'inquire-daily-price', 'FHKST01010400', dict(p, FID_PERIOD_DIV_CODE='D', FID_ORG_ADJ_PRC='0'))
for hh in ('153000', '150000', '140000', '120000', '100000', '090500'):
    c(f'시간대별체결_{hh}', 'inquire-time-itemconclusion', 'FHPST01060000', dict(p, FID_INPUT_HOUR_1=hh))
c('최근체결', 'inquire-ccnl', 'FHKST01010300', p)
d0 = time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600))
c('투자자일별', 'investor-trade-by-stock-daily', 'FHPTJ04160001', dict(p, FID_INPUT_DATE_1=d0, FID_ORG_ADJ_PRC='', FID_ETC_CLS_CODE=''))
c('일봉차트', 'inquire-daily-itemchartprice', 'FHKST03010100', dict(p, FID_INPUT_DATE_1=d0, FID_INPUT_DATE_2=d0, FID_PERIOD_DIV_CODE='D', FID_ORG_ADJ_PRC='0'))
c('시간외호가', 'inquire-overtime-asking-price', 'FHPST02300400', p)
c('분봉', 'inquire-time-itemchartprice', 'FHKST03010200', dict(p, FID_ETC_CLS_CODE='', FID_INPUT_HOUR_1='180000', FID_PW_DATA_INCU_YN='Y'))
d = time.strftime('%Y%m%d', time.gmtime(time.time() + 9 * 3600))
os.makedirs(os.path.join(ROOT, 'data', 'kis', 'probe'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'data', 'kis', 'probe', f'block_{code}_{d}.json'), 'w'), ensure_ascii=False, indent=1)
