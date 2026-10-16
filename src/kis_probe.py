"""한국투자증권 Open API 조회 기능 점검(GitHub Actions에서 실행).

환경변수 KIS_APP_KEY, KIS_APP_SECRET (저장소 Secrets). 주문·계좌 기능은 호출하지 않는다.
결과: data/kis/probe/summary.json, data/kis/probe/<이름>.json (응답 앞부분만)
"""
import json, os, time, urllib.request, urllib.parse

BASE = 'https://openapi.koreainvestment.com:9443'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'kis', 'probe')
os.makedirs(OUT, exist_ok=True)
KEY, SEC = os.environ['KIS_APP_KEY'], os.environ['KIS_APP_SECRET']


def token():
    body = json.dumps({'grant_type': 'client_credentials', 'appkey': KEY, 'appsecret': SEC}).encode()
    req = urllib.request.Request(BASE + '/oauth2/tokenP', data=body, headers={'content-type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())['access_token']


def call(tok, path, tr, params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f'{BASE}{path}?{q}', headers={
        'content-type': 'application/json; charset=utf-8', 'authorization': f'Bearer {tok}',
        'appkey': KEY, 'appsecret': SEC, 'tr_id': tr, 'custtype': 'P', 'tr_cont': ''})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode('utf-8', 'ignore')[:500]}


Q = '/uapi/domestic-stock/v1/quotations/'
TESTS = [
    ('종목_투자자_최근', Q + 'inquire-investor', 'FHKST01010900', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930'}),
    ('종목_투자자_일별_과거', Q + 'investor-trade-by-stock-daily', 'FHPTJ04160001', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930', 'FID_INPUT_DATE_1': '20200103', 'FID_ORG_ADJ_PRC': '', 'FID_ETC_CLS_CODE': ''}),
    ('종목_투자자_일별_최근', Q + 'investor-trade-by-stock-daily', 'FHPTJ04160001', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930', 'FID_INPUT_DATE_1': '20261015', 'FID_ORG_ADJ_PRC': '', 'FID_ETC_CLS_CODE': ''}),
    ('시장_투자자_일별', Q + 'inquire-investor-daily-by-market', 'FHPTJ04040000', {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': '0001', 'FID_INPUT_DATE_1': '20261015', 'FID_INPUT_ISCD_1': 'KSP', 'FID_INPUT_DATE_2': '20261015', 'FID_INPUT_ISCD_2': '0001'}),
    ('시장_투자자_일별_과거', Q + 'inquire-investor-daily-by-market', 'FHPTJ04040000', {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': '0001', 'FID_INPUT_DATE_1': '20150105', 'FID_INPUT_ISCD_1': 'KSP', 'FID_INPUT_DATE_2': '20150105', 'FID_INPUT_ISCD_2': '0001'}),
    ('외인기관_순매수_상위', Q + 'foreign-institution-total', 'FHPTJ04400000', {'FID_COND_MRKT_DIV_CODE': 'V', 'FID_COND_SCR_DIV_CODE': '16449', 'FID_INPUT_ISCD': '0000', 'FID_DIV_CLS_CODE': '0', 'FID_RANK_SORT_CLS_CODE': '0', 'FID_ETC_CLS_CODE': '0'}),
    ('프로그램매매_일별', Q + 'comp-program-trade-daily', 'FHPPG04600001', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_MRKT_CLS_CODE': 'K', 'FID_INPUT_DATE_1': '20260901', 'FID_INPUT_DATE_2': '20261015'}),
    ('공매도_일별', Q + 'daily-short-sale', 'FHPST04830000', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930', 'FID_INPUT_DATE_1': '20260901', 'FID_INPUT_DATE_2': '20261015'}),
    ('신용잔고_일별', Q + 'daily-credit-balance', 'FHPST04760000', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_COND_SCR_DIV_CODE': '20476', 'FID_INPUT_ISCD': '005930', 'FID_INPUT_DATE_1': '20261015'}),
    ('대차_일별', Q + 'daily-loan-trans', 'HHPST074500C0', {'MRKT_DIV_CLS_CODE': '1', 'MKSC_SHRN_ISCD': '005930', 'START_DATE': '20260901', 'END_DATE': '20261015', 'CTS': ''}),
    ('증시자금_예탁금', Q + 'mktfunds', 'FHKST649100C0', {'FID_INPUT_DATE_1': ''}),
    ('금리_종합', Q + 'comp-interest', 'FHPST07020000', {'FID_COND_MRKT_DIV_CODE': 'I', 'FID_COND_SCR_DIV_CODE': '20702', 'FID_DIV_CLS_CODE': '1', 'FID_DIV_CLS_CODE1': ''}),
    ('수정주가_일봉', Q + 'inquire-daily-itemchartprice', 'FHKST03010100', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '005930', 'FID_INPUT_DATE_1': '20100104', 'FID_INPUT_DATE_2': '20100630', 'FID_PERIOD_DIV_CODE': 'D', 'FID_ORG_ADJ_PRC': '0'}),
    ('추정실적_컨센서스', Q + 'estimate-perform', 'HHKST668300C0', {'SHT_CD': '005930'}),
    ('증권사_투자의견', Q + 'invest-opinion', 'FHKST663300C0', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_COND_SCR_DIV_CODE': '16633', 'FID_INPUT_ISCD': '005930', 'FID_INPUT_DATE_1': '20260701', 'FID_INPUT_DATE_2': '20261015'}),
    ('뉴스_제목', Q + 'news-title', 'FHKST01011800', {'FID_NEWS_OFER_ENTP_CODE': '', 'FID_COND_MRKT_CLS_CODE': '', 'FID_INPUT_ISCD': '', 'FID_TITL_CNTT': '', 'FID_INPUT_DATE_1': '', 'FID_INPUT_HOUR_1': '', 'FID_RANK_SORT_CLS_CODE': '', 'FID_INPUT_SRNO': ''}),
    ('휴장일', Q + 'chk-holiday', 'CTCA0903R', {'BASS_DT': '20261001', 'CTX_AREA_FK': '', 'CTX_AREA_NK': ''}),
    ('ETF_구성종목', '/uapi/etfetn/v1/quotations/inquire-component-stock-price', 'FHKST121600C0', {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': '091160', 'FID_COND_SCR_DIV_CODE': '11216'}),
    ('해외지수_일봉', '/uapi/overseas-price/v1/quotations/inquire-daily-chartprice', 'FHKST03030100', {'FID_COND_MRKT_DIV_CODE': 'N', 'FID_INPUT_ISCD': 'SOX', 'FID_INPUT_DATE_1': '20260901', 'FID_INPUT_DATE_2': '20261015', 'FID_PERIOD_DIV_CODE': 'D'}),
    ('업종지수_일봉', Q + 'inquire-daily-indexchartprice', 'FHKUP03500100', {'FID_COND_MRKT_DIV_CODE': 'U', 'FID_INPUT_ISCD': '0001', 'FID_INPUT_DATE_1': '20260901', 'FID_INPUT_DATE_2': '20261015', 'FID_PERIOD_DIV_CODE': 'D'}),
]


def main():
    summary = {}
    try:
        tok = token()
    except Exception as e:
        summary['token'] = f'실패: {e!r}'[:300]
        json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
        return
    summary['token'] = '성공'
    for name, path, tr, params in TESTS:
        r = call(tok, path, tr, params)
        rows = {k: (len(v) if isinstance(v, list) else (1 if isinstance(v, dict) else 0)) for k, v in r.items() if k.startswith('output')}
        summary[name] = {'rt_cd': r.get('rt_cd'), 'msg': (r.get('msg1') or r.get('body') or '')[:120], 'rows': rows, 'http_error': r.get('http_error')}
        sample = {k: (v[:5] if isinstance(v, list) else v) for k, v in r.items()}
        json.dump(sample, open(os.path.join(OUT, name + '.json'), 'w'), ensure_ascii=False, indent=1)
        time.sleep(0.4)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
