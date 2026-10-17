"""상장사 최근 사업보고서 '사업의 내용'에서 테마 키워드 빈도와 사업 개요 요약을 뽑는다(섹터 분류용).
결과: data/dart/business.csv (이어받기). 현재 상장 종목 기준."""
import csv, io, os, re, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import dart_fin as D

KW = {
    '반도체': ['반도체', '웨이퍼', 'DRAM', '디램', '낸드', 'NAND', 'HBM', '파운드리', '팹리스', '시스템반도체', '메모리반도체'],
    '반도체장비': ['증착', '식각', '세정장비', '검사장비', '테스트 소켓', '테스트소켓', '프로브카드', '번인', '노광', '후공정 장비', '반도체 장비', '반도체장비'],
    '반도체소재': ['포토레지스트', '전구체', '슬러리', '특수가스', '반도체 소재', '반도체소재', '블랭크마스크', '웨이퍼 소재'],
    '2차전지': ['2차전지', '이차전지', '양극재', '음극재', '전해액', '분리막', '배터리', '리튬'],
    '디스플레이': ['디스플레이', 'OLED', 'LCD', '패널'],
    '조선': ['선박', '조선', '해양플랜트', 'LNG선', '선박엔진'],
    '방산': ['방산', '방위산업', '국방', '유도무기', '탄약', '군수', '무기체계'],
    '원자력': ['원전', '원자력', '원자로'],
    '전력기기': ['변압기', '차단기', '전력기기', '송배전', '배전반', '전력케이블', '전선'],
    '자동차': ['자동차부품', '완성차', '차체', '전기차'],
    '바이오': ['신약', '임상', '항체', '바이오시밀러', '의약품', '세포치료'],
    '의료기기': ['의료기기', '체외진단', '임플란트', '진단키트'],
    '화장품': ['화장품', '코스메틱', '스킨케어'],
    '게임': ['게임'],
    '엔터': ['엔터테인먼트', '음반', '아티스트', '드라마', '영화', '공연'],
    '로봇': ['로봇', '협동로봇', '자동화설비'],
    '우주항공': ['항공기', '위성', '우주', '발사체'],
    '음식료': ['식품', '라면', '음료', '제과', '주류'],
    '통신장비': ['통신장비', '5G', '기지국', '광통신'],
    '인터넷·소프트웨어': ['플랫폼', '소프트웨어', '클라우드', 'SaaS', '인공지능'],
    'PCB·기판': ['PCB', '인쇄회로기판', '패키지기판', 'FC-BGA', '연성회로기판', 'FPCB'],
    '스마트폰부품': ['스마트폰', '카메라모듈', '폴더블', '힌지', '휴대폰'],
    '수동부품': ['MLCC', '적층세라믹', '커패시터', '인덕터', '커넥터'],
    '자동차전장': ['전장', '차량용', '자율주행', 'ADAS', '인포테인먼트'],
    'RF·무선': ['RF', '안테나', '중계기', '무선통신', '레이더'],
}
OUT = os.path.join(D.OUT, os.environ.get('BUS_OUT', 'business.csv'))
ONLY = os.environ.get('BUS_ONLY', '')   # 쉼표 구분 종목코드 파일 경로(지정 시 그 종목만)


def text_of(rcept):
    key = os.environ['DART_KEY']
    for i in range(3):
        try:
            import urllib.request
            b = urllib.request.urlopen(f"https://opendart.fss.or.kr/api/document.xml?crtfc_key={key}&rcept_no={rcept}", timeout=180).read()
            z = zipfile.ZipFile(io.BytesIO(b))
            name = sorted(z.namelist(), key=len)[0]
            x = z.read(name).decode('utf-8', 'ignore')
            t = re.sub(r'<[^>]+>', ' ', x); t = re.sub(r'\s+', ' ', t)
            idx = [m.start() for m in re.finditer('사업의 개요', t)]
            i = idx[1] if len(idx) > 1 else (idx[0] if idx else 0)
            j = t.find('재무에 관한 사항', i)
            return t[i:j if j > i else i + 60000]
        except Exception:
            time.sleep(5 * (i + 1))
    return ''


def one(row):
    key = os.environ['DART_KEY']
    sc, cc = row
    d = D._get(f"https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&corp_code={cc}&bgn_de=20240101&end_de=20261231&pblntf_detail_ty=A001&page_count=10")
    lst = d.get('list') or []
    if not lst:
        return {'stock_code': sc, 'rcept_no': '', 'len': 0}
    rc = lst[0]['rcept_no']
    sec = text_of(rc)
    r = {'stock_code': sc, 'rcept_no': rc, 'len': len(sec), 'summary': sec[6:400].replace(',', ' ')}
    for k, ws in KW.items():
        r[k] = sum(sec.count(w) for w in ws)
    return r


def main():
    cmap = D.corp_map()
    codes = [l.split(',')[0] for l in open(os.path.join(D.ROOT, 'config', 'universe_market.csv'), encoding='utf-8')]
    have = set()
    if os.path.exists(OUT):
        have = {r['stock_code'] for r in csv.DictReader(open(OUT, encoding='utf-8'))}
    if ONLY:
        keep = set(open(ONLY).read().split())
        codes = [c for c in codes if c in keep]
    todo = [(c, cmap[c]) for c in codes if c in cmap and c not in have]
    cols = ['stock_code', 'rcept_no', 'len', 'summary'] + list(KW)
    f = open(OUT, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, cols)
    if not have:
        w.writeheader()
    n = 0
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(one, todo):
            w.writerow(r); n += 1
            if n % 50 == 0:
                f.flush(); print(n, len(todo), flush=True)
    f.close(); print('ALL_DONE', n)


if __name__ == '__main__':
    main()
