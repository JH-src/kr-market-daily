"""테마 섹터 분류: 국내 주식형 ETF 전체 구성종목(이름으로 테마 판정) + DART 산업분류(KSIC)로 세부 구분.
여러 테마 ETF에 반복 편입된 종목은 그 테마로, 나머지는 산업분류 기반 섹터로 둔다.
현재 편입 기준을 과거에도 적용하는 한계가 있다(시점별 편입 이력 없음)."""
import glob, os, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THEMES = [  # (ETF 이름 패턴, 테마) — 위에서부터 먼저 맞는 것
    (r'반도체.*(소부장|장비|소재|부품)|소부장', '반도체 소부장'), (r'반도체|메모리|HBM', '반도체'),
    (r'2차전지|이차전지|배터리|양극재|리튬|전고체', '2차전지'), (r'조선|해운', '조선'), (r'방산|방위|K-?방산|우주|항공', '방산·우주'),
    (r'원자력|원전|SMR', '원자력'), (r'전력|변압기|전선|그리드|에너지인프라', '전력기기'),
    (r'바이오|헬스케어|제약|의료', '바이오·헬스케어'), (r'화장품|뷰티|K-?뷰티', '화장품'), (r'게임', '게임'),
    (r'엔터|K-?POP|KPOP|콘텐츠|미디어|웹툰', '엔터·콘텐츠'), (r'로봇|휴머노이드|자동화', '로봇'),
    (r'자동차|모빌리티|자율주행', '자동차'), (r'은행', '은행'), (r'증권', '증권'), (r'보험', '보험'),
    (r'건설|인프라', '건설'), (r'철강|비철|금속', '철강·금속'), (r'화학|정유|석유', '화학·정유'),
    (r'인터넷|플랫폼|소프트웨어|SW|클라우드', '인터넷·소프트웨어'), (r'K-?푸드|음식료|식품|라면', '음식료'),
    (r'지주|홀딩스', '지주'),
]


def etf_themes():
    fs = sorted(glob.glob(os.path.join(ROOT, 'data', 'kis', 'etf_all', '*.csv')))
    if not fs:
        return pd.DataFrame(columns=['code', 'theme', 'n_etf', 'w'])
    h = pd.read_csv(fs[-1], dtype=str)
    h = h[h['code'].str.fullmatch(r'\d{6}', na=False)]
    h['w'] = pd.to_numeric(h['weight'], errors='coerce').fillna(0)
    def th(nm):
        for pat, t in THEMES:
            if re.search(pat, nm, flags=re.I):
                return t
        return None
    h['theme'] = h['etf_name'].map(th)
    h = h.dropna(subset=['theme'])
    g = h.groupby(['code', 'theme']).agg(n_etf=('etf', 'nunique'), w=('w', 'sum')).reset_index()
    return g


def classify(ksic_sector, induty):
    """ksic_sector: 종목코드 -> 산업분류 기반 섹터, induty: 종목코드 -> KSIC 코드."""
    g = etf_themes()
    g = g[(g['n_etf'] >= 2) | (g['w'] >= 5)]
    best = g.sort_values(['n_etf', 'w'], ascending=False).drop_duplicates('code').set_index('code')['theme']
    out = ksic_sector.copy()
    for c, t in best.items():
        out[c] = t
    # 반도체 안 세부: 장비(KSIC 292 특수목적기계·272 측정검사)·소재(20·22·24)·부품(262)·소자(261)
    for c in out.index:
        if out[c] in ('반도체', '반도체 소부장'):
            k = str(induty.get(c, ''))
            if k.startswith(('292', '272', '271')):
                out[c] = '반도체 장비'
            elif k.startswith(('20', '22', '24', '23')):
                out[c] = '반도체 소재'
            elif k.startswith('262'):
                out[c] = '반도체 부품'
            else:
                out[c] = '반도체 소자'
    return out
