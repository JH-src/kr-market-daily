"""테마 ETF 자금 흐름(섹터 판정 후보): ETF 이름으로 테마를 붙이고 상장좌수 변화 × 기준가로 설정·환매를 계산해 섹터별로 모은다.
레버리지·인버스·선물형은 뺌. 과거 전 기간 계산 가능(ETF 이름-테마 연결은 시점과 무관).
결과: 날짜×섹터 표 (flow20_pct: 20일 순유입 / 테마 ETF 순자산 %, flow60_pct, new_etf180: 최근 180일 신규 상장 테마 ETF 수)."""
import re
import numpy as np
import pandas as pd
from sim import data as Dm
from sim.theme import THEMES

TO_SECTORS = {
    '반도체': ['반도체 소자', '반도체 장비', '반도체 소재', '반도체 부품'], '반도체 소부장': ['반도체 장비', '반도체 소재', '반도체 부품'],
    '2차전지': ['2차전지'], '조선': ['조선'], '방산·우주': ['방산·우주', '항공·방산'], '원자력': ['원자력'], '전력기기': ['전력기기', '전기장비'],
    '바이오·헬스케어': ['바이오·헬스케어', '제약·바이오', '바이오연구', '의료기기'], '화장품': ['화장품'], '게임': ['게임', '소프트웨어·게임'],
    '엔터·콘텐츠': ['엔터·콘텐츠', '미디어·콘텐츠'], '로봇': ['로봇'], '자동차': ['자동차', '자동차부품', '자동차 전장·부품'], '은행': ['은행', '금융'],
    '증권': ['증권'], '보험': ['보험'], '건설': ['건설', '건자재'], '철강·금속': ['철강', '비철금속', '금속가공', '철강·금속'],
    '화학·정유': ['화학', '화학·정유', '정유'], '인터넷·소프트웨어': ['인터넷·소프트웨어', '소프트웨어·게임', '인터넷·IT서비스'],
    '음식료': ['음식료'], '지주': ['지주'],
}
EXCL = re.compile(r'레버리지|인버스|2X|선물|커버드|콜|채권|머니|금리|달러|미국|차이나|중국|일본|인도|베트남|글로벌|S&P|나스닥|TSMC|엔비디아|테슬라|빅테크', re.I)


def theme_of(nm):
    for pat, t in THEMES:
        if re.search(pat, nm, flags=re.I):
            return t
    return None


def build():
    e = Dm._read('etf', ['BAS_DD', 'ISU_CD', 'ISU_NM', 'TDD_CLSPRC', 'NAV', 'LIST_SHRS'])
    e = e[~e['ISU_NM'].str.contains(EXCL)]
    e['theme'] = e['ISU_NM'].map(theme_of)
    e = e.dropna(subset=['theme'])
    for c in ('TDD_CLSPRC', 'NAV', 'LIST_SHRS'):
        e[c] = pd.to_numeric(e[c].str.replace(',', ''), errors='coerce')
    e['date'] = pd.to_datetime(e['BAS_DD'])
    e = e.sort_values(['ISU_CD', 'date'])
    nav = e['NAV'].where(e['NAV'] > 0, e['TDD_CLSPRC'])
    e['flow'] = e.groupby('ISU_CD')['LIST_SHRS'].diff() * nav
    e['aum'] = e['LIST_SHRS'] * nav
    first = e.groupby('ISU_CD')['date'].min()
    g = e.groupby(['date', 'theme']).agg(flow=('flow', 'sum'), aum=('aum', 'sum')).reset_index()
    out = []
    for th, x in g.groupby('theme'):
        x = x.set_index('date').sort_index()
        f20 = x['flow'].rolling(20, min_periods=10).sum(); f60 = x['flow'].rolling(60, min_periods=30).sum()
        a = x['aum'].rolling(20, min_periods=10).mean()
        codes = e[e['theme'] == th]['ISU_CD'].unique()
        fd = first.reindex(codes)
        new180 = pd.Series([(fd > d - pd.Timedelta(days=180)).sum() - (fd > d).sum() for d in x.index], index=x.index)
        out.append(pd.DataFrame({'theme': th, 'flow20_pct': f20 / a * 100, 'flow60_pct': f60 / a * 100, 'new_etf180': new180}))
    t = pd.concat(out).reset_index()
    rows = []
    for th, secs in TO_SECTORS.items():
        x = t[t['theme'] == th]
        for s in secs:
            rows.append(x.assign(sector=s))
    s = pd.concat(rows) if rows else pd.DataFrame()
    # 한 섹터에 테마가 둘 이상이면 평균
    return s.groupby(['date', 'sector'])[['flow20_pct', 'flow60_pct', 'new_etf180']].mean().reset_index()
