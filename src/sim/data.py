"""시뮬레이션용 자료 적재: 거래소 시세(연도 보관 + 최근 일별), 지수, DART 실적·공시, 산업분류.
결과는 넓은 표(날짜 x 종목)로 만들어 작업 폴더에 캐시한다."""
import glob, gzip, os, pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
CACHE = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
os.makedirs(CACHE, exist_ok=True)


def _read(name, cols):
    fs = sorted(glob.glob(os.path.join(ROOT, 'archive', name, '*.csv.gz')))
    have_years = {os.path.basename(f)[:4] for f in fs}
    fs += [f for f in sorted(glob.glob(os.path.join(DATA, name, '20*', '*.csv.gz'))) if os.path.basename(f)[:4] not in have_years]
    out = []
    for f in fs:
        d = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=lambda c: c in cols)
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')


def prices(refresh=False):
    cp = os.path.join(CACHE, 'prices.pkl')
    if os.path.exists(cp) and not refresh:
        return pickle.load(open(cp, 'rb'))
    cols = ['BAS_DD', 'ISU_CD', 'ISU_NM', 'MKT_NM', 'TDD_CLSPRC', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS']
    df = pd.concat([_read('stk', cols), _read('ksq', cols)], ignore_index=True)
    df = df[~df['ISU_NM'].str.contains('스팩') & ~df['ISU_NM'].str.match(r'.*우[B-C]?$') & ~df['ISU_NM'].str.contains(r'\d+우|우\(')]
    df = df.drop_duplicates(['BAS_DD', 'ISU_CD'])
    for c in ['TDD_CLSPRC', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS']:
        df[c] = num(df[c]).astype('float64')
    df['BAS_DD'] = pd.to_datetime(df['BAS_DD'])
    W = {}
    for c, k in [('TDD_CLSPRC', 'close'), ('ACC_TRDVAL', 'val'), ('MKTCAP', 'cap'), ('LIST_SHRS', 'shr')]:
        W[k] = df.pivot(index='BAS_DD', columns='ISU_CD', values=c).sort_index()
    W['close'] = W['close'].where(W['close'] > 0)
    names = df.sort_values('BAS_DD').drop_duplicates('ISU_CD', keep='last').set_index('ISU_CD')
    W['name'] = names['ISU_NM']; W['mkt'] = names['MKT_NM']
    # 수정 수익률: 상장주식수가 크게 바뀌고 가격이 그만큼 반대로 움직였으면 분할·병합으로 보고 보정
    c, s = W['close'], W['shr']
    r = c / c.shift(1) - 1
    rs = s / s.shift(1)
    adj = c * rs / c.shift(1) - 1
    # 분할·병합·무상증자로 보려면 주식수 변화와 반대 방향으로 가격이 실제로 움직였어야 한다.
    # (가격 변동 없이 주식수만 늘어난 날 = 유상증자·합병·신주 상장 -> 보정하지 않고 실제 수익률 사용)
    split = ((rs - 1).abs() > 0.15) & (adj.abs() < 0.35) & (r * (rs - 1) < 0) & (r.abs() > 0.1)
    r = r.where(~split, adj)
    lim = pd.Series(np.where(r.index < pd.Timestamp('2015-06-15'), 0.16, 0.31), index=r.index)
    bad = r.abs().gt(lim, axis=0) & ~split
    r = r.mask(bad)              # 설명 안 되는 급변(자료 오류·감자 등)은 그날 수익률 제외
    W['ret'] = r
    # 상장 기간 안에서만 지수화(상장폐지 뒤는 마지막 값 유지 -> 정리매매 손실 반영)
    alive = c.notna()
    P = (1 + r.fillna(0)).cumprod().where(alive.cummax())
    W['adj'] = P.ffill()
    pickle.dump(W, open(cp, 'wb'))
    return W


def index_series(name, idx_nm):
    d = _read(name, ['BAS_DD', 'IDX_NM', 'CLSPRC_IDX'])
    d = d[d['IDX_NM'].str.strip() == idx_nm]
    s = pd.Series(num(d['CLSPRC_IDX']).values, index=pd.to_datetime(d['BAS_DD'])).sort_index()
    return s[~s.index.duplicated()].dropna()


SECTOR_KSIC = [  # (접두 코드, 섹터) — 가장 긴 접두가 우선
    ('261', '반도체'), ('262', '디스플레이·전자부품'), ('263', '컴퓨터·통신장비'), ('264', '컴퓨터·통신장비'), ('265', '컴퓨터·통신장비'),
    ('282', '2차전지'), ('281', '전력기기'), ('283', '전력기기'), ('284', '전기장비'), ('285', '전기장비'), ('289', '전기장비'),
    ('301', '자동차'), ('302', '자동차'), ('303', '자동차부품'), ('311', '조선'), ('252', '방산'), ('313', '항공·방산'),
    ('29', '기계'), ('27', '의료·정밀기기'), ('21', '제약·바이오'), ('2042', '화장품'), ('20', '화학'), ('19', '정유'),
    ('10', '음식료'), ('11', '음식료'), ('12', '음식료'), ('13', '섬유·의복'), ('14', '섬유·의복'), ('15', '섬유·의복'),
    ('241', '철강'), ('242', '비철금속'), ('243', '철강'), ('22', '화학'), ('23', '건자재'), ('25', '금속가공'),
    ('41', '건설'), ('42', '건설'), ('49', '운송'), ('50', '운송'), ('51', '운송'), ('52', '운송'),
    ('45', '유통'), ('46', '유통'), ('47', '유통'), ('55', '레저'), ('56', '레저'), ('91', '레저'),
    ('58', '소프트웨어·게임'), ('62', '소프트웨어·게임'), ('63', '인터넷·IT서비스'), ('59', '미디어·콘텐츠'), ('60', '미디어·콘텐츠'),
    ('61', '통신'), ('35', '유틸리티'), ('36', '유틸리티'), ('64', '금융'), ('65', '금융'), ('66', '금융'),
    ('715', '지주'), ('70', '바이오연구'), ('72', '엔지니어링'), ('73', '전문서비스'), ('75', '사업지원'),
    ('68', '부동산'), ('85', '교육'), ('86', '의료서비스'), ('31', '운송장비'), ('32', '기타제조'), ('33', '기타제조'),
    ('16', '목재·종이'), ('17', '목재·종이'), ('18', '인쇄'), ('01', '농림수산'), ('03', '농림수산'), ('05', '광업'), ('06', '광업'), ('07', '광업'), ('08', '광업'),
]


def sectors(level='coarse'):
    p = os.path.join(DATA, 'dart', 'industry.csv')
    d = pd.read_csv(p, dtype=str).dropna(subset=['induty_code'])
    if level == 'ksic3':
        # 산업분류 앞 3자리(소분류 수준, 예: 261 반도체·262 전자부품·2929 특수목적 기계 -> 292)로 잘게 나눔
        return d.set_index('stock_code')['induty_code'].str[:3]
    rules = sorted(SECTOR_KSIC, key=lambda x: -len(x[0]))
    def m(code):
        for pre, sec in rules:
            if code.startswith(pre):
                return sec
        return '기타'
    return d.set_index('stock_code')['induty_code'].map(m)


def fin_quarters():
    """분기(3개월) 매출·영업이익과 공시일. 사업보고서는 연간에서 1~3분기를 빼 4분기로 환산."""
    rows = []
    for f in glob.glob(os.path.join(DATA, 'dart', 'fin', '*.csv')):
        y, rc = os.path.basename(f)[:-4].split('_')
        d = pd.read_csv(f, dtype={'stock_code': str, 'rcept_no': str})
        d = d[d['fs'].isin(['CFS', 'OFS'])].copy()
        d['y'], d['q'] = int(y), {'11013': 1, '11012': 2, '11014': 3, '11011': 4}[rc]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    f = pd.concat(rows, ignore_index=True)
    f['avail'] = pd.to_datetime(f['rcept_no'].str[:8], errors='coerce')
    q = f[f['q'] < 4].set_index(['stock_code', 'y', 'q'])
    a = f[f['q'] == 4]
    add = []
    for _, r in a.iterrows():
        try:
            parts = q.loc[[(r['stock_code'], r['y'], k) for k in (1, 2, 3)]]
        except KeyError:
            continue
        if len(parts) != 3 or parts['fs'].nunique() != 1 or parts['fs'].iloc[0] != r['fs']:
            continue
        e = {'stock_code': r['stock_code'], 'y': r['y'], 'q': 4, 'fs': r['fs'], 'avail': r['avail']}
        for k in ['rev', 'rev_py', 'op', 'op_py']:
            e[k] = r[k] - parts[k].sum() if pd.notna(r[k]) and parts[k].notna().all() else np.nan
        add.append(e)
    out = pd.concat([f[f['q'] < 4], pd.DataFrame(add)], ignore_index=True)
    out['ord'] = out['y'] * 4 + out['q']
    return out.dropna(subset=['avail']).sort_values(['stock_code', 'ord'])


def events():
    p = os.path.join(DATA, 'dart', 'events_major.csv')
    d = pd.read_csv(p, dtype=str)
    d['dt'] = pd.to_datetime(d['rcept_dt'])
    return d


def fin_ext():
    """분기 순이익(3개월, 4분기는 연간-1~3분기), 분기 말 자본·부채와 공시일."""
    rows = []
    for f in glob.glob(os.path.join(DATA, 'dart', 'fin_ext', '*.csv')):
        y, rc = os.path.basename(f)[:-4].split('_')
        d = pd.read_csv(f, dtype={'stock_code': str, 'rcept_no': str})
        d['y'], d['q'] = int(y), {'11013': 1, '11012': 2, '11014': 3, '11011': 4}[rc]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    f = pd.concat(rows, ignore_index=True)
    f['avail'] = pd.to_datetime(f['rcept_no'].str[:8], errors='coerce')
    q = f[f['q'] < 4].set_index(['stock_code', 'y', 'q'])['ni']
    a = f[f['q'] == 4].copy()
    def q4(r):
        try:
            p = q.loc[[(r['stock_code'], r['y'], k) for k in (1, 2, 3)]]
        except KeyError:
            return np.nan
        return r['ni'] - p.sum() if len(p) == 3 and p.notna().all() and pd.notna(r['ni']) else np.nan
    a['ni'] = a.apply(q4, axis=1) if len(a) else np.nan
    out = pd.concat([f[f['q'] < 4], a], ignore_index=True)
    out['ord'] = out['y'] * 4 + out['q']
    return out.dropna(subset=['avail']).sort_values(['stock_code', 'ord'])
