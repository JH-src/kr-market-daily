"""52주 신고가 종목(코스피·코스닥, 스팩·우선주 제외).
- 기준: 그날 장중 고가가 직전 250거래일(그날 제외) 장중 고가의 최고치보다 높으면 신고가. 상장 250거래일 미만은 제외.
- 액면분할·병합·무상증자는 상장주식수 변화와 반대 방향 가격 변화로 찾아 과거 가격을 보정(sim/data.py와 같은 판단).
- 역대 신고가: 2010년 이후 자료(상장이 더 늦으면 상장 이후) 전체 최고치도 넘었는지 함께 표시.
- 거래대금 5억 미만은 '거래 적음'으로 표시(호가 한두 건으로 고가가 튈 수 있음).
- 자료: 거래소 일별(data/stk·ksq, archive), 거래소 자료가 아직 없는 날은 한투 장 마감 시세(data/kis/eod).
python3 new_high.py [YYYYMMDD] → docs/new_high/<날짜>.md, .csv"""
import glob, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import data as Dm

ROOT = Dm.ROOT
OUT = os.path.join(ROOT, 'docs', 'new_high')
COLS = ['BAS_DD', 'ISU_CD', 'ISU_NM', 'MKT_NM', 'TDD_CLSPRC', 'TDD_HGPRC', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS']


NUMS = ['TDD_CLSPRC', 'TDD_HGPRC', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS']


def _slim(d):
    d = d[~d['ISU_NM'].str.contains('스팩') & ~d['ISU_NM'].str.match(r'.*우[B-C]?$') & ~d['ISU_NM'].str.contains(r'\d+우|우\(')]
    out = pd.DataFrame({'BAS_DD': d['BAS_DD'].values, 'ISU_CD': d['ISU_CD'].values})
    for c in NUMS:
        out[c] = Dm.num(d[c]).astype('float64').values if c in d.columns else np.nan
    return out


def read(name, names):
    """archive(연도 묶음) + data(일별) 모두 읽고 같은 날은 data 우선, 거래소 자료 없는 날은 한투 장 마감 시세로 채움.
    메모리를 아끼려고 파일마다 숫자로 바꿔 쌓는다."""
    fs = sorted(glob.glob(os.path.join(ROOT, 'archive', name, '*.csv.gz'))) + sorted(glob.glob(os.path.join(ROOT, 'data', name, '20*', '*.csv.gz')))
    parts = []
    for f in fs:
        d = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=lambda c: c in COLS)
        for r in d.drop_duplicates('ISU_CD', keep='last')[['ISU_CD', 'ISU_NM', 'MKT_NM']].values:
            names[r[0]] = (r[1], r[2])
        parts.append(_slim(d))
    d = pd.concat(parts, ignore_index=True)
    have = set(d['BAS_DD'])
    mk = 'KOSPI' if name == 'stk' else 'KOSDAQ'
    for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'kis', 'eod', '20*.csv'))):
        if os.path.basename(f)[:8] in have:
            continue
        e = pd.read_csv(f, dtype=str, keep_default_na=False)
        e = e[e['MKT_NM'] == mk].copy()
        e['ISU_NM'] = e['ISU_CD'].map(lambda c: names.get(c, ('', ''))[0])
        parts = [d, _slim(e)]; d = pd.concat(parts, ignore_index=True)
    return d


def load():
    names = {}
    df = pd.concat([read('stk', names), read('ksq', names)], ignore_index=True)
    df = df.drop_duplicates(['BAS_DD', 'ISU_CD'], keep='last')
    df['BAS_DD'] = pd.to_datetime(df['BAS_DD'])
    W = {}
    for c, k in [('TDD_CLSPRC', 'close'), ('TDD_HGPRC', 'high'), ('ACC_TRDVAL', 'val'), ('MKTCAP', 'cap'), ('LIST_SHRS', 'shr')]:
        W[k] = df.pivot(index='BAS_DD', columns='ISU_CD', values=c).sort_index().astype('float32' if k != 'shr' else 'float64')
    del df
    nm = pd.Series({k: v[0] for k, v in names.items()}); mk = pd.Series({k: v[1] for k, v in names.items()})
    return W, nm, mk


def main():
    W, name, mkt = load()
    c = W['close'].where(W['close'] > 0); h = W['high'].where(W['high'] > 0); s = W['shr']
    day = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else c.index[-1]
    c, h, s = c.loc[:day], h.loc[:day], s.loc[:day]
    # 분할·병합 보정 계수: 그날 이전 가격에 곱할 값(주식수 비율의 역수를 뒤에서부터 누적)
    r = c / c.shift(1) - 1; rs = s / s.shift(1)
    adj = c * rs / c.shift(1) - 1
    split = ((rs - 1).abs() > 0.15) & (adj.abs() < 0.35) & (r * (rs - 1) < 0) & (r.abs() > 0.1)
    step = (1 / rs).where(split, 1.0).fillna(1.0)
    fac = step.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)   # 그날 다음부터의 분할 누적
    ha = h * fac
    today_h = ha.loc[day]
    prev = ha.iloc[:-1]
    n_prev = prev.notna().iloc[-250:].sum()
    hi250 = prev.iloc[-250:].max()
    hi_all = prev.max()
    ok = today_h.notna() & (n_prev >= 250) & (today_h > hi250)
    codes = ok[ok].index
    first = c.notna().idxmax()
    T = pd.DataFrame({
        '종목': name.reindex(codes), '시장': mkt.reindex(codes).str.replace('KOSDAQ GLOBAL', 'KOSDAQ'),
        '종가': c.loc[day, codes], '등락률': (c.loc[day, codes] / (c.iloc[-2][codes] * fac.iloc[-2][codes]) - 1) * 100,
        '고가': h.loc[day, codes], '직전 52주 최고': hi250[codes] / fac.loc[day, codes],
        '역대 신고가': np.where(today_h[codes] > hi_all[codes], 'O', ''),
        '자료 시작': [('2010 이전' if first[x] <= c.index[0] else first[x].strftime('%Y.%m')) for x in codes],
        '시가총액(억)': W['cap'].loc[day, codes] / 1e8, '거래대금(억)': W['val'].loc[day, codes] / 1e8,
    })
    T['비고'] = np.where(T['거래대금(억)'] < 5, '거래 적음', '')
    T.index.name = '코드'
    T = T.sort_values('시가총액(억)', ascending=False)
    os.makedirs(OUT, exist_ok=True)
    d = day.strftime('%Y%m%d')
    T.round(2).to_csv(os.path.join(OUT, d + '.csv'), encoding='utf-8-sig')
    n_all = int((n_prev >= 250).sum())
    md = [f'# {day:%Y-%m-%d} 52주 신고가', '',
          f'{len(T)}종목(대상 {n_all}종목, 코스피 {int((T["시장"] == "KOSPI").sum())} · 코스닥 {int((T["시장"] == "KOSDAQ").sum())}), 이 중 역대 신고가 {int((T["역대 신고가"] == "O").sum())}종목', '']
    show = T.copy()
    for col in ['종가', '고가', '직전 52주 최고', '시가총액(억)', '거래대금(억)']:
        show[col] = show[col].map(lambda v: f'{v:,.0f}')
    show['등락률'] = show['등락률'].map(lambda v: f'{v:+.1f}%')
    md.append(show.reset_index().to_markdown(index=False))
    open(os.path.join(OUT, d + '.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md[:3]))
    return T


if __name__ == '__main__':
    main()
