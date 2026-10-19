"""세분화 섹터 대표 ETF 추세로 주도 섹터 판정 검증.
세부 섹터마다 대표 ETF 묶음(레버리지·인버스 제외)을 정하고, 순자산 가중 일간 수익률로 섹터 지수를 만든다.
신호: 20일선 위, 10주선 위(금요일 종가 기준), 둘 다, 코스피 대비 상대강도선이 20일선·10주선 위, 최근 돌파 등.
결과: 신호가 켜진 섹터의 이후 1·3·6개월 성과 - 같은 날 전체 세부 섹터 평균. 연도별로 플러스 해와 평균 차이를 본다.
python3 -m sim.etf_trend  → docs/sim/etf_trend.md, data/sim/etf_trend_daily.csv(오늘 판정)"""
import os
import numpy as np
import pandas as pd
from sim import data as Dm

SEG = {
    '반도체(전체)': ['091160', '091230', '396500', '395270'],
    '반도체 대형(TOP2)': ['395160', '0167A0', '469150', '0210A0', '0182R0'],
    '반도체 소부장·장비': ['455850', '471990', '475300', '471760', '476260'],
    '반도체 후공정': ['475310'],
    '2차전지': ['305720', '305540', '364980', '465330'],
    '2차전지 소재': ['462010', '461950', '455860', '466810'],
    '전력기기·인프라': ['487240', '0117V0', '0101N0', '491820', '0209Z0'],
    '원자력': ['434730', '0091P0', '433500', '0098F0', '0092B0'],
    '조선': ['466920', '494670', '0115D0', '441540', '0141S0'],
    '방산': ['449450', '463250', '0080G0', '490480'],
    '우주항공': ['421320'],
    '자동차': ['091180', '138540', '466930', '464600'],
    '로봇': ['0148J0', '469070', '0177X0', '0155N0'],
    '바이오': ['244580', '364970'],
    '헬스케어': ['143860', '266420', '253280'],
    '의료기기': ['307510', '464610'],
    '화장품': ['228790', '0008T0', '479850'],
    '음식료': ['438900', '266410'],
    '증권': ['102970', '157500'],
    '은행': ['091170', '091220'],
    '보험': ['140700'],
    '건설': ['117700'],
    '철강': ['117680'],
    '에너지화학': ['117460'],
    '기계장비': ['102960'],
    '운송': ['140710'],
    '지주회사': ['307520'],
    '게임': ['300950', '364990', '300640', '300610', '395280'],
    '미디어·엔터': ['228810', '266360', '475050', '395290'],
    '인터넷': ['365000'],
    '소프트웨어': ['157490', '0105D0'],
    'IT(정보기술)': ['266370', '326240'],
    '여행레저': ['228800'],
    '신재생·태양광': ['377990', '457990'],
    '수소': ['367770'],
    '네트워크·통신장비': ['367760', '367740'],
    '경기소비재': ['266390'],
}
HZ = {'1개월': 21, '3개월': 63, '6개월': 126}


def build():
    codes = sorted({c for v in SEG.values() for c in v})
    e = Dm._read('etf', ['BAS_DD', 'ISU_CD', 'TDD_CLSPRC', 'MKTCAP'])
    e = e[e['ISU_CD'].isin(codes)].copy()
    for c in ('TDD_CLSPRC', 'MKTCAP'):
        e[c] = pd.to_numeric(e[c].astype(str).str.replace(',', ''), errors='coerce')
    e['date'] = pd.to_datetime(e['BAS_DD'])
    px = e.pivot_table(index='date', columns='ISU_CD', values='TDD_CLSPRC').sort_index()
    cap = e.pivot_table(index='date', columns='ISU_CD', values='MKTCAP').reindex(px.index)
    r = px.pct_change(fill_method=None)
    r = r.where(r.abs() < 0.35)  # 분할·오류 튐 제거
    w = cap.shift(1)
    idx = {}
    for s, cs in SEG.items():
        cs = [c for c in cs if c in r.columns]
        if not cs:
            continue
        rr, ww = r[cs], w[cs].where(r[cs].notna())
        ret = (rr * ww).sum(axis=1) / ww.sum(axis=1)
        ret = ret.where(ww.sum(axis=1) > 0)
        first = ret.first_valid_index()
        lv = (1 + ret.fillna(0)).cumprod()
        lv[lv.index < first] = np.nan
        # 상장 뒤 60거래일은 이동평균이 불안정하므로 신호에서 빼기 위해 표시만 남김
        idx[s] = lv
    return pd.DataFrame(idx)


def signals(L, kospi):
    k = kospi.reindex(L.index).ffill()
    S = {}
    ma20 = L.rolling(20, min_periods=20).mean()
    wk = L.resample('W-FRI').last()
    ma10w = wk.rolling(10, min_periods=10).mean().reindex(L.index, method='ffill')
    wkclose = wk.reindex(L.index, method='ffill')
    rs = L.div(k, axis=0)
    rs20 = rs.rolling(20, min_periods=20).mean()
    rsw = rs.resample('W-FRI').last()
    rs10w = rsw.rolling(10, min_periods=10).mean().reindex(L.index, method='ffill')
    age = L.notna().cumsum()
    ok = age >= 60
    S['20일선 위'] = (L > ma20)
    S['10주선 위(주봉)'] = (wkclose > ma10w)
    S['20일선·10주선 둘 다'] = S['20일선 위'] & S['10주선 위(주봉)']
    S['상대강도 20일선 위'] = (rs > rs20)
    S['상대강도 10주선 위'] = (rsw.reindex(L.index, method='ffill') > rs10w)
    S['상대강도 둘 다'] = S['상대강도 20일선 위'] & S['상대강도 10주선 위']
    S['절대·상대 넷 다'] = S['20일선·10주선 둘 다'] & S['상대강도 둘 다']
    up20 = S['20일선 위'].astype(float)
    S['20일선 최근 돌파(10일 내)'] = (up20.diff().rolling(10, min_periods=1).max() > 0) & S['20일선 위']
    S['10주선 위 + 20일선 이격 0~8%'] = S['10주선 위(주봉)'] & ((L / ma20 - 1) >= 0) & ((L / ma20 - 1) <= 0.08)
    S['20일선 기울기 상승 + 위'] = S['20일선 위'] & (ma20 > ma20.shift(5))
    return {n: (v & ok).where(L.notna() & ok) for n, v in S.items()}


def evaluate(L, S):
    me = L.groupby(L.index.to_period('M')).apply(lambda x: x.index.max())
    dates = [pd.Timestamp(d) for d in me.values]
    pos = {d: i for i, d in enumerate(L.index)}
    rows = []
    for name, sig in S.items():
        for hz, h in HZ.items():
            fwd = L.shift(-h) / L - 1
            for d in dates:
                i = pos[d]
                if i + h >= len(L):
                    continue
                f = fwd.loc[d]; s = sig.loc[d]
                v = f.notna() & s.notna()
                if v.sum() < 6:
                    continue
                ex = (f[v] - f[v].mean()) * 100
                on = s[v].astype(bool)
                if on.sum() == 0 or (~on).sum() == 0:
                    continue
                rows.append({'신호': name, '기간': hz, 'date': d, 'year': d.year, 'on': ex[on].mean(), 'off': ex[~on].mean(),
                             'n_on': int(on.sum()), 'n': int(v.sum())})
    return pd.DataFrame(rows)


def summarize(R):
    out = []
    for (name, hz), x in R.groupby(['신호', '기간']):
        y = x.groupby('year').apply(lambda z: (z['on'] - z['off']).mean())
        yo = x.groupby('year')['on'].mean()
        out.append({'신호': name, '기간': hz, '켜진 섹터 초과(%p)': x['on'].mean(), '켜짐-꺼짐(%p)': (x['on'] - x['off']).mean(),
                    '플러스 해': f"{(y > 0).sum()}/{len(y)}", '켜진 섹터 플러스 해': f"{(yo > 0).sum()}/{len(yo)}",
                    '2024~ 켜짐-꺼짐': (x[x.year >= 2024]['on'] - x[x.year >= 2024]['off']).mean(),
                    '평균 켜진 수/전체': f"{x['n_on'].mean():.1f}/{x['n'].mean():.1f}", '개월': len(x)})
    return pd.DataFrame(out)


def main():
    L = build()
    kospi = Dm.index_series('idx_kospi', '코스피')
    S = signals(L, kospi)
    R = evaluate(L, S)
    sm = summarize(R)
    first = L.apply(lambda c: c.first_valid_index())
    cov = L.notna().sum(axis=1).resample('YE').last()
    md = ['# 세분화 섹터 대표 ETF 추세로 본 주도 섹터 판정',
          '', f'세부 섹터 {L.shape[1]}개, 매월 말 판정 → 이후 성과를 같은 날 전체 세부 섹터 평균과 비교(%p). 섹터 지수는 대표 ETF 순자산 가중.',
          '켜짐-꺼짐: 신호가 켜진 섹터 평균 초과 - 꺼진 섹터 평균 초과. 플러스 해: 연도별 켜짐-꺼짐이 플러스인 해.', '']
    for hz in HZ:
        md += [f'## {hz} 뒤', '', sm[sm['기간'] == hz].drop(columns='기간').sort_values('켜짐-꺼짐(%p)', ascending=False).round(2).to_markdown(index=False), '']
    yr = R[R['기간'] == '3개월'].groupby(['신호', 'year']).apply(lambda z: (z['on'] - z['off']).mean()).unstack().round(1)
    md += ['## 연도별 켜짐-꺼짐(3개월, %p)', '', yr.to_markdown(), '', '## 연말 기준 살아 있는 세부 섹터 수', '',
           cov.rename(lambda d: d.year).to_frame('섹터 수').T.to_markdown(), '', '## 세부 섹터 시작일', '',
           first.dt.strftime('%Y-%m-%d').to_frame('시작').to_markdown(), '']
    # 오늘 판정
    last = L.index.max()
    today = pd.DataFrame({n: s.loc[last] for n, s in S.items() if n in ('20일선 위', '10주선 위(주봉)', '상대강도 20일선 위', '상대강도 10주선 위')})
    ma20 = L.rolling(20).mean().loc[last]
    today['20일선 이격(%)'] = ((L.loc[last] / ma20 - 1) * 100).round(1)
    today['1개월(%)'] = ((L.loc[last] / L.shift(21).loc[last] - 1) * 100).round(1)
    md += [f'## 현재 판정({last:%Y-%m-%d})', '', today.sort_values('1개월(%)', ascending=False).to_markdown(), '']
    os.makedirs('../docs/sim', exist_ok=True)
    open('../docs/sim/etf_trend.md', 'w').write('\n'.join(md))
    print('\n'.join(md[:40]))


if __name__ == '__main__':
    main()
