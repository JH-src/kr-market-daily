"""공포 바닥 확인 → 직전 주도 섹터에서 많이 빠진 종목 매수 (2026-10-28 사전 등록, 결과 보고 기준 바꾸지 않음)

모든 판단은 신호일까지 알 수 있던 자료만 사용.
- 신호: 코스피 공포·탐욕 지수가 L 이하로 내려갔다가 L 위로 올라온 날(L = 35, 30, 25, 20). 같은 L에서 직전 신호 뒤 20거래일 안의 신호는 무시.
- t0(공포 시작 전): 이번 하락 구간에서 지수가 L 이하로 들어가기 직전 날.
- 주도 섹터(t0 기준, 산업분류 대분류 약 40개, 대상 종목 5개 이상): ① 126거래일 섹터 수익률(시총 가중) - 시장 ② 63거래일 같은 것 ③ 거래대금 비중(60일 평균 / 250일 평균) 세 순위의 평균 상위 3개.
- 대상 종목(신호일): 시총 상위 700 & 20일 평균 거래대금 상위 1000, 신호일까지 공시된 최근 분기 기준 영업적자이거나 매출·영업이익이 모두 전년 같은 분기보다 줄었으면 제외.
- 매수: 주도 섹터별로 [t0-20, 신호일] 최고가 대비 낙폭이 큰 3종목(최대 9종목) 동일 비중, 신호 다음 거래일 종가 매수. 왕복 비용 0.3%.
- 매도: ① 63거래일 ② 126거래일 ③ 공포·탐욕 75 이상 확인 다음 날(최대 252거래일).
- 비교(같은 날 사고 같은 날 팖): 코스피, 대상 종목 동일가중, 주도 섹터 전체, 하위 3개 섹터의 낙폭 상위 종목, 무작위 9종목 1000번.
한계: 섹터 분류는 현재 산업분류 코드(잘 안 바뀌지만 과거와 다를 수 있음). 수급은 2012~13년 자료가 없어 쓰지 않음.
python3 -m sim.fg_leader_dip → docs/sim/fg_leader_dip.md"""
import os
import numpy as np
import pandas as pd
from . import data as Dm

ROOT = Dm.ROOT if hasattr(Dm, 'ROOT') else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LS = (35, 30, 25, 20); GAP = 20; COST = 0.003; NR = 1000
rng = np.random.default_rng(21)


def fin_ok(fq, codes, d):
    """신호일까지 공시된 최근 분기로 실적 악화 여부. True = 통과(자료 없으면 통과)."""
    f = fq[(fq['avail'] <= d) & fq['stock_code'].isin(codes)].sort_values(['stock_code', 'ord']).drop_duplicates('stock_code', keep='last').set_index('stock_code')
    bad = (f['op'] < 0) | ((f['op'] < f['op_py']) & (f['rev'] < f['rev_py']))
    return pd.Series(True, index=codes).where(~pd.Series(codes, index=codes).map(bad).fillna(False).astype(bool), False)


def main():
    W = Dm.prices(); P = W['adj']; cap = W['cap']; val = W['val']
    sec = Dm.sectors('coarse').reindex(P.columns)
    fg = pd.read_csv(os.path.join(ROOT, 'data', 'fg', 'fg_daily.csv'), index_col=0, parse_dates=True)['코스피'].reindex(P.index).ffill(limit=3)
    kospi = Dm.index_series('idx_kospi', '코스피').reindex(P.index).ffill()
    fq = Dm.fin_quarters()
    dates = P.index; pos = {d: i for i, d in enumerate(dates)}
    val20 = val.rolling(20, min_periods=15).mean()
    capr = cap.rank(axis=1, ascending=False); valr = val20.rank(axis=1, ascending=False)
    uni = (capr <= 700) & (valr <= 1000)
    r = P.pct_change(fill_method=None)
    # 섹터 지수(전일 시총 가중)와 거래대금 비중
    secs = [s for s in sec.dropna().unique() if s != '기타']
    capp = cap.shift(1)
    SI, SV = {}, {}
    for s in secs:
        cols = sec.index[sec == s]
        ri = (capp[cols] * r[cols]).sum(axis=1) / capp[cols].sum(axis=1)
        SI[s] = (1 + ri.fillna(0)).cumprod(); SV[s] = val[cols].sum(axis=1)
    SI = pd.DataFrame(SI); SV = pd.DataFrame(SV)
    share = SV.div(SV.sum(axis=1), axis=0)
    mkt = kospi
    rel126 = SI.div(SI.shift(126)).sub(mkt / mkt.shift(126), axis=0)
    rel63 = SI.div(SI.shift(63)).sub(mkt / mkt.shift(63), axis=0)
    vshare = share.rolling(60, min_periods=40).mean() / share.rolling(250, min_periods=200).mean()
    rows = []; picks_log = []
    for L in LS:
        below = fg <= L
        sig = (~below) & below.shift(1, fill_value=False) & fg.notna()
        last = -10 ** 9
        for ts in dates[sig.values]:
            i = pos[ts]
            if ts < pd.Timestamp('2012-03-01') or i - last < GAP or i + 2 >= len(dates):
                continue
            last = i
            j = i - 1
            while j > 0 and below.iloc[j]:
                j -= 1
            t0 = dates[j]                                        # L 이하로 들어가기 직전 날
            u = uni.loc[ts]
            n_sec = pd.Series({s: int((u & (sec == s)).sum()) for s in secs})
            ok_s = n_sec[n_sec >= 5].index
            sc = (rel126.loc[t0, ok_s].rank(pct=True) + rel63.loc[t0, ok_s].rank(pct=True) + vshare.loc[t0, ok_s].rank(pct=True)) / 3
            sc = sc.dropna().sort_values(ascending=False)
            if len(sc) < 6:
                continue
            top, bot = list(sc.index[:3]), list(sc.index[-3:])
            hi = P.iloc[max(0, j - 20):i + 1].max()
            dd = P.loc[ts] / hi - 1
            cand = u[u].index
            okf = fin_ok(fq, list(cand), ts)
            cand = [c for c in cand if okf.get(c, True)]
            def pick(sectors):
                out = []
                for s in sectors:
                    cs = [c for c in cand if sec.get(c) == s and pd.notna(dd.get(c))]
                    out += list(dd[cs].sort_values().index[:3])
                return out
            pk = pick(top); pb = pick(bot)
            leaders_all = [c for c in cand if sec.get(c) in top]
            e = i + 1                                            # 신호 다음 거래일 종가 매수
            exits = {'63일': min(e + 63, len(dates) - 1), '126일': min(e + 126, len(dates) - 1)}
            k = e
            while k < len(dates) - 1 and k < e + 252 and not (fg.iloc[k] >= 75):
                k += 1
            exits['탐욕75'] = min(k + 1, len(dates) - 1)
            for nm, x in exits.items():
                if x - e < 20 or (nm != '탐욕75' and x - e < (63 if nm == '63일' else 126)):
                    continue                                    # 아직 결과가 다 안 나온 신호는 제외
                if nm == '탐욕75' and k >= len(dates) - 1:
                    continue
                pe, px_ = P.iloc[e], P.iloc[x]
                ret = lambda cs: float(np.nanmean((px_[cs] / pe[cs] - 1).values)) - COST if len(cs) else np.nan
                rp = ret(pk)
                allc = list(cand)
                rnd = np.array([ret(list(rng.choice(allc, len(pk), replace=False))) for _ in range(NR)]) if len(pk) else np.array([np.nan])
                rows.append({'L': L, '매도': nm, '신호일': ts.date(), 't0': t0.date(), '보유일': x - e, '종목수': len(pk), '주도섹터': '·'.join(top),
                             '전략': rp * 100, '코스피': (kospi.iloc[x] / kospi.iloc[e] - 1) * 100,
                             '동일가중': ret(allc) * 100, '주도섹터 전체': ret(leaders_all) * 100,
                             '하위섹터 낙폭주': ret(pb) * 100, '무작위 백분위': float((rnd < rp).mean() * 100)})
            picks_log.append({'L': L, '신호일': ts.date(), '주도섹터': top, '종목': [W['name'].get(c, c) for c in pk]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(ROOT, 'docs', 'sim', 'fg_leader_dip.csv'), index=False, encoding='utf-8-sig')
    md = ['# 공포 바닥 확인 → 직전 주도 섹터 낙폭주 매수', '', __doc__.split('\n\n')[0], '']
    for (L, nm), g in R.groupby(['L', '매도'], sort=False):
        ex_k = g['전략'] - g['코스피']; ex_e = g['전략'] - g['동일가중']; ex_l = g['전략'] - g['주도섹터 전체']; ex_b = g['전략'] - g['하위섹터 낙폭주']
        t = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
        md.append(f"- L{L}·{nm}: 신호 {len(g)}회, 전략 평균 {g['전략'].mean():+.1f}% / 코스피 대비 {ex_k.mean():+.1f}%p(이긴 비율 {(ex_k>0).mean():.0%}, t {t(ex_k):.1f}) / 동일가중 대비 {ex_e.mean():+.1f}%p({(ex_e>0).mean():.0%}) / 주도섹터 전체 대비 {ex_l.mean():+.1f}%p({(ex_l>0).mean():.0%}) / 하위섹터 낙폭주 대비 {ex_b.mean():+.1f}%p({(ex_b>0).mean():.0%}) / 무작위 백분위 평균 {g['무작위 백분위'].mean():.0f}")
    md += ['', '## 신호별(30 기준, 126일 보유)', '', R[(R.L == 30) & (R['매도'] == '126일')].round(1).to_markdown(index=False), '',
           '## 고른 종목(30 기준)', '']
    for p in picks_log:
        if p['L'] == 30:
            md.append(f"- {p['신호일']} [{'·'.join(p['주도섹터'])}] " + ', '.join(p['종목']))
    open(os.path.join(ROOT, 'docs', 'sim', 'fg_leader_dip.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md[:40]))


if __name__ == '__main__':
    main()
