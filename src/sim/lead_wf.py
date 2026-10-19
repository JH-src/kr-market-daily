"""주도 섹터 판정 — 완전 걷기식 검증.
모든 선택(어떤 방법을 쓸지, 몇 개를 섞을지, 상위 몇 개 섹터를 고를지)을 그해 1월 1일 이전에 '결과까지 확인된' 자료로만 한다.
- 결과 확인 시점 차단: 월말 d의 h일 뒤 성과는 d+h 거래일에야 알 수 있으므로, 그날이 그해 1월 1일 이전인 월만 과거 자료로 씀.
- 1단: 설정(N개 섞기, 상위 K개 섹터)마다, 해마다 과거 연도별 성과로 단일 방법의 t값을 내 플러스 해 70% 이상인 것 중 상위 N개를 골라
       순위 평균 점수로 그해 매월 상위 K 섹터를 뽑음 → 설정별 걷기식 성과 시계열.
- 2단: 해마다 과거 연도의 1단 걷기식 성과(이것도 결과 확인 시점 차단)로 설정을 고름 → 최종 성과.
python3 -m sim.lead_wf [stock|etf] → docs/sim/lead_wf_<mode>.md, docs/sim/lead_current_<mode>.json"""
import json
import sys
import numpy as np
import pandas as pd
from sim import etf_lead as E

NS = (1, 3, 5, 10)
KS = (3, 5, 8)
HZ = {'3개월': 63, '6개월': 126}
START = 2014


def prep(mode):
    if mode == 'stock':
        L, TV, FL, AUM, BR, HI = E.build_stock()
        F, ok = E.features(L, TV, FL, AUM, BR, HI)
    else:
        L, TV, FL, AUM = E.build()
        F, ok = E.features(L, TV, FL, AUM)
    return L, F


def tstat(Y):
    c = Y.count()
    return Y.mean() / (Y.std(ddof=1) / np.sqrt(c)), (Y > 0).sum() / c, c


def run(L, F, hz):
    me = E.month_ends(L.index)
    pos = {d: i for i, d in enumerate(L.index)}
    fwd = L.shift(-hz) / L - 1
    me = [d for d in me if fwd.loc[d].notna().sum() >= 8]
    known = pd.Series({d: L.index[pos[d] + hz] for d in me})  # 결과가 확인되는 날
    ex = fwd.loc[me]; ex = ex.sub(ex.mean(axis=1), axis=0) * 100
    rk = {k: v.loc[me].rank(axis=1, pct=True).where(ex.notna()) for k, v in F.items()}

    def topk(score, K):
        t = (score.rank(axis=1, ascending=False, method='first') <= K) & score.notna()
        n = t.sum(axis=1)
        return (ex.where(t).sum(axis=1) / n).where(n >= min(3, K)), t

    single = {K: pd.DataFrame({k: topk(v, K)[0] for k, v in rk.items()}) for K in KS}
    years = sorted({d.year for d in me})
    cfg_series, cfg_members = {}, {}
    for N in NS:
        for K in KS:
            S = single[K]; out = pd.Series(dtype=float); mem = {}
            for y in years:
                if y < START:
                    continue
                cut = pd.Timestamp(f'{y}-01-01')
                past = S[known < cut]
                Y = past.groupby(past.index.year).mean()
                Y = Y.loc[:, Y.count() >= 3]
                if Y.empty:
                    continue
                t, p, c = tstat(Y)
                cand = t[p >= 0.7].sort_values(ascending=False)
                if len(cand) < N:
                    cand = t.sort_values(ascending=False)
                ch = list(cand.index[:N]); mem[y] = ch
                cur = [d for d in me if d.year == y]
                sc = sum(rk[k].loc[cur] for k in ch)
                out = pd.concat([out, topk(sc, K)[0].reindex(cur)])
            cfg_series[(N, K)] = out; cfg_members[(N, K)] = mem
    C = pd.DataFrame(cfg_series)
    final, picks = pd.Series(dtype=float), {}
    for y in years:
        if y < START + 3:
            continue
        cut = pd.Timestamp(f'{y}-01-01')
        past = C[known.reindex(C.index) < cut]
        Y = past.groupby(past.index.year).mean()
        Y = Y.loc[:, Y.count() >= 3]
        if Y.empty:
            continue
        t, p, c = tstat(Y)
        best = t.sort_values(ascending=False).index[0]
        picks[y] = best
        cur = C.index[C.index.year == y]
        final = pd.concat([final, C.loc[cur, best]])
    return C, cfg_members, final, picks, rk, ex


def yr(s):
    return s.groupby(s.index.year).mean()


def main(mode='stock'):
    L, F = prep(mode)
    md = [f'# 주도 섹터 판정 — 완전 걷기식 검증({"종목 기준 섹터" if mode == "stock" else "세분화 ETF 섹터"} {L.shape[1]}개)', '',
          '방법 선택·섞는 개수(N)·고르는 섹터 수(K)를 모두 그해 이전에 결과까지 확인된 자료로만 정함. 성과는 고른 섹터의 이후 수익률 - 같은 날 전체 섹터 평균(%p).',
          f'후보: 단일 방법 {len(F)}개, N {NS}, K {KS}. 1단(방법 고르기)은 {START}년부터, 2단(설정 고르기)은 {START + 3}년부터.', '']
    cur_out = {}
    for hzn, hz in HZ.items():
        C, mem, final, picks, rk, ex = run(L, F, hz)
        Yf = yr(final)
        cfg = pd.DataFrame({f'N{n}·K{k}': yr(C[(n, k)]) for (n, k) in C.columns})
        summ = pd.DataFrame({'평균': cfg.mean(), '플러스 해': (cfg > 0).sum().astype(str) + '/' + cfg.count().astype(str),
                             '2016~23': cfg.loc[2016:2023].mean(), '2024~': cfg.loc[2024:].mean(), '최악': cfg.min()}).sort_values('평균', ascending=False)
        md += [f'## {hzn} 뒤', '',
               f'**최종(2단까지 걷기식)**: 평균 {Yf.mean():+.2f}%p, 플러스 해 {(Yf > 0).sum()}/{Yf.count()}, 2016~23 {Yf.loc[2016:2023].mean():+.2f}%p, 2024~ {Yf.loc[2024:].mean():+.2f}%p, 최악 {Yf.min():+.2f}%p', '',
               pd.DataFrame({'설정': {y: f'N{n}·K{k}' for y, (n, k) in picks.items()}, '성과': Yf.round(2)}).to_markdown(), '',
               '### 설정별(1단 걷기식 결과, 참고)', '', summ.round(2).to_markdown(), '',
               '### 연도별 고른 방법(N5·K5 예시)', '']
        for y, ch in mem.get((5, 5), {}).items():
            md.append(f'- {y}: ' + ', '.join(ch))
        md.append('')
        # 현재 판정: 마지막 해 설정과 방법을 오늘 자료에 적용
        ly = max(picks); n, k = picks[ly]
        ch = mem[(n, k)][ly]
        last = L.index.max()
        sc = sum(F[c].loc[last].rank(pct=True) for c in ch).dropna().sort_values(ascending=False)
        cur_out[hzn] = {'asof': f'{last:%Y-%m-%d}', 'N': int(n), 'K': int(k), 'methods': ch, 'top': list(sc.index[:k]),
                        'score': sc.round(3).to_dict()}
        md += [f'### 현재 판정({last:%Y-%m-%d}, {ly}년 설정 N{n}·K{k})', '', '방법: ' + ', '.join(ch), '',
               sc.head(15).round(2).to_frame('점수').to_markdown(), '']
    open(f'../docs/sim/lead_wf_{mode}.md', 'w').write('\n'.join(md))
    json.dump(cur_out, open(f'../docs/sim/lead_current_{mode}.json', 'w'), ensure_ascii=False, indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'stock')
