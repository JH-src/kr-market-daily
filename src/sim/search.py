"""로직 자동 탐색(토큰 사용 없음). 패널 위에서 로직 조합을 무작위로 만들어 평가하고,
'그해 이전 자료로 고른 최고 로직을 그해에 적용'하는 방식(걷기식 검증)으로 정직한 성과를 같이 낸다.
python3 -m sim.search <초 예산> [시드]
결과: SIM_CACHE/search_results.csv (누적), docs/sim/search.md (요약)
"""
import json, os, pickle, sys, time
import numpy as np
import pandas as pd

WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COST = 0.005
D0, D1 = pd.Timestamp('2011-01-01'), pd.Timestamp('2023-12-31')   # 자료가 있는 가장 이른 시점부터(실적 요소는 2016년 이후에만 존재)

SPACE = {
    'level': ['coarse', 'ksic3', 'theme'],
    'stages': [['주도', '초기'], ['주도'], ['초기'], ['주도', '초기', '중립'], 'all'],
    'types': [[], ['A'], ['B'], ['A', 'B']],
    'b_lag': [-10, -15, -25], 'b_ma60': [-10, -5, 0],
    'rsi_hot': [None, 70, 75, 80], 'rights': [0, 90, 180], 'loss_gate': [True, False],
    'earn': ['off', 'strict', 'lenient'], 'rev': [0, 5, 10, 15, 20], 'accel': [False, True],
    'cap': [300, 500, 700, 1000], 'rank': ['rs3', 'r1rel', 'ry', 'accel_d', 'mix', 'low_rsi', 'low_per', 'low_por', 'value_mom', 'mom12_1', 'hi52', 'opm_chg', 'roe', 'frgn20_cap', 'fo20_cap'],
    'n': ['5/6/7'],
    'value': [None, 'per<15', 'per<25', 'pbr<1.5', 'por<15', 'per_sec<0.5', 'pbr_sec<0.5', 'roe>10', 'debt<200', 'opm_up'],
    'trend': [None, 'mom12>0', 'hi52>-20', 'vol_low', 'val_surge'],
    'flow': [None, 'frgn20>0', 'fo20>0', 'fo_buy_prsn_sell', 'frgn_streak5', 'frgn60>0', 'major_foreign'],
}
BASE = {'level': 'coarse', 'stages': ['주도', '초기'], 'types': ['A', 'B'], 'b_lag': -15, 'b_ma60': -5, 'rsi_hot': 75,
        'rights': 90, 'loss_gate': True, 'earn': 'strict', 'rev': 10, 'accel': False, 'cap': 700, 'rank': 'rs3', 'n': '5/6/7',
        'value': None, 'trend': None, 'flow': None}
NMAP = {'5/6/7': {'위험회피': 5, '중립': 6, '위험선호': 7}}

_P = {}


def panel(level):
    if level not in _P:
        px = os.path.join(WORK, f'panel_{level}_x.pkl')
        d = pickle.load(open(px if os.path.exists(px) else os.path.join(WORK, f'panel_{level}.pkl'), 'rb'))
        d = d[(d['date'] >= D0 - pd.Timedelta(days=1)) & d['ret126'].notna()].copy()
        d['rs3'] = d['r3'] - d['sec_m3']; d['r1rel'] = d['r1'] - d['sec_m1']
        d['accel_d'] = d['ry'] - d['ryp']; d['low_rsi'] = -d['rsi']
        for h in (63, 126):
            d[f'ex{h}'] = d[f'ret{h}'] - COST - d[f'mkt{h}']
            d[f'idio{h}'] = d[f'ret{h}'] - COST - d['beta'].fillna(1) * d[f'mkt{h}']
            d[f'stk{h}'] = d[f'ret{h}'] - d[f'sec{h}']
            d[f'secp{h}'] = d[f'sec{h}'] - d[f'mkt{h}']
        _P[level] = d
    return _P[level]


def filt(cfg):
    d = panel(cfg['level'])
    m = d['caprank'] <= cfg['cap']
    if cfg['stages'] != 'all':
        m &= d['stage'].isin(cfg['stages'])
    if cfg['types']:
        A = (d['stage'] == '초기') & (d['r1'] > d['sec_m1']) & (d['r3'] > d['sec_m3'])
        B = (d['stage'] == '주도') & (d['r1'] < d['sec_m1']) & (d['r3'] - d['sec_m3'] > cfg['b_lag']) & (d['g60'] > cfg['b_ma60'])
        t = pd.Series(False, index=d.index)
        if 'A' in cfg['types']:
            t |= A
        if 'B' in cfg['types']:
            t |= B
        m &= t
    if cfg['rsi_hot']:
        m &= ~(d['rsi'] >= cfg['rsi_hot'])
    if cfg['rights']:
        m &= d['rights_days'] > cfg['rights']
    if cfg['loss_gate']:
        m &= ~((d['op'] < 0) & (d['ry'] < 0))
    if cfg['earn'] == 'strict':
        m &= (d['ry'] >= cfg['rev']) & (d['ryp'] > 0) & (d['op'] > 0) & ((d['opp'] <= 0) | (d['op'] >= d['opp']))
    elif cfg['earn'] == 'lenient':
        m &= (d['ry'] >= cfg['rev']) & (d['op'] > 0)
    if cfg['accel']:
        m &= d['ry'] > d['ryp']
    v = cfg.get('value')
    if v and 'per' in d:
        m &= {'per<15': d['per'] < 15, 'per<25': d['per'] < 25, 'pbr<1.5': d['pbr'] < 1.5, 'por<15': d['por'] < 15,
              'per_sec<0.5': d['per_secpct'] < 0.5, 'pbr_sec<0.5': d['pbr_secpct'] < 0.5, 'roe>10': d['roe'] > 10,
              'debt<200': ~(d['debt_ratio'] > 200), 'opm_up': d['opm_chg'] > 0}[v]
    tr = cfg.get('trend')
    if tr and 'mom12_1' in d:
        m &= {'mom12>0': d['mom12_1'] > 0, 'hi52>-20': d['hi52'] > -20,
              'vol_low': d['vol60'] < d.groupby('date')['vol60'].transform('median'), 'val_surge': d['val_surge'] > 1.3}[tr]
    fl = cfg.get('flow')
    if fl and 'frgn20_cap' in d:
        m &= {'frgn20>0': d['frgn20'] > 0, 'fo20>0': (d['frgn20'] > 0) & (d['orgn20'] > 0),
              'fo_buy_prsn_sell': (d['frgn20'] > 0) & (d['orgn20'] > 0) & (d['prsn20'] < 0),
              'frgn_streak5': d['frgn_streak'] >= 5, 'frgn60>0': d['frgn60'] > 0,
              'major_foreign': d.get('major_foreign90', pd.Series(0, index=d.index)) > 0}[fl]
    return d[m]


def picks(cfg, c=None):
    c = filt(cfg) if c is None else c
    if c.empty:
        return c, c
    key = cfg['rank']
    if key == 'mix':
        c = c.assign(_k=c.groupby('date')['rs3'].rank(pct=True) + c.groupby('date')['ry'].rank(pct=True))
    elif key in ('low_per', 'low_por'):
        c = c.assign(_k=-c[key[4:]].fillna(1e9))
    elif key == 'value_mom':
        c = c.assign(_k=(1 - c['per_secpct'].fillna(1)) + c.groupby('date')['rs3'].rank(pct=True))
    elif key == 'fo20_cap':
        c = c.assign(_k=(c.get('frgn20_cap', 0) + c.get('orgn20_cap', 0)))
    elif key not in c:
        c = c.assign(_k=0.0)
    else:
        c = c.assign(_k=c[key])
    c = c.sort_values(['date', '_k'], ascending=[True, False])
    nm = NMAP[cfg['n']]
    c['_n'] = c['regime'].map(nm)
    c['_r'] = c.groupby('date').cumcount()
    return c[c['_r'] < c['_n']], c


def monthly(p):
    """로직의 달별 성과(걷기식 검증용 캐시)."""
    g = p.groupby('date')
    return pd.DataFrame({'m63': g['ex63'].apply(lambda x: x.clip(-0.6, 1.5).mean()) * 100,
                         'm126': g['ex126'].apply(lambda x: x.clip(-0.6, 1.5).mean()) * 100,
                         'hit126': g['ex126'].apply(lambda x: (x > 0).mean()) * 100, 'n': g.size()})


def stats(p, c, a=D0, b=D1):
    p = p[(p['date'] >= a) & (p['date'] <= b)]
    c = c[(c['date'] >= a) & (c['date'] <= b)]
    out = {'n': len(p), 'months': p['date'].nunique()}
    if len(p) < 30:
        return out
    for h in (63, 126):
        ex = p[f'ex{h}'].clip(-0.6, 1.5)
        mm = ex.groupby(p['date']).mean()
        out[f'm{h}'] = mm.mean() * 100
        out[f'trim{h}'] = p.drop(p[f'ret{h}'].nlargest(5).index)[f'ex{h}'].groupby(p['date']).mean().mean() * 100
        out[f'med{h}'] = p[f'ex{h}'].median() * 100
        out[f'hit{h}'] = (p[f'ex{h}'] > 0).mean() * 100
        out[f'idio{h}'] = p[f'idio{h}'].clip(-0.6, 1.5).groupby(p['date']).mean().mean() * 100
        out[f'stk{h}'] = p[f'stk{h}'].clip(-0.6, 1.5).groupby(p['date']).mean().mean() * 100
        out[f'secp{h}'] = p[f'secp{h}'].groupby(p['date']).mean().mean() * 100
        h1 = mm[mm.index < '2018-01-01'].mean() * 100; h2 = mm[mm.index >= '2018-01-01'].mean() * 100
        out[f'h1_{h}'], out[f'h2_{h}'] = h1, h2
        # 정보계수: 같은 달 후보들 안에서 순위 점수와 시장 대비 성과의 순위 상관
        ic = c.groupby('date').apply(lambda g: g['_k'].rank().corr(g[f'ex{h}'].rank()) if len(g) >= 8 else np.nan).dropna()
        out[f'ic{h}'] = ic.mean() if len(ic) else np.nan
        out[f'ic_t{h}'] = ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 5 and ic.std() > 0 else np.nan
    # 목적함수: 3·6개월 평균(이상치 완화)에 이긴 비율·두 구간 일관성 반영
    cons = min(out['h1_63'], out['h2_63'], out['h1_126'], out['h2_126'])
    out['score'] = (out['trim63'] + out['trim126']) / 2 + 0.1 * ((out['hit63'] + out['hit126']) / 2 - 50) + 0.5 * min(cons, 0)
    return out


def available_levels():
    return [l for l in SPACE['level'] if os.path.exists(os.path.join(WORK, f'panel_{l}_x.pkl')) or os.path.exists(os.path.join(WORK, f'panel_{l}.pkl'))]


def sample(rng):
    cfg = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    lv = available_levels()
    if cfg['level'] not in lv:
        cfg['level'] = lv[rng.integers(len(lv))]
    # 절반은 기준 로직 근처(한두 개만 변경)
    if rng.random() < 0.5:
        cfg = dict(BASE)
        for k in rng.choice(list(SPACE), size=rng.integers(1, 3), replace=False):
            cfg[k] = SPACE[k][rng.integers(len(SPACE[k]))]
        if cfg['level'] not in lv:
            cfg['level'] = lv[0]
    return cfg


def walk_forward(series, K=1, cov=0.5):
    """연도마다 그 이전(6개월 보유 겹침 제외) 달들의 성과로 로직을 골라, 처음 보는 그해에 적용(행렬 계산).
    고르는 기준: 종목이 없던 달은 0(시장과 같음)으로 채운 3·6개월 평균. 학습 기간 중 절반 이상 달에 종목이 있어야 함.
    K>1이면 상위 K개 로직의 평균(앙상블)."""
    if not series:
        return pd.DataFrame()
    keys = list(series)
    months = sorted(set().union(*[set(series[k].index) for k in keys]))
    pos = {m: i for i, m in enumerate(months)}
    A = np.full((len(keys), len(months), 3), np.nan, dtype=np.float32)
    for j, k in enumerate(keys):
        m = series[k]; idx = [pos[x] for x in m.index]
        A[j, idx, 0] = m['m63'].values; A[j, idx, 1] = m['m126'].values; A[j, idx, 2] = m['hit126'].values
    M = pd.DatetimeIndex(months)
    out = []
    import warnings
    warnings.filterwarnings('ignore')
    for y in range(2013, 2024):
        tr = (M >= D0) & (M <= pd.Timestamp(f'{y - 1}-06-30'))
        te = (M >= pd.Timestamp(f'{y}-01-01')) & (M <= pd.Timestamp(f'{y}-12-31'))
        if tr.sum() == 0:
            continue
        cnt = np.sum(~np.isnan(A[:, tr, 1]), axis=1)
        sc = (np.nan_to_num(A[:, tr, 0]).mean(1) + np.nan_to_num(A[:, tr, 1]).mean(1)) / 2
        sc[(cnt < 18) | (cnt < cov * tr.sum())] = -np.inf
        top = np.argsort(-sc)[:K]
        top = top[np.isfinite(sc[top])]
        if not len(top):
            continue
        t0 = np.nan_to_num(np.nanmean(A[top][:, te, 0], 0)); t1 = np.nan_to_num(np.nanmean(A[top][:, te, 1], 0))
        hh = np.nanmean(A[top][:, te, 2], 0)
        out.append({'year': y, 'chosen': keys[top[0]] if K == 1 else f'상위 {len(top)}개 평균', 'train_score': float(sc[top[0]]),
                    'months': int(np.sum(~np.isnan(A[top[0], te, 1]))),
                    'oos_ex63': float(np.mean(t0)), 'oos_ex126': float(np.mean(t1)), 'oos_hit126': float(np.nanmean(hh))})
    return pd.DataFrame(out)


STATE = os.environ.get('SIM_STATE', WORK)


def save_csv(df, path):
    # 저장 중 강제 종료돼도 기존 파일이 깨지지 않게: 임시 파일에 쓴 뒤 바꿔치기
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False, compression='gzip' if path.endswith('.gz') else None)
    os.replace(tmp, path)


def save_series(series, path):
    tmp = path[:-4] + '.tmp.npz'
    _save_series(series, tmp)
    os.replace(tmp, path)


def _save_series(series, path):
    keys = list(series)
    months = sorted(set().union(*[set(series[k].index) for k in keys])) if keys else []
    arr = np.full((len(keys), len(months), 3), np.nan, dtype=np.float32)
    pos = {m: i for i, m in enumerate(months)}
    for j, k in enumerate(keys):
        m = series[k]
        idx = [pos[x] for x in m.index]
        arr[j, idx, 0] = m['m63'].values; arr[j, idx, 1] = m['m126'].values; arr[j, idx, 2] = m['hit126'].values
    np.savez_compressed(path, keys=np.array(keys, dtype=object), months=np.array([str(x)[:10] for x in months]), arr=arr)


def load_series(path):
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=True)
    months = pd.to_datetime(z['months'])
    out = {}
    for j, k in enumerate(z['keys']):
        a = z['arr'][j]
        ok = ~np.isnan(a[:, 1])
        out[str(k)] = pd.DataFrame({'m63': a[ok, 0], 'm126': a[ok, 1], 'hit126': a[ok, 2]}, index=months[ok])
    return out


def main(budget, seed):
    rng = np.random.default_rng(seed)
    os.makedirs(STATE, exist_ok=True)
    rp = os.path.join(STATE, 'search_results_v2.csv.gz')
    sp = os.path.join(STATE, 'search_series_v2.npz')
    old_rp, old_sp = os.path.join(WORK, 'search_results.csv'), os.path.join(WORK, 'search_series.pkl')
    done = pd.read_csv(rp) if os.path.exists(rp) else pd.DataFrame()
    series = load_series(sp)
    seen = set(done['cfg']) if len(done) else set()
    t0 = time.time(); new = []
    cands = [dict(BASE)] + [sample(rng) for _ in range(100000)]
    for cfg in cands:
        if time.time() - t0 > budget:
            break
        key = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        p, c = picks(cfg)
        s = stats(p, c)
        s['cfg'] = key
        new.append(s)
        if len(p) >= 30:
            series[key] = monthly(p)
        if len(new) % 50 == 0:
            save_csv(pd.concat([done, pd.DataFrame(new)]), rp)
            print(len(new), round(time.time() - t0), flush=True)
        if len(new) % 500 == 0:
            save_series(series, sp)
    allr = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
    save_csv(allr, rp); save_series(series, sp)
    ok = allr.dropna(subset=['score']) if 'score' in allr else allr.iloc[0:0]
    wf = walk_forward(series)
    wfe = walk_forward(series, K=20)
    write_md(allr, ok, wf, wfe)
    print('done', len(new), len(allr))


def write_md(allr, ok, wf, wfe=None):
    cols = ['score', 'n', 'm63', 'm126', 'trim63', 'trim126', 'med126', 'hit63', 'hit126', 'idio126', 'stk126', 'secp126', 'h1_126', 'h2_126', 'ic126', 'ic_t126']
    top = ok.sort_values('score', ascending=False).head(15)
    if ok.empty:
        return
    base = ok[ok['cfg'] == json.dumps(BASE, ensure_ascii=False, sort_keys=True)]
    md = ['# 로직 자동 탐색 요약', '', f'평가한 로직 수: {len(allr)} (설계 구간 2011~2023, 실적 요소는 2016년 이후만 존재, 매월 말 5~7종목, 비용 0.5% 차감, 2024년 이후 미사용)', '',
          '열: m=시장 평균 대비 월평균(%p), trim=상위 5건 제외, med=종목 중앙값, hit=시장 평균 이긴 비율(%), idio=베타 반영 종목 고유 몫, stk=섹터 대비 종목 몫, secp=섹터가 시장보다 더 간 몫, h1/h2=2011~17 / 2018~23, ic=로직 순위와 성과의 순위 상관(월평균), ic_t=그 t값', '',
          '## 걷기식 검증(그해 이전 자료로 고른 로직을 그해에 적용 = 정직한 성과)', '', wf.round(2).to_markdown(index=False) if len(wf) else '(없음)', '',
          f"걷기식 평균: 3개월 {wf['oos_ex63'].mean():+.2f}%p, 6개월 {wf['oos_ex126'].mean():+.2f}%p, 6개월 이긴 비율 {wf['oos_hit126'].mean():.1f}%, 6개월 플러스인 해 {(wf['oos_ex126'] > 0).sum()}/{len(wf)}" if len(wf) else '', '',
          (f"상위 20개 로직 평균(앙상블) 걷기식: 3개월 {wfe['oos_ex63'].mean():+.2f}%p, 6개월 {wfe['oos_ex126'].mean():+.2f}%p, 6개월 플러스인 해 {(wfe['oos_ex126'] > 0).sum()}/{len(wfe)} · 연도별 6개월 " + ' '.join(f"{int(r.year) % 100}:{r.oos_ex126:+.1f}" for r in wfe.itertuples())) if wfe is not None and len(wfe) else '', '',
          '종목이 없던 달은 0(시장과 같음)으로 계산. 로직을 많이 시험할수록 학습 구간 1등은 운으로 뽑힐 가능성이 커져, 걷기식 결과가 학습 성적보다 크게 낮은 것이 정상.', '',
          '## 기준 로직', '', base[cols + ['cfg']].round(2).to_markdown(index=False) if len(base) else '', '',
          '## 점수 상위 15', '', top[cols + ['cfg']].round(2).to_markdown(index=False)]
    open(os.path.join(ROOT, 'docs', 'sim', 'search.md'), 'w', encoding='utf-8').write('\n'.join(md))


if __name__ == '__main__':
    main(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1)
