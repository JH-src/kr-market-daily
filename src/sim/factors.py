"""요소 하나하나가 정말 앞으로의 성과를 가르는지 따로 본다(요소 성적표).
매월 말 후보군 안에서 요소 순위와 이후 3·6개월 시장 대비 성과의 순위 상관(정보계수)과
요소 상위 20% 종목의 초과수익을 연도별로 계산해 docs/sim/factors.md 로 쓴다.
python3 -m sim.factors"""
import os
import numpy as np
import pandas as pd
from sim import search as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 이름: (식, 설명). 값이 클수록 좋다고 보는 방향으로 맞춤
F = {
    'ry': (lambda d: d['ry'], '최근 분기 매출 전년비'),
    'accel_d': (lambda d: d['ry'] - d['ryp'], '매출 증가율 가속'),
    'op_g': (lambda d: np.where(d['opp'] > 0, d['op'] / d['opp'] - 1, np.nan), '영업이익 전년비(전년 흑자만)'),
    'opm_chg': (lambda d: d.get('opm_chg'), '영업이익률 개선폭'),
    'roe': (lambda d: d.get('roe'), '자기자본이익률'),
    'ep': (lambda d: 1 / d['per'] if 'per' in d else np.nan, '이익수익률(1/PER)'),
    'bp': (lambda d: 1 / d['pbr'] if 'pbr' in d else np.nan, '1/PBR'),
    'op_yield': (lambda d: 1 / d['por'] if 'por' in d else np.nan, '1/(시총/영업이익)'),
    'mom12_1': (lambda d: d.get('mom12_1'), '12개월 모멘텀(최근 1개월 제외)'),
    'r1': (lambda d: d['r1'], '1개월 수익률'),
    'rs3': (lambda d: d['r3'] - d['sec_m3'], '섹터 대비 3개월'),
    'r1rel': (lambda d: d['r1'] - d['sec_m1'], '섹터 대비 1개월'),
    'hi52': (lambda d: d.get('hi52'), '52주 고점 근접'),
    'low_vol': (lambda d: -d['vol60'] if 'vol60' in d else np.nan, '낮은 변동성'),
    'val_surge': (lambda d: d.get('val_surge'), '거래대금 급증'),
    'small': (lambda d: -np.log(d['cap']) if 'cap' in d else np.nan, '작은 시가총액'),
    'frgn20_cap': (lambda d: d.get('frgn20_cap'), '외국인 20일 순매수/시총'),
    'frgn60_cap': (lambda d: d.get('frgn60_cap'), '외국인 60일 순매수/시총'),
    'orgn20_cap': (lambda d: d.get('orgn20_cap'), '기관 20일 순매수/시총'),
    'fo_minus_p': (lambda d: (d['frgn20_cap'] + d['orgn20_cap'] - d['prsn20_cap']) if 'prsn20_cap' in d else ((d['frgn20_cap'] + d['orgn20_cap']) if 'orgn20_cap' in d else np.nan), '외국인+기관 매수, 개인 매도 강도'),
    'frgn_streak': (lambda d: d.get('frgn_streak'), '외국인 연속 순매수 일수'),
    'major_foreign90': (lambda d: d.get('major_foreign90'), '90일 내 외국계 5% 공시 건수'),
    'sec_m3': (lambda d: d['sec_m3'], '소속 섹터 3개월 강도'),
    'buyback90': (lambda d: d.get('buyback90'), '90일 내 자사주 취득 결정'),
    'supply90': (lambda d: d.get('supply90'), '90일 내 단일판매·공급계약 공시'),
    'exp_g3': (lambda d: d.get('exp_g3'), '소속 섹터 관련 수출 3개월 전년비'),
    'pension60_cap': (lambda d: d.get('pension60_cap'), '연기금 60일 순매수/시총'),
    'fo60_cap': (lambda d: d.get('fo60_cap'), '외국인+기관 60일 순매수/시총'),
    'frgn_hold_chg60': (lambda d: d.get('frgn_hold_chg60'), '외국인 보유비율 60일 변화(%p)'),
    'rep_cap': (lambda d: -d['sec_cap_rank'] if 'sec_cap_rank' in d else np.nan, '섹터 안 시총 순위(대표성)'),
    'rep_val': (lambda d: -d['sec_val_rank'] if 'sec_val_rank' in d else np.nan, '섹터 안 거래대금 순위(대표성)'),
}


def ic_table(d, col, h):
    rows = []
    for t, g in d.groupby('date'):
        x = g[col]; y = g[f'ex{h}']
        ok = x.notna() & y.notna()
        if ok.sum() < 30 or x[ok].nunique() < 5:
            continue
        ic = x[ok].rank().corr(y[ok].rank())
        q = x[ok] >= x[ok].quantile(0.8)
        top = y[ok][q].clip(-0.6, 1.5).mean() - y[ok].clip(-0.6, 1.5).mean()
        rows.append((t, ic, top * 100, ok.mean()))
    return pd.DataFrame(rows, columns=['date', 'ic', 'top', 'cov']).set_index('date')


def main():
    d = S.panel('theme').copy()
    d = d[(d['date'] >= S.D0) & (d['date'] <= S.D1)]
    if 'prsn20' in d and 'prsn20_cap' not in d:
        d['prsn20_cap'] = d['prsn20'] / d['cap'] * 100
    for k, (f, _) in F.items():
        try:
            v = f(d)
            d['_' + k] = pd.to_numeric(pd.Series(np.asarray(v, dtype=float) if v is not None else np.nan, index=d.index), errors='coerce')
        except Exception:
            d['_' + k] = np.nan
    lead = d['stage'].isin(['주도', '초기'])
    out, yr = [], {}
    for k, (_, desc) in F.items():
        for scope, dd in (('전체', d), ('주도·초기 섹터', d[lead])):
            r = {}
            for h in (63, 126):
                t = ic_table(dd, '_' + k, h)
                if t.empty:
                    continue
                ann = t['ic'].groupby(t.index.year).mean()
                # 6개월 보유는 달끼리 겹치므로 t값은 연도 평균으로 계산(보수적)
                r[f'ic{h}'] = t['ic'].mean()
                r[f't{h}'] = ann.mean() / ann.std() * np.sqrt(len(ann)) if len(ann) > 2 and ann.std() > 0 else np.nan
                r[f'top{h}'] = t['top'].mean()
                r[f'yrs{h}'] = f"{(ann > 0).sum()}/{len(ann)}"
                r['cov'] = t['cov'].mean() * 100
                if h == 126 and scope == '전체':
                    yr[k] = ann
            if r:
                out.append({'요소': k, '설명': desc, '범위': scope, **r})
    o = pd.DataFrame(out)
    o = o.sort_values(['범위', 'ic126'], ascending=[True, False])
    y = pd.DataFrame(yr).T.round(3)
    p = os.path.join(ROOT, 'docs', 'sim', 'factors.md')
    with open(p, 'w', encoding='utf-8') as f:
        f.write("# 요소 성적표\n\n2011~2023 매월 말 후보군(시총 상위 1200·20일 평균 거래대금 상위 1500) 안에서 요소 순위와 이후 성과(시총 상위 700 동일가중 대비, 비용 0.5% 차감)의 순위 상관.\n"
                "ic: 월평균 정보계수 / t: 연도 평균 정보계수의 t값(겹침 보정용 보수적 계산) / top: 요소 상위 20% 종목의 후보군 평균 대비 초과(%p) / yrs: 정보계수가 플러스인 해 / cov: 값이 있는 종목 비율(%).\n"
                "수급 요소는 한투 과거 수급을 받은 종목만 있어 cov가 낮으면 해석 주의.\n\n")
        for sc in ('전체', '주도·초기 섹터'):
            f.write(f"## {sc}\n\n" + o[o['범위'] == sc].drop(columns=['범위']).round(3).to_markdown(index=False) + "\n\n")
        f.write("## 연도별 6개월 정보계수(전체)\n\n" + y.to_markdown() + "\n")
    print(o.round(3).to_string())


if __name__ == '__main__':
    main()
