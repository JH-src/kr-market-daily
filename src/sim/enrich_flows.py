"""패널에 수급(외국인·기관·개인 순매수)과 5% 지분공시(외국계 기관·국민연금) 지표를 덧붙인다.
수급은 한투 과거 자료(archive/kis_flows, 현재 상장 종목만 → 상장폐지 종목은 비어 있음, 생존 편향 주의).
python3 -m sim.enrich_flows
"""
import glob, os, pickle, re
import numpy as np
import pandas as pd

WORK = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOREIGN = r'골드만|Goldman|모건|Morgan|JP ?Morgan|제이피모간|블랙록|BlackRock|Black ?Rock|뱅가드|Vanguard|노르웨이|Norges|싱가포르|GIC|템플턴|Templeton|캐피탈 ?그룹|Capital ?(Group|Research|International)|피델리티|Fidelity|웰링턴|Wellington|UBS|씨티|Citi|크레디트|Credit ?Suisse|맥쿼리|Macquarie|라자드|Lazard|슈로더|Schroder|애버딘|Aberdeen|매튜스|Matthews|디멘셔널|Dimensional|스테이트 ?스트리트|State Street|인베스코|Invesco|노무라|Nomura|HSBC|알리안츠|Allianz|도이치|Deutsche|BNP|소시에테|Societe|메릴|Merrill|바클레이|Barclays|아부다비|Abu ?Dhabi|ADIA|카타르|Qatar|Wasatch|와사치|First ?State|퍼스트스테이트|Oppenheimer|오펜하이머|T\. ?Rowe|티로우|프랭클린|Franklin|Artisan|Baillie|베일리'
NPS = r'국민연금'


def hold_now():
    """전 종목 오늘 외국인 보유비율·상장주식수(data/kis/frgn_hold/<날짜>.csv 중 최신)."""
    fs = sorted(glob.glob(os.path.join(ROOT, 'data', 'kis', 'frgn_hold', '*.csv')))
    if not fs:
        return pd.DataFrame()
    h = pd.read_csv(fs[-1], dtype={'code': str, 'date': str})
    for c in ('frgn_ehrt', 'lstn_stcn'):
        h[c] = pd.to_numeric(h[c], errors='coerce')
    return h.dropna(subset=['frgn_ehrt', 'lstn_stcn']).drop_duplicates('code').set_index('code')


def daily_flows():
    """과거 파일 이후의 일별 수급(data/kis/flows/<연도>.csv) — 보유비율 역산에 오늘까지 필요."""
    fs = glob.glob(os.path.join(ROOT, 'data', 'kis', 'flows', '*.csv'))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f, dtype={'date': str, 'code': str}) for f in fs], ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    return d


def flows_table():
    out = []
    H = hold_now(); DF = daily_flows()
    shr = None
    try:
        from . import data as Dm
        shr = Dm.prices()['shr']
    except Exception:
        pass
    for f in glob.glob(os.path.join(ROOT, 'archive', 'kis_flows', '*.csv.gz')):
        code = os.path.basename(f)[:6]
        d = pd.read_csv(f, dtype={'date': str})
        if d.empty:
            continue
        d['date'] = pd.to_datetime(d['date'])
        if len(DF):
            x = DF[(DF['code'] == code) & (DF['date'] > d['date'].max())]
            if len(x):
                d = pd.concat([d, x[d.columns.intersection(x.columns)]], ignore_index=True)
        d = d.sort_values('date').drop_duplicates('date', keep='last')
        # 외국인 보유비율 역산: 오늘 보유주식수에서 그 뒤 날짜의 외국인 순매수 주식수(금액÷종가)를 빼 나감. 주식수가 20% 넘게 달라진 구간은 비움(분할·증자).
        d['frgn_hold'] = np.nan
        if len(H) and code in H.index and d['date'].max() >= pd.Timestamp(str(H.loc[code, 'date'])) - pd.Timedelta(days=7):
            q = (pd.to_numeric(d['frgn'], errors='coerce') * 1e6 / pd.to_numeric(d['close'], errors='coerce').replace(0, np.nan)).fillna(0)
            s_now = H.loc[code, 'lstn_stcn']
            hold = H.loc[code, 'frgn_ehrt'] / 100 * s_now - (q[::-1].cumsum()[::-1] - q)
            ratio = hold / s_now * 100
            if shr is not None and code in shr.columns:
                st = shr[code].reindex(d['date']).ffill().values
                ratio = ratio.where(np.abs(st / s_now - 1) <= 0.2)
            d['frgn_hold'] = ratio.clip(0, 100)
        d['frgn_hold_chg60'] = d['frgn_hold'] - d['frgn_hold'].shift(60)
        for c in ('frgn', 'orgn', 'prsn', 'pension'):
            d[c] = pd.to_numeric(d[c], errors='coerce') * 1e6     # 백만원 -> 원
            d[c + '20'] = d[c].rolling(20, min_periods=15).sum()
            d[c + '60'] = d[c].rolling(60, min_periods=45).sum()
        pos = (d['frgn'] > 0).astype(int)
        d['frgn_streak'] = pos.groupby((pos == 0).cumsum()).cumsum()
        d['code'] = code
        out.append(d[['date', 'code', 'frgn20', 'orgn20', 'prsn20', 'pension20', 'frgn60', 'orgn60', 'prsn60', 'pension60', 'frgn_streak', 'frgn_hold', 'frgn_hold_chg60']])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    F = flows_table()
    mh = pd.read_csv(os.path.join(ROOT, 'data', 'dart', 'major_holders.csv'), dtype=str) if os.path.exists(os.path.join(ROOT, 'data', 'dart', 'major_holders.csv')) else pd.DataFrame()
    if len(mh):
        mh['dt'] = pd.to_datetime(mh['rcept_dt'])
        mh['foreign'] = mh['flr_nm'].str.contains(FOREIGN, flags=re.I, regex=True, na=False)
        mh['nps'] = mh['flr_nm'].str.contains(NPS, na=False)
    for level in ('coarse', 'ksic3', 'theme'):
        if not os.path.exists(os.path.join(WORK, f'panel_{level}_x.pkl')):
            continue
        pth = os.path.join(WORK, f'panel_{level}_x.pkl')
        p = pickle.load(open(pth, 'rb'))
        p = p.drop(columns=[c for c in p.columns if c.startswith(('frgn', 'orgn', 'prsn', 'pension', 'fo_', 'fo60', 'major_', 'sec_cap_rank', 'sec_val_rank', 'rep_combo'))], errors='ignore')
        if len(F):
            p = pd.merge_asof(p.sort_values('date'), F.sort_values('date'), on='date', by='code', direction='backward', tolerance=pd.Timedelta(days=7))
            for c in ('frgn20', 'orgn20', 'prsn20', 'frgn60', 'orgn60', 'pension60'):
                p[c + '_cap'] = p[c] / p['cap'] * 100       # 시가총액 대비 %
            p['fo60_cap'] = p['frgn60_cap'] + p['orgn60_cap']
            fo20 = p['frgn20'] + p['orgn20']; fo60 = p['frgn60'] + p['orgn60']
            p['fo_accel_cap'] = (fo20 - (fo60 - fo20) / 2) / p['cap'] * 100   # 종목 수급 기세: 최근 20일 매수 속도 - 그 전 40일 평균 속도
        # 섹터 안 대표성: 섹터 시총 순위, 섹터 거래대금 순위(valrank가 작을수록 거래대금 큼)
        g = p.groupby(['date', 'sector'])
        p['sec_cap_rank'] = g['cap'].rank(ascending=False)
        p['sec_val_rank'] = g['valrank'].rank(ascending=True)
        p['rep_combo'] = p['sec_cap_rank'] + p['sec_val_rank']
        if len(mh):
            for col, nm in (('foreign', 'major_foreign90'), ('nps', 'major_nps90')):
                ev = mh[mh[col]][['stock_code', 'dt']]
                cnt = []
                for t, g in p.groupby('date'):
                    e = ev[(ev['dt'] <= t) & (ev['dt'] > t - pd.Timedelta(days=90))]['stock_code'].value_counts()
                    cnt.append(g['code'].map(e).fillna(0).rename(nm))
                p[nm] = pd.concat(cnt)
        pickle.dump(p, open(pth, 'wb'))
        print(level, p.shape, p[[c for c in ['frgn20_cap', 'major_foreign90'] if c in p]].notna().mean().round(2).to_dict(), flush=True)


if __name__ == '__main__':
    main()
