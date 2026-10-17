"""리포트 보강: 수출 품목 비교 그래프와 대표 기업, 선정 종목 설명 카드, 선정 종목 추적.

report.py에서 불러 쓴다. 숫자는 저장된 원자료에서만 계산하고, 없으면 '확인 불가'.
"""
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import report as R

ROOT, DATA = R.ROOT, R.DATA
SUBSET = {'전력용 변압기(대형)', '전기 승용차', '철강 판재(열연·후판)', '태양전지'}   # 다른 품목에 포함되는 하위 품목: 묶음 합계에서 제외


def _sectors():
    from sim import data as D
    return D.sectors('theme')


def _last_caps(px, asof):
    d = px[px['BAS_DD'] <= asof]
    last = d[d['BAS_DD'] == d['BAS_DD'].max()]
    return last.set_index('ISU_CD')['MKTCAP'], last.set_index('ISU_CD')['ISU_NM']


def _ret(px, codes, asof, days):
    d = px[(px['BAS_DD'] <= asof) & px['ISU_CD'].isin(codes)].pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    if len(d) <= days:
        return {}
    return ((d.iloc[-1] / d.iloc[-1 - days] - 1) * 100).to_dict()


# ---------------------------------------------------------------- 수출: 평균·중위 대비
def export_beat(px, asof):
    p = os.path.join(DATA, 'customs', 'items_monthly.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype={'yymm': str})
    pv = d.pivot(index='yymm', columns='item', values='exp_usd') / 1e6
    t = pd.read_csv(os.path.join(DATA, 'customs', 'export10d.csv'), dtype=str)
    t = t[t['period'].str[3:5].astype(int) > 20]
    tot = pd.Series(pd.to_numeric(t['전체'].str.replace(',', ''), errors='coerce').values / 1e3, index=t['yymm'].values)
    ym = pv.index.max()
    y, m = int(ym[:4]), int(ym[4:])
    last3 = [f'{y - (m - k <= 0)}{(m - k - 1) % 12 + 1:02d}' for k in (2, 1, 0)]
    prev3 = [f'{int(x[:4]) - 1}{x[4:]}' for x in last3]
    py = f'{y - 1}{m:02d}'

    def g3(s):
        a, b = s.reindex(last3).sum(min_count=3), s.reindex(prev3).sum(min_count=3)
        # 기저가 너무 작으면(전년 3개월 1억 달러 미만) 증가율이 의미 없어 제외
        return (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b >= 100 else np.nan

    def g1(s):
        a, b = s.get(ym, np.nan), s.get(py, np.nan)
        return (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b >= 30 else np.nan

    rows = []
    for it in pv.columns:
        grp, nm = it.split('|', 1)
        rows.append({'key': it, 'grp': grp, 'item': nm, 'g3': g3(pv[it]), 'g1': g1(pv[it]),
                     'amt3': pv[it].reindex(last3).sum(), 'amt1': pv[it].get(ym, np.nan)})
    df = pd.DataFrame(rows)
    tot3, tot1 = g3(tot), g1(tot)
    med3 = df['g3'].median()
    # 묶음(섹터) 합계: 하위 품목 빼고 더함
    grows = []
    for grp, gg in df[~df['item'].isin(SUBSET)].groupby('grp'):
        s = pv[gg['key']].sum(axis=1, min_count=1)
        grows.append({'grp': grp, 'g3': g3(s), 'g1': g1(s), 'amt3': s.reindex(last3).sum()})
    gdf = pd.DataFrame(grows).sort_values('g3', ascending=False)

    # 대표 기업: 지정 코드 우선, 다음 섹터 시가총액 상위
    mp = json.load(open(os.path.join(ROOT, 'config', 'export_map.json')))
    cap, nm = _last_caps(px, asof)
    reps = {}
    for it in df['key']:
        e = mp.get(it, {})
        codes = list(e.get('codes', []))
        seen = []
        for c in codes:
            if c in cap.index and c not in seen:
                seen.append(c)
        reps[it] = seen[:4]
    allc = sorted({c for v in reps.values() for c in v})
    r3 = _ret(px, allc, asof, 63)
    df['reps'] = df['key'].map(lambda k: [(nm.get(c, c), r3.get(c, np.nan)) for c in reps.get(k, [])])
    df = df.sort_values('g3', ascending=False)
    return {'ym': ym, 'months': f"{last3[0][:4]}.{last3[0][4:]}~{last3[-1][4:]}", 'tot3': tot3, 'tot1': tot1, 'med3': med3,
            'items': df, 'groups': gdf, 'reps_map': reps}


def chart_export_beat(eb):
    df = eb['items'].dropna(subset=['g3']).iloc[::-1]
    cap = 150
    v = df['g3'].clip(-60, cap)
    col = [R.UP if x > eb['tot3'] else ('#e3a39b' if x > eb['med3'] else '#9ca3af') for x in df['g3']]
    fig, ax = plt.subplots(figsize=(7.2, 0.17 * len(df) + 0.9))
    ax.barh(range(len(df)), v, color=col, height=0.7)
    ax.scatter(df['g1'].clip(-60, cap), range(len(df)), s=9, color=R.INK, zorder=3, label=f"{eb['ym'][4:]}월 한 달")
    for i, (x, raw) in enumerate(zip(v, df['g3'])):
        if raw > cap:
            ax.text(cap + 2, i, f"{raw:+.0f}%", va='center', fontsize=6.5, color=R.UP)
    ax.axvline(eb['tot3'], color=R.UP, lw=1, ls='--')
    ax.axvline(eb['med3'], color=R.MUTED, lw=1, ls=':')
    ax.text(eb['tot3'], len(df) + 0.2, f"전체 수출 {eb['tot3']:+.0f}%", color=R.UP, fontsize=7, ha='left')
    ax.text(eb['med3'], len(df) + 0.9, f"품목 중위 {eb['med3']:+.0f}%", color=R.MUTED, fontsize=7, ha='left')
    ax.axvline(0, color=R.INK, lw=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{i}" for i in df['item']], fontsize=6.8)
    ax.set_xlim(-62, cap + 25)
    R.style_ax(ax); ax.grid(axis='x', color=R.GRID, lw=0.5); ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax.set_title(f"품목별 수출 증가율: 최근 3개월({eb['months']}) 합계 전년 대비(막대), 점은 최근 한 달", fontsize=8.5, loc='left', color=R.INK)
    return R.fig_b64(fig)


def chart_export_groups(eb):
    g = eb['groups'].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    col = [R.UP if x > eb['tot3'] else R.MUTED for x in g['g3']]
    ax.barh(range(len(g)), g['g3'], color=col, height=0.6)
    for i, (x, a) in enumerate(zip(g['g3'], g['amt3'])):
        ax.text(max(x, 0) + 2, i, f"{x:+.0f}% (3개월 ${a / 1000:,.1f}십억)", va='center', ha='left', fontsize=7)
    ax.axvline(eb['tot3'], color=R.UP, lw=1, ls='--')
    ax.axvline(0, color=R.INK, lw=0.6)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g['grp'], fontsize=7.5)
    R.style_ax(ax); ax.grid(axis='y', visible=False)
    ax.set_xlim(min(-10, g['g3'].min() - 5), g['g3'].max() * 1.5 + 10)
    ax.set_title(f"묶음별 최근 3개월 수출 전년 대비(빨강: 전체 {eb['tot3']:+.0f}% 상회)", fontsize=8.5, loc='left', color=R.INK)
    return R.fig_b64(fig)


def export_section(eb):
    if eb is None:
        return ["<h2>수출 품목 비교</h2><div class=box>품목별 수출 자료 확인 불가</div>"]
    df = eb['items']
    beat = df[df['g3'] > eb['tot3']]
    above = df[df['g3'] > eb['med3']]
    h = ["<h2>수출: 전체를 이기는 품목과 대표 기업</h2>",
         f"<div class=box><b>최근 3개월 전체 수출 {R.fmt_pct(eb['tot3'], 0)}, 41개 품목 중위 {R.fmt_pct(eb['med3'], 0)}. "
         f"전체보다 빨리 느는 품목 {len(beat)}개</b>: " + ', '.join(f"{r.item}({r.g3:+.0f}%)" for r in beat.itertuples()) + "</div>",
         f"<img src='{chart_export_groups(eb)}'>", f"<img src='{chart_export_beat(eb)}'>",
         "<div class=note>진한 빨강: 전체 수출보다 높음 / 연한 빨강: 품목 중위보다만 높음 / 회색: 중위 이하. 150% 넘는 값은 잘라 그리고 숫자로 표시. "
         + (f"전년 기저가 작아(3개월 1억 달러 미만) 증가율에서 뺀 품목: {', '.join(df[df['g3'].isna()]['item'])}. " if df['g3'].isna().any() else '') +
         "묶음 합계는 다른 품목에 포함되는 하위 품목(대형 변압기, 전기 승용차, 열연·후판, 태양전지)을 빼고 더함. 3개월 합계는 같은 달끼리 비교라 조업일수 차이가 작음.</div>",
         "<h3>품목 중위보다 빨리 느는 품목과 대표 상장사(최근 3개월 주가)</h3>",
         "<table><tr><th>품목</th><th class=r>3개월 수출</th><th class=r>전년 대비(▲ 전체 상회)</th><th>대표 기업(최근 3개월 주가)</th></tr>"]
    for r in above.itertuples():
        reps = ', '.join(f"{n} {R.color_pct(x, 0)}" for n, x in r.reps) or '<span class=mut>확인 불가</span>'
        h.append(f"<tr><td>{r.item}<br><span class=mut>{r.grp}</span></td><td class=r>${r.amt3:,.0f}백만</td><td class=r>{R.color_pct(r.g3, 0)}{' ▲' if r.g3 > eb['tot3'] else ''}</td><td>{reps}</td></tr>")
    h.append("</table><div class=note>대표 기업은 품목별로 미리 정해 둔 주요 상장사(config/export_map.json). 수출 실적이 그 기업 매출과 바로 같다는 뜻은 아님.</div>")
    return h


# ---------------------------------------------------------------- 선정 종목 설명 카드
def _business(code):
    p = os.path.join(DATA, 'dart', 'business.csv')
    if not os.path.exists(p):
        return ''
    b = pd.read_csv(p, dtype={'stock_code': str}, usecols=['stock_code', 'summary'])
    s = b[b['stock_code'] == code]
    if s.empty or not isinstance(s['summary'].iloc[0], str):
        return ''
    t = re.sub(r'\s+', ' ', s['summary'].iloc[0]).strip()
    sents = [x.strip() for x in re.split(r'(?<=[다음함])\.\s', t) if len(x.strip()) > 15]
    keep = [x for x in sents if not re.search(r'연결대상|종속회사|회계|기준서|작성|본 사업보고서|보고기간', x)]
    out = '. '.join(keep[:2])
    return (out[:220] + '…') if len(out) > 220 else out


_FQ = None


def _quarters(code):
    global _FQ
    from sim import data as D
    if _FQ is None:
        _FQ = D.fin_quarters()
        _FQ['pri'] = (_FQ['fs'] != 'CFS').astype(int)
        _FQ = _FQ.sort_values(['stock_code', 'y', 'q', 'pri']).drop_duplicates(['stock_code', 'y', 'q'])
    f = _FQ[_FQ['stock_code'] == code].sort_values(['y', 'q'])
    return f.tail(5)


def _flows(code, asof):
    ds = asof.strftime('%Y%m%d')
    rows = []
    p = os.path.join(ROOT, 'archive', 'kis_flows', f'{code}.csv.gz')
    if os.path.exists(p):
        rows.append(pd.read_csv(p, dtype=str))
    for f in glob.glob(os.path.join(DATA, 'kis', 'flows', '*.csv')):
        d = pd.read_csv(f, dtype=str)
        rows.append(d[d['code'] == code])
    if not rows:
        return None
    d = pd.concat(rows).drop_duplicates('date', keep='last')
    d = d[d['date'] <= ds].sort_values('date')
    if len(d) < 20:
        return None
    for c in ['frgn', 'orgn', 'prsn', 'pension', 'close']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    # 단위: 주식 수 → 금액(억원) = 주 × 종가
    out = {}
    for w in (5, 20, 60):
        x = d.tail(w)
        for c in ['frgn', 'orgn', 'prsn', 'pension']:
            out[f'{c}{w}'] = (x[c] * x['close']).sum() / 1e8
    s = 0
    for v in d['frgn'].iloc[::-1]:
        if v > 0:
            s += 1
        else:
            break
    out['streak'] = s
    out['last'] = d['date'].iloc[-1]
    return out


FOREIGN = re.compile(r'골드만|Goldman|모건|Morgan|JP\s?모간|J\.?P\.?\s?Morgan|블랙록|BlackRock|피델리티|Fidelity|GIC|싱가포르|노르웨이|Norges|뱅가드|Vanguard|캐피탈\s?그룹|Capital|UBS|씨티|Citi|메릴|Merrill|크레디트|Credit|노무라|Nomura|맥쿼리|Macquarie|아부다비|ADIA|템플턴|Templeton|매튜스|Matthews|라자드|Lazard|국민연금', re.I)


def _holders(code, asof, days=365):
    p = os.path.join(DATA, 'dart', 'major_holders.csv')
    d = pd.read_csv(p, dtype=str)
    d = d[(d['stock_code'] == code)]
    d['dt'] = pd.to_datetime(d['rcept_dt'])
    d = d[(d['dt'] <= asof) & (d['dt'] > asof - pd.Timedelta(days=days))]
    d = d[d['flr_nm'].fillna('').str.contains(FOREIGN)]
    g = d.groupby('flr_nm')['rcept_dt'].agg(['max', 'count']).sort_values('max', ascending=False)
    return [(r['max'], n, r['count']) for n, r in g.iterrows()][:4]


def _news(asof, code):
    p = os.path.join(DATA, 'news', f"{asof:%Y%m%d}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get(code)


def pick_cards(sp, eb, px, asof):
    if not sp['rows']:
        return []
    item_by_sec = {}
    mp = json.load(open(os.path.join(ROOT, 'config', 'export_map.json')))
    for it, e in mp.items():
        if it.startswith('_'):
            continue
        for s in e.get('sectors', []):
            item_by_sec.setdefault(s, []).append(it)
        for c in e.get('codes', []):
            item_by_sec.setdefault(c, []).append(it)
    h = ["<h2 class=pb>선정 종목 설명</h2><div class=note>회사 소개는 최근 사업보고서 '사업의 내용' 앞부분, 실적은 DART 분기(3개월) 기준, "
         "수급은 한국투자증권 종목별 투자자 매매(주식 수 × 종가로 금액 환산), 5% 공시는 최근 1년 외국계·국민연금 대량보유 보고. 기사는 제목과 출처만.</div>"]
    tn = {'A': '초기 섹터의 주도주', 'B': '주도 섹터의 차기주자'}
    for x in sp['rows']:
        c = x['code']
        h.append(f"<h3>{x['name']} <span class=mut>({c}) · {x['sector']}({x['stage']}) · {tn.get(x['type'], '')}</span></h3>")
        import biz_text
        try:
            biz, src = biz_text.summary(c)
        except Exception:
            biz, src = '', ''
        h.append(f"<div><b>무엇을 하나</b> {biz + f' <span class=mut>({src})</span>' if biz else '<span class=mut>사업보고서 요약 확인 불가</span>'}</div>")
        q = _quarters(c)
        if len(q):
            cells = []
            for r in q.itertuples():
                cells.append(f"<td class=r>{r.y % 100}.{r.q}Q<br>{r.rev / 1e8:,.0f}<br>{r.op / 1e8:,.0f}</td>")
            yo = ''
            a = q.iloc[-1]
            if pd.notna(a.get('rev_py')) and a['rev_py'] > 0:
                yo = f" · {a['y'] % 100}.{a['q']}Q 전년 대비 매출 {R.fmt_pct((a['rev'] / a['rev_py'] - 1) * 100, 0)}, 영업이익 " + \
                     (f"{R.fmt_pct((a['op'] / a['op_py'] - 1) * 100, 0)}" if pd.notna(a.get('op_py')) and a['op_py'] > 0 else ('흑자전환' if a['op'] > 0 else '적자 지속'))
            h.append(f"<div><b>실적</b>(억원, 위: 매출 / 아래: 영업이익){yo}</div><table style='width:auto'><tr><td class=mut>분기<br>매출<br>영업이익</td>{''.join(cells)}</tr></table>")
        else:
            h.append("<div><b>실적</b> <span class=mut>확인 불가</span></div>")
        fl = _flows(c, asof)
        if fl:
            jb = fl['frgn20'] > 0 and fl['orgn20'] > 0 and fl['prsn20'] < 0
            h.append(f"<div><b>수급</b>(억원) 외국인 5일 {fl['frgn5']:+,.0f} · 20일 {fl['frgn20']:+,.0f} · 60일 {fl['frgn60']:+,.0f} / "
                     f"기관 20일 {fl['orgn20']:+,.0f} / 개인 20일 {fl['prsn20']:+,.0f} / 연기금 20일 {fl['pension20']:+,.0f}"
                     f" / 외국인 연속 순매수 {fl['streak']}일" + (" · <b>외국인·기관 동반 매수, 개인 매도</b>" if jb else '') +
                     f" <span class=mut>({fl['last'][4:6]}/{fl['last'][6:]}까지)</span></div>")
        else:
            h.append("<div><b>수급</b> <span class=mut>확인 불가</span></div>")
        ho = _holders(c, asof)
        h.append("<div><b>외국계·연금 5% 공시(1년)</b> " + (', '.join(f"{n} {k}건(최근 {d[2:4]}.{d[4:6]}.{d[6:]})" for d, n, k in ho) if ho else '없음') + "</div>")
        its = item_by_sec.get(c, []) + item_by_sec.get(x['sector'], [])
        if eb is not None and its:
            dfi = eb['items'].set_index('key')
            parts = []
            for it in dict.fromkeys(its):
                if it in dfi.index:
                    r = dfi.loc[it]
                    mark = ' ▲전체 상회' if r['g3'] > eb['tot3'] else ''
                    parts.append(f"{r['item']} 3개월 {R.fmt_pct(r['g3'], 0)}, {eb['ym'][4:]}월 {R.fmt_pct(r['g1'], 0)}{mark}")
            h.append("<div><b>관련 수출</b> " + ' / '.join(parts) + "</div>")
        else:
            h.append("<div><b>관련 수출</b> <span class=mut>연결된 수출 품목 없음</span></div>")
        try:
            _, hold, watch = active_etfs(asof)
            tg = etf_tags(c, hold, watch)
        except Exception:
            tg = []
        h.append("<div><b>액티브 ETF 보유</b> " + (', '.join(tg) if tg else '추적 ETF 상위 30 안에 없음') + "</div>")
        nw = _news(asof, c)
        if nw is None:
            h.append("<div><b>최근 기사</b> <span class=mut>확인 불가</span></div>")
        elif nw:
            h.append("<div><b>최근 기사</b><br>" + '<br>'.join(f"· {a.get('date', '')} {a['title']} <span class=mut>({a.get('src', '')})</span>" for a in nw[:4]) + "</div>")
            if any(a.get('point') for a in nw):
                h.append("<div><b>기사로 본 이유</b> " + ' '.join(a['point'] for a in nw if a.get('point')) + "</div>")
        else:
            h.append("<div><b>최근 기사</b> 찾지 못함</div>")
    return h


# ---------------------------------------------------------------- 선정 종목 추적
LOG = os.path.join(DATA, 'picks', 'log.csv')


def track_update(sp, asof):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    ds = sp['asof'].strftime('%Y%m%d')
    old = pd.read_csv(LOG, dtype=str) if os.path.exists(LOG) else pd.DataFrame(columns=['date', 'code', 'name', 'sector', 'stage', 'type', 'close', 'logic'])
    old = old[old['date'] != ds]
    new = pd.DataFrame([{'date': ds, 'code': x['code'], 'name': x['name'], 'sector': x['sector'], 'stage': x['stage'],
                         'type': x['type'], 'close': f"{x['close']:.0f}", 'logic': R.SIM_LOGIC['name']} for x in sp['rows']])
    pd.concat([old, new]).sort_values(['date', 'code']).to_csv(LOG, index=False)


def track_section(px, asof):
    if not os.path.exists(LOG):
        return []
    lg = pd.read_csv(LOG, dtype=str)
    if lg.empty:
        return []
    first = lg.sort_values('date').groupby('code').agg(date=('date', 'first'), name=('name', 'first'), sector=('sector', 'last'),
                                                       close=('close', 'first'), n=('date', 'nunique'), last=('date', 'last')).reset_index()
    d = px[(px['BAS_DD'] <= asof) & px['ISU_CD'].isin(first['code'])]
    last = d.sort_values('BAS_DD').groupby('ISU_CD')['TDD_CLSPRC'].last()
    # 시장: 시가총액 상위 700 동일가중(시뮬 기준선과 같은 성격)의 근사로 코스피 지수 사용
    k = R.series_from('idx_kospi', 'IDX_NM', '코스피', 'CLSPRC_IDX') if hasattr(R, 'series_from') else None
    rows = []
    today = asof.strftime('%Y%m%d')
    for r in first.itertuples():
        p0 = float(r.close); p1 = last.get(r.code, np.nan)
        kr = np.nan
        try:
            k0 = k[k.index <= pd.Timestamp(r.date)].iloc[-1]; k1 = k[k.index <= asof].iloc[-1]
            kr = (k1 / k0 - 1) * 100
        except Exception:
            pass
        rr = (p1 / p0 - 1) * 100
        rows.append((r, rr, kr, r.last == today))
    rows.sort(key=lambda z: z[0].date, reverse=True)
    h = ["<h2>선정 종목 추적</h2>", "<table><tr><th>종목</th><th>처음 선정</th><th class=r>선정 일수</th><th class=r>선정 후</th><th class=r>코스피</th><th class=r>초과</th><th>오늘</th></tr>"]
    for r, rr, kr, on in rows[:30]:
        h.append(f"<tr><td>{r.name} <span class=mut>{r.sector}</span></td><td>{r.date[2:4]}.{r.date[4:6]}.{r.date[6:]}</td><td class=r>{r.n}</td>"
                 f"<td class=r>{R.color_pct(rr)}</td><td class=r>{R.color_pct(kr)}</td><td class=r>{R.color_pct(rr - kr)}</td><td>{'유지' if on else '<span class=mut>제외</span>'}</td></tr>")
    v = [rr - kr for _, rr, kr, _ in rows if pd.notna(rr - kr)]
    h.append("</table>")
    if v:
        h.append(f"<div class=note>추적 {len(rows)}종목, 코스피 대비 초과수익 평균 {np.mean(v):+.1f}%p, 중위 {np.median(v):+.1f}%p, 이긴 비율 {np.mean(np.array(v) > 0) * 100:.0f}%. "
                 "선정 후 수익은 처음 선정된 날 종가 기준, 비용 미반영.</div>")
    return h


def replay_section(px, asof):
    p = os.path.join(DATA, 'picks', 'replay.csv')
    if not os.path.exists(p):
        return []
    rp = pd.read_csv(p, dtype=str)
    if rp.empty:
        return []
    d0 = px[px['BAS_DD'] <= asof]
    cl = d0.pivot_table(index='BAS_DD', columns='ISU_CD', values='TDD_CLSPRC')
    cap = d0.pivot_table(index='BAS_DD', columns='ISU_CD', values='MKTCAP')
    k = R.series_from('idx_kospi', 'IDX_NM', '코스피', 'CLSPRC_IDX').reindex(cl.index).ffill()
    res = []
    for d, g in rp.groupby('date'):
        t = pd.Timestamp(d)
        if t not in cl.index:
            continue
        i = cl.index.get_loc(t)
        if i + 1 >= len(cl):
            continue
        j = min(i + 64, len(cl) - 1)          # 다음 날 종가 매수, 63거래일 뒤 또는 오늘
        b, e = cl.iloc[i + 1], cl.iloc[j]
        top = cap.iloc[i].rank(ascending=False) <= 700
        ew = ((e / b - 1)[top]).replace([np.inf, -np.inf], np.nan).dropna().clip(-0.9, 5).mean() * 100
        kr = (k.iloc[j] / k.iloc[i + 1] - 1) * 100
        for r in g.itertuples():
            if r.code in cl.columns and pd.notna(b.get(r.code)):
                res.append({'date': d, 'name': r.name, 'full': i + 64 < len(cl), 'ret': (e[r.code] / b[r.code] - 1) * 100, 'kr': kr, 'ew': ew})
    df = pd.DataFrame(res)
    if df.empty:
        return []
    df['xk'], df['xu'] = df['ret'] - df['kr'], df['ret'] - df['ew']
    f = df[df['full']]
    h = ["<h3>같은 로직으로 지난 1년 매주 골랐다면(재현)</h3>"]
    if len(f):
        h.append(f"<div class=box>3개월이 지난 선정 {len(f)}건({f['date'].min()[2:4]}.{f['date'].min()[4:6]}~{f['date'].max()[2:4]}.{f['date'].max()[4:6]} 매주): 평균 수익 {f['ret'].mean():+.1f}%<br>"
                 f"· <b>시총 상위 700 동일가중 대비</b>(보통 종목 대비) 평균 {f['xu'].mean():+.1f}%p, 중위 {f['xu'].median():+.1f}%p, 이긴 비율 {(f['xu'] > 0).mean() * 100:.0f}%<br>"
                 f"· <b>코스피 대비</b> 평균 {f['xk'].mean():+.1f}%p, 중위 {f['xk'].median():+.1f}%p, 이긴 비율 {(f['xk'] > 0).mean() * 100:.0f}%<br>"
                 f"<span class=note>같은 기간 코스피 3개월 수익 평균 {f['kr'].mean():+.1f}%, 상위 700 동일가중 {f['ew'].mean():+.1f}%: 지수가 소수 대형주로 오른 장이라 중소형 위주 선정은 코스피를 크게 밑돎. "
                 f"보통 종목보다는 조금 낫지만 중위가 마이너스라 소수 종목 성과에 기댐.</span></div>")
    rec = df[~df['full']]
    if len(rec):
        g = rec.groupby('date').agg(names=('name', lambda x: ', '.join(x)), ret=('ret', 'mean'), xu=('xu', 'mean'), xk=('xk', 'mean')).sort_index(ascending=False).head(10)
        h.append("<table><tr><th>선정일</th><th>종목</th><th class=r>오늘까지</th><th class=r>동일가중 대비</th><th class=r>코스피 대비</th></tr>")
        for d, r in g.iterrows():
            h.append(f"<tr><td>{d[2:4]}.{d[4:6]}.{d[6:]}</td><td>{r['names']}</td><td class=r>{R.color_pct(r['ret'])}</td><td class=r>{R.color_pct(r['xu'])}</td><td class=r>{R.color_pct(r['xk'])}</td></tr>")
        h.append("</table>")
    h.append("<div class=note>재현은 지금 로직과 그 시점까지 공시된 실적만으로 과거 매주 마지막 거래일에 골랐을 때의 결과. 다음 거래일 종가 매수, 3개월(63거래일) 보유, 비용 미반영, 같은 종목이 여러 주 연속 뽑히면 매번 따로 셈. 실제 선정 기록과 다름.</div>")
    return h


# ---------------------------------------------------------------- 추적 액티브 ETF: 자금 유출입·보유 종목
def active_etfs(asof):
    watch = {k: v for k, v in json.load(open(os.path.join(ROOT, 'config', 'watch_etf.json'), encoding='utf-8')).items() if not k.startswith('_')}
    e = R.load('etf', ['BAS_DD', 'ISU_CD', 'TDD_CLSPRC', 'NAV', 'LIST_SHRS', 'INVSTASST_NETASST_TOTAMT'])
    e = e[e['ISU_CD'].isin(watch.values())].copy()
    for c in ['TDD_CLSPRC', 'NAV', 'LIST_SHRS', 'INVSTASST_NETASST_TOTAMT']:
        e[c] = R.num(e[c])
    e['BAS_DD'] = pd.to_datetime(e['BAS_DD'])
    e = e[e['BAS_DD'] <= asof].sort_values('BAS_DD')
    # 보유 종목: 한투 추적 수집분(이전 수집일과 비교) 없으면 전체 ETF 구성 스냅샷
    hold = {}
    fs = sorted(glob.glob(os.path.join(DATA, 'kis', 'holdings', '*.csv')))
    fs = [f for f in fs if os.path.basename(f)[:8] <= (asof + pd.Timedelta(days=1)).strftime('%Y%m%d')]
    cur = pd.read_csv(fs[-1], dtype=str) if fs else pd.DataFrame(columns=['etf', 'code', 'name', 'weight'])
    snap = sorted(glob.glob(os.path.join(DATA, 'kis', 'etf_all', '*.csv')))
    alls = pd.read_csv(snap[-1], dtype=str) if snap else pd.DataFrame(columns=['etf', 'code', 'name', 'weight'])
    out = []
    for nm, code in watch.items():
        g = e[e['ISU_CD'] == code]
        r = {'name': nm, 'code': code}
        if len(g) > 21:
            nav = g['NAV'].where(g['NAV'] > 0, g['TDD_CLSPRC'])
            flow = g['LIST_SHRS'].diff() * nav / 1e8          # 설정(+)·환매(-) 금액, 억원
            aum = g['INVSTASST_NETASST_TOTAMT'].replace(0, np.nan).ffill().iloc[-1] / 1e8 if g['INVSTASST_NETASST_TOTAMT'].gt(0).any() else (g['LIST_SHRS'] * nav).iloc[-1] / 1e8
            r.update({'aum': aum, 'f5': flow.tail(5).sum(), 'f20': flow.tail(20).sum(), 'f60': flow.tail(60).sum(),
                      'r1': (g['TDD_CLSPRC'].iloc[-1] / g['TDD_CLSPRC'].iloc[-22] - 1) * 100, 'last': g['BAS_DD'].iloc[-1]})
        h = cur[cur['etf'] == code] if 'etf' in cur else cur.iloc[0:0]
        src = '한투 추적'
        if h.empty:
            h = alls[alls['etf'] == code]; src = '전체 ETF 구성 스냅샷'
        h = h.assign(w=R.num(h['weight'])).sort_values('w', ascending=False)
        h = h[h['code'].str.fullmatch(r'\d{6}', na=False)]
        r['top'] = list(zip(h['name'].head(10), h['w'].head(10))); r['src'] = src
        hold[code] = dict(zip(h['code'], h['w']))
        out.append(r)
    return out, hold, watch


def active_etf_section(asof):
    try:
        rows, hold, watch = active_etfs(asof)
    except Exception as ex:
        return [f"<div class=note>추적 ETF 자료 확인 불가({ex})</div>"]
    h = ["<h2>추적 액티브 ETF: 자금 흐름과 담은 종목</h2>",
         "<table><tr><th>ETF</th><th class=r>순자산(억)</th><th class=r>설정·환매 5일</th><th class=r>20일</th><th class=r>60일</th><th class=r>1개월 수익</th><th>비중 상위</th></tr>"]
    for r in rows:
        if 'aum' not in r:
            h.append(f"<tr><td>{r['name']}</td><td colspan=5 class=mut>거래소 자료 확인 불가</td><td>{', '.join(f'{n} {w:.1f}' for n, w in r['top'][:6])}</td></tr>"); continue
        h.append(f"<tr><td>{r['name']}</td><td class=r>{r['aum']:,.0f}</td><td class=r>{r['f5']:+,.0f}</td><td class=r>{r['f20']:+,.0f}</td><td class=r>{r['f60']:+,.0f}</td>"
                 f"<td class=r>{R.color_pct(r['r1'])}</td><td style='font-size:7.8pt'>{', '.join(f'{n} {w:.1f}' for n, w in r['top'][:6])}</td></tr>")
    h.append("</table><div class=note>설정·환매는 상장좌수 변화 × 기준가(억원, +는 돈이 들어옴). 보유 비중은 한국투자증권 ETF 구성종목(상위 30까지). "
             "보유 이력이 쌓이기 전이라 과거 시험에는 아직 못 넣고, 매일 수집해 편입·편출과 비중 변화를 쌓는 중.</div>")
    return h


def etf_tags(code, hold, watch):
    nm = {v: k for k, v in watch.items()}
    return [f"{nm[e]} {w:.1f}%" for e, d in hold.items() for c, w in d.items() if c == code]
