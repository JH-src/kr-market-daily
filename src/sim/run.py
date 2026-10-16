"""로직 버전들을 한 번에 돌려 요약표를 만든다.
python3 -m sim.run <회차 이름> [변경안 json 파일]
결과: docs/sim/<회차>.md (요약), 작업 폴더에 종목별 상세 csv
"""
import copy, json, os, sys
import pandas as pd
from . import engine as E

OUT = os.path.join(E.Dm.ROOT, 'docs', 'sim')


def variants(base):
    v = {}
    def mk(name, **kw):
        L = copy.deepcopy(base); L.update(kw); L['name'] = name; v[name] = L
    mk('0 전체 유니버스(비교 기준)', stages=['주도', '초기', '중립', '과열 주의', '쇠퇴'], types=[], rsi_hot=None, gate_rights_days=0,
       gate_loss=False, need_earn=False, n_max=None, stop=False)
    mk('1 주도·초기 섹터만', types=[], rsi_hot=None, gate_rights_days=0, gate_loss=False, need_earn=False, n_max=None, stop=False)
    mk('2 + 유형(초기 주도주·차기주자)', rsi_hot=None, gate_rights_days=0, gate_loss=False, need_earn=False, n_max=None, stop=False)
    mk('3 + 탈락 관문', need_earn=False, n_max=None, stop=False)
    mk('4 + 실적 상승세', n_max=None, stop=False)
    mk('5 최종(국면별 개수·순위)', stop=False)
    mk('6 최종 + 손절')
    mk('6A 최종: 초기 섹터의 주도주만', types=['A'])
    mk('6B 최종: 주도 섹터의 차기주자만', types=['B'])
    return v


def main(tag, change_file=None):
    base = copy.deepcopy(E.BASE)
    if change_file:
        base.update(json.load(open(change_file, encoding='utf-8')))
    C = E.Ctx()
    rows, detail = [], []
    for name, L in variants(base).items():
        res, regs = E.run(C, L)
        rows.append(E.summary(res, regs, name))
        for h, d in res.items():
            if not d.empty:
                detail.append(d.assign(logic=name, h=h))
    s = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    work = os.environ.get('SIM_CACHE', '/tmp/sim_cache')
    if detail:
        pd.concat(detail).to_csv(os.path.join(work, f'{tag}_detail.csv.gz'), index=False)
    fmt = s.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == 'f':
            fmt[c] = fmt[c].map(lambda x: f'{x:.1f}')
    md = [f'# 시뮬레이션 {tag}', '', f'기준 로직: {json.dumps(base, ensure_ascii=False)}', '',
          f'자료 기간: {C.dates.min():%Y-%m-%d} ~ {C.dates.max():%Y-%m-%d}, 종목 {C.P.shape[1]}개, 매월 말 판단·다음 거래일 종가 매수, 비용 {base["cost"]*100:.1f}% 차감', '',
          fmt.to_markdown(index=False)]
    open(os.path.join(OUT, f'{tag}.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
