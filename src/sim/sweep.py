"""한계 효과 점검: 기준 설정에서 한 번에 하나씩만 바꿔(+분할 구간 검증) 결과를 표로.
앞 구간(2016.6~2019)과 뒤 구간(2020~2023)에서 모두 좋아야 의미 있는 변화로 본다. 2024년 이후는 보지 않는다.
python3 -m sim.sweep <회차> <설정 json> [coarse|ksic3]
"""
import copy, json, os, sys
import numpy as np
import pandas as pd
from . import engine as E
from . import grid as G


def main(tag, cfg_file, level='coarse'):
    cfg = json.load(open(cfg_file, encoding='utf-8'))
    base = cfg.pop('_base', {})
    C = E.Ctx(level)
    rng = np.random.default_rng(11)
    G.SUBS = {'앞 2016.6-19': ('2016-06-01', '2019-12-31'), '뒤 2020-23': ('2020-01-01', '2023-12-31')}
    rows = []
    for name, ch in cfg.items():
        L = copy.deepcopy(E.BASE); L.update(base); L.update(ch)
        res, regs = E.run(C, L, start='2016-06-01', end='2023-12-31')
        rows += G.table(res, name, rng, '2016-06-01', '2023-12-31')
        print(name, flush=True)
    df = pd.DataFrame(rows)
    fmt = df.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == 'f':
            fmt[c] = fmt[c].map(lambda x: '' if pd.isna(x) else f'{x:.1f}')
    out = os.path.join(E.Dm.ROOT, 'docs', 'sim')
    md = [f'# 한계 효과 점검 {tag} (섹터 분류: {level})', '', '기준: ' + json.dumps(base, ensure_ascii=False), '',
          '앞·뒤 구간 칸은 각 구간의 시장 평균 대비(월평균). 두 구간 모두 기준보다 나아야 의미 있는 변화.', '', fmt.to_markdown(index=False)]
    open(os.path.join(out, f'{tag}.md'), 'w', encoding='utf-8').write('\n'.join(md))
    with open(os.path.join(out, 'log.md'), 'a', encoding='utf-8') as f:
        f.write(f'\n- {tag}: 한계 효과 점검 {len(cfg)}개(분류 {level})\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'coarse')
