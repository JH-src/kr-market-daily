"""두 탐색 누적 상태(src -> dst)를 합친다: 결과표는 설정 기준 중복 제거, 달별 성과도 합집합."""
import os, sys
import pandas as pd
from . import search as S


def main(src, dst):
    rs = [pd.read_csv(os.path.join(d, 'search_results.csv.gz')) for d in (dst, src) if os.path.exists(os.path.join(d, 'search_results.csv.gz'))]
    r = pd.concat(rs, ignore_index=True).drop_duplicates('cfg', keep='last') if rs else pd.DataFrame()
    s = S.load_series(os.path.join(dst, 'search_series.npz'))
    s.update(S.load_series(os.path.join(src, 'search_series.npz')))
    os.makedirs(dst, exist_ok=True)
    r.to_csv(os.path.join(dst, 'search_results.csv.gz'), index=False)
    S.save_series(s, os.path.join(dst, 'search_series.npz'))
    ok = r.dropna(subset=['score'])
    S.write_md(r, ok, S.walk_forward(s))
    print('merged', len(r), len(s))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
