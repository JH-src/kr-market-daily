"""두 탐색 누적 상태(src -> dst)를 합친다: 결과표는 설정 기준 중복 제거, 달별 성과도 합집합."""
import os, sys
import pandas as pd
from . import search as S


def main(src, dst):
    rs = []
    for d in (dst, src):
        f = os.path.join(d, 'search_results_v2.csv.gz')
        try:
            if os.path.exists(f):
                rs.append(pd.read_csv(f))
        except Exception as e:
            print('결과표 읽기 실패(건너뜀)', f, e)
    r = pd.concat(rs, ignore_index=True).drop_duplicates('cfg', keep='last') if rs else pd.DataFrame()
    s = S.load_series(os.path.join(dst, 'search_series_v2.npz'))
    try:
        s.update(S.load_series(os.path.join(src, 'search_series_v2.npz')))
    except Exception as e:
        print('달별 성과 읽기 실패(건너뜀)', e)
    os.makedirs(dst, exist_ok=True)
    S.save_csv(r, os.path.join(dst, 'search_results_v2.csv.gz'))
    S.save_series(s, os.path.join(dst, 'search_series_v2.npz'))
    ok = r.dropna(subset=['score']) if 'score' in r else r.iloc[0:0]
    S.write_md(r, ok, S.walk_forward(s), S.walk_forward(s, K=20))
    print('merged', len(r), len(s))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
