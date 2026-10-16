# 로직 시뮬레이션·검토 절차

1. 엔진(src/sim)이 매월 말 그날까지 알 수 있던 자료만으로 종목을 뽑고 3·6개월 뒤 성과를 매긴다. 계산은 전부 코드로 한다.
2. 회차마다 로직 단계별 요약표(docs/sim/<회차>.md)를 만든다.
3. 다섯 관점(버핏·드라켄밀러·최준철·버리·캐시 우드) 검토자가 요약표만 읽고 각자 수정안(최대 3개, 바꿀 설정값과 이유)을 낸다.
4. 수정안을 종합해 다음 회차 설정을 정하고 다시 돌린다. 바꾼 이유와 전후 결과를 docs/sim/log.md에 남긴다.
5. 2024년 이후 구간은 수정 판단에 쓰지 않는다(최종 확인용). 회차가 늘수록 결과를 보수적으로 본다.

조정 가능한 설정: cap_rank, val_rank, stages, types, b_lag_floor, b_ma60_floor, rsi_hot, gate_rights_days, gate_loss, need_earn, rev_yoy, n_max, rank(ry=매출 증가율 / rs3=섹터 대비 3개월 강도), stop, stop_pct, cost
