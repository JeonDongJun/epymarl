#!/bin/bash

# QMIX 타겟 우선순위 시나리오 테스트 스크립트
# 학습된 모델을 사용하여 각 시나리오를 테스트

echo "QMIX 타겟 우선순위 시나리오 테스트를 시작합니다..."

# 환경 등록
echo "환경을 등록합니다..."
python -c "
from src.envs import register_target_priority
register_target_priority()
print('타겟 우선순위 환경이 등록되었습니다.')
"

# 1. 적 질럿 우선 공격 시나리오 테스트
echo "=== 적 질럿 우선 공격 시나리오 테스트 ==="
python src/main.py \
    --config=src/config/algs/qmix_zealot_priority.yaml \
    --env-config=src/config/envs/zealot_priority.yaml \
    --test-nepisode=100 \
    --test-greedy=True \
    --evaluate=True \
    --load-step=2000000

echo "적 질럿 우선 공격 시나리오 테스트가 완료되었습니다."

# 2. 적 스토커 우선 공격 시나리오 테스트
echo "=== 적 스토커 우선 공격 시나리오 테스트 ==="
python src/main.py \
    --config=src/config/algs/qmix_stalker_priority.yaml \
    --env-config=src/config/envs/stalker_priority.yaml \
    --test-nepisode=100 \
    --test-greedy=True \
    --evaluate=True \
    --load-step=2000000

echo "적 스토커 우선 공격 시나리오 테스트가 완료되었습니다."

echo "모든 시나리오 테스트가 완료되었습니다!"
