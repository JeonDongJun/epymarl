#!/bin/bash

# QMIX 타겟 우선순위 시나리오 학습 스크립트
# 적 질럿 우선 공격과 스토커 우선 공격 시나리오를 각각 학습

echo "QMIX 타겟 우선순위 시나리오 학습을 시작합니다..."

# 환경 등록
echo "환경을 등록합니다..."
python -c "
import sys
import os
sys.path.append('src')
from src.envs import register_target_priority
register_target_priority()
print('타겟 우선순위 환경이 등록되었습니다.')
"

# 1. 적 질럿 우선 공격 시나리오 학습
echo "=== 적 질럿 우선 공격 시나리오 학습 시작 ==="
python src/main.py \
    --config=src/config/algs/qmix_zealot_priority.yaml \
    --env-config=src/config/envs/zealot_priority.yaml \
    --t-max=2050000 \
    --test-interval=50000 \
    --log-interval=50000 \
    --runner-log-interval=10000 \
    --learner-log-interval=10000 \
    --test-nepisode=100 \
    --test-greedy=True \
    --save-model=True \
    --save-model-interval=50000

echo "적 질럿 우선 공격 시나리오 학습이 완료되었습니다."

# 2. 적 스토커 우선 공격 시나리오 학습
echo "=== 적 스토커 우선 공격 시나리오 학습 시작 ==="
python src/main.py \
    --config=src/config/algs/qmix_stalker_priority.yaml \
    --env-config=src/config/envs/stalker_priority.yaml \
    --t-max=2050000 \
    --test-interval=50000 \
    --log-interval=50000 \
    --runner-log-interval=10000 \
    --learner-log-interval=10000 \
    --test-nepisode=100 \
    --test-greedy=True \
    --save-model=True \
    --save-model-interval=50000

echo "적 스토커 우선 공격 시나리오 학습이 완료되었습니다."

echo "모든 시나리오 학습이 완료되었습니다!"
echo "결과는 results/ 디렉토리에서 확인할 수 있습니다."
