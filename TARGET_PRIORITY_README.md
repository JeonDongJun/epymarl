# QMIX 타겟 우선순위 시나리오

이 프로젝트는 QMIX 알고리즘을 사용하여 3s5z_vs_3s6z SMAC 환경에서 두 가지 다른 공격 전략을 학습합니다:

1. **적 질럿 우선 공격 시나리오**: 적의 Zealot을 먼저 공격하는 전략
2. **적 스토커 우선 공격 시나리오**: 적의 Stalker를 먼저 공격하는 전략

## 파일 구조

### 새로운 파일들
- `src/envs/target_priority_wrapper.py`: 타겟 우선순위 보상 래퍼
- `src/config/algs/qmix_zealot_priority.yaml`: 질럿 우선 공격 알고리즘 설정
- `src/config/algs/qmix_stalker_priority.yaml`: 스토커 우선 공격 알고리즘 설정
- `src/config/envs/zealot_priority.yaml`: 질럿 우선 공격 환경 설정
- `src/config/envs/stalker_priority.yaml`: 스토커 우선 공격 환경 설정
- `run_target_priority_training.sh`: 학습 스크립트
- `test_target_priority.sh`: 테스트 스크립트

### 수정된 파일들
- `src/envs/__init__.py`: 새로운 환경 등록 함수 추가

## 사용법

### 1. 학습 실행

```bash
# 모든 시나리오 학습
./run_target_priority_training.sh

# 또는 개별 실행
python src/main.py \
    --config=src/config/algs/qmix_zealot_priority.yaml \
    --env-config=src/config/envs/zealot_priority.yaml
```

### 2. 테스트 실행

```bash
# 학습된 모델 테스트
./test_target_priority.sh
```

### 3. 환경 등록

새로운 환경을 사용하기 전에 환경을 등록해야 합니다:

```python
from src.envs import register_target_priority
register_target_priority()
```

## 설정 파라미터

### 환경 설정 (env_args)
- `priority_target`: 우선 공격할 타겟 ("zealot" 또는 "stalker")
- `priority_reward`: 우선 타겟 공격 시 추가 보상 (기본값: 0.2)
- `stalker_role_id`: Stalker의 role ID (기본값: 0)
- `zealot_role_id`: Zealot의 role ID (기본값: 1)

### 알고리즘 설정
- `use_role_embedding`: True (role embedding 사용)
- `n_roles`: 2 (Stalker와 Zealot)
- `role_embed_dim`: 8 (role embedding 차원)

## 보상 함수

각 시나리오에서는 다음과 같은 보상이 제공됩니다:

1. **기본 SMAC 보상**: 적 유닛 처치, 승리/패배 보상
2. **우선순위 보상**: 우선 타겟을 공격할 때 추가 보상

## 결과 분석

학습 결과는 `results/` 디렉토리에 저장됩니다:
- `results/models/qmix_zealot_priority_*`: 질럿 우선 공격 모델
- `results/models/qmix_stalker_priority_*`: 스토커 우선 공격 모델
- `results/sacred/qmix_zealot_priority/`: 질럿 우선 공격 실험 로그
- `results/sacred/qmix_stalker_priority/`: 스토커 우선 공격 실험 로그

## 주의사항

1. SMAC 환경이 설치되어 있어야 합니다.
2. StarCraft II가 설치되어 있어야 합니다.
3. 학습에는 상당한 시간이 소요될 수 있습니다 (약 2M 스텝).
4. 타겟 ID 추출은 휴리스틱 기반이므로 실제 환경에 맞게 조정이 필요할 수 있습니다.

## 커스터마이징

### 보상 값 조정
환경 설정 파일에서 `priority_reward` 값을 조정하여 우선순위 보상의 강도를 변경할 수 있습니다.

### 타겟 식별 개선
`target_priority_wrapper.py`의 `_is_priority_target` 메서드를 수정하여 더 정확한 타겟 식별을 구현할 수 있습니다.

### 다른 맵 사용
환경 설정에서 `map_name`을 변경하여 다른 SMAC 맵을 사용할 수 있습니다.
