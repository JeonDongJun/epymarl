#!/usr/bin/env python3
"""
Stalker Coordination 테스트 스크립트
"""

import sys
import os
sys.path.append('src')

import numpy as np
from envs.stalker_coordination_wrapper import StalkerCoordinationRewardWrapper
from envs.smac_wrapper import SMACWrapper

def test_stalker_coordination():
    """Stalker Coordination 래퍼 테스트"""
    print("=== Stalker Coordination 테스트 시작 ===")
    
    try:
        # 기본 SMAC 환경 생성
        print("1. 기본 SMAC 환경 생성 중...")
        base_env = SMACWrapper(map_name="3s5z_vs_3s6z", seed=42)
        
        # Stalker Coordination 래퍼 적용
        print("2. Stalker Coordination 래퍼 적용 중...")
        wrapped_env = StalkerCoordinationRewardWrapper(
            base_env,
            coordination_reward=0.2,
            stalker_role_id=0
        )
        
        print(f"   - 에이전트 수: {wrapped_env.n_agents}")
        print(f"   - 에피소드 제한: {wrapped_env.episode_limit}")
        
        # 환경 리셋
        print("3. 환경 리셋 중...")
        obs = wrapped_env.reset()
        print(f"   - 관찰 형태: {[len(o) for o in obs]}")
        
        # 몇 스텝 실행하여 협력 리워드 테스트
        print("4. 협력 공격 시뮬레이션 중...")
        
        for step in range(5):
            # 랜덤 액션 생성 (일부는 공격 액션으로 설정)
            actions = []
            for i in range(wrapped_env.n_agents):
                if i < 3:  # 처음 3개 에이전트는 Stalker로 가정
                    # 공격 액션 시뮬레이션 (1~5는 적 공격)
                    action = np.random.choice([0, 1, 2, 3, 4, 5])
                else:
                    # Zealot는 다른 액션
                    action = np.random.choice([0, 6, 7, 8])
                actions.append(action)
            
            print(f"   스텝 {step + 1}: 액션 = {actions}")
            
            # 스텝 실행
            obs, reward, terminated, truncated, info = wrapped_env.step(actions)
            
            print(f"   - 리워드: {reward}")
            print(f"   - 종료: {terminated}")
            
            if terminated:
                print("   에피소드 종료!")
                break
        
        print("5. 테스트 완료!")
        
        # Stalker 식별 테스트
        print("6. Stalker 식별 테스트...")
        for i in range(wrapped_env.n_agents):
            is_stalker = wrapped_env._is_stalker(i)
            print(f"   에이전트 {i}: {'Stalker' if is_stalker else 'Zealot'}")
        
        wrapped_env.close()
        print("=== 테스트 성공적으로 완료 ===")
        return True
        
    except Exception as e:
        print(f"=== 테스트 실패: {e} ===")
        import traceback
        traceback.print_exc()
        return False

def test_reward_calculation():
    """리워드 계산 로직 테스트"""
    print("\n=== 리워드 계산 테스트 ===")
    
    try:
        # 가상의 환경과 래퍼 생성
        base_env = SMACWrapper(map_name="3s5z_vs_3s6z", seed=42)
        wrapped_env = StalkerCoordinationRewardWrapper(
            base_env,
            coordination_reward=0.2,
            stalker_role_id=0
        )
        
        # 협력 공격 시뮬레이션
        actions = [1, 1, 2, 0, 0, 0, 0, 0]  # Stalker 3명이 같은 적(1) 공격
        
        obs = wrapped_env.reset()
        obs, reward, terminated, truncated, info = wrapped_env.step(actions)
        
        print(f"협력 공격 리워드: {reward}")
        
        wrapped_env.close()
        return True
        
    except Exception as e:
        print(f"리워드 계산 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("Stalker Coordination 시스템 테스트")
    print("=" * 50)
    
    # 기본 테스트
    success1 = test_stalker_coordination()
    
    # 리워드 계산 테스트
    success2 = test_reward_calculation()
    
    if success1 and success2:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("\n사용 방법:")
        print("1. 환경 설정: src/config/envs/stalker_coordination.yaml")
        print("2. 알고리즘 설정: src/config/algs/qmix_stalker_coordination.yaml")
        print("3. 실행 명령:")
        print("   python src/main.py --config=stalker_coordination --config=qmix_stalker_coordination")
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
        sys.exit(1)
