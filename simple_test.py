#!/usr/bin/env python3
"""
간단한 환경 등록 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_wrapper_import():
    """래퍼 클래스 임포트 테스트"""
    print("=== 래퍼 클래스 임포트 테스트 ===")
    
    try:
        from src.envs.target_priority_wrapper import TargetPriorityRewardWrapper
        print("[OK] TargetPriorityRewardWrapper 임포트 성공")
        return True
    except Exception as e:
        print(f"[FAIL] TargetPriorityRewardWrapper 임포트 실패: {e}")
        return False

def test_config_loading():
    """설정 파일 로딩 테스트"""
    print("\n=== 설정 파일 로딩 테스트 ===")
    
    import yaml
    
    config_files = [
        "src/config/algs/qmix_zealot_priority.yaml",
        "src/config/algs/qmix_stalker_priority.yaml",
        "src/config/envs/zealot_priority.yaml",
        "src/config/envs/stalker_priority.yaml"
    ]
    
    success = True
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"[OK] {config_file} 로딩 성공")
        except Exception as e:
            print(f"[FAIL] {config_file} 로딩 실패: {e}")
            success = False
    
    return success

def main():
    """메인 테스트 함수"""
    print("QMIX 타겟 우선순위 시나리오 기본 테스트")
    print("=" * 50)
    
    tests = [
        test_wrapper_import,
        test_config_loading
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("테스트 결과 요약:")
    
    if all(results):
        print("[SUCCESS] 모든 테스트가 성공했습니다!")
        print("\n구현이 완료되었습니다. 다음 단계:")
        print("1. 학습 실행: ./run_target_priority_training.sh")
        print("2. 테스트 실행: ./test_target_priority.sh")
        print("3. 결과 확인: results/ 디렉토리")
    else:
        print("[FAIL] 일부 테스트가 실패했습니다.")
        print("오류를 확인하고 수정해주세요.")

if __name__ == "__main__":
    main()
