# StarCraft II Headless Mode 설정 가이드

## 🚫 학습 시 StarCraft II 화면이 나타나지 않도록 설정

### 1. 기본 설정 (권장)
기본적으로 모든 설정이 headless 모드로 구성되어 있습니다:

```bash
python src/main.py --config=qmix --env-config=sc2 env_args.map_name=3m
```

### 2. 명시적 headless 설정
환경 변수를 통해 headless 모드를 강제할 수 있습니다:

```bash
# Windows
set SC2HEADLESS=1
python src/main.py --config=qmix --env-config=sc2 env_args.map_name=3m

# Linux/Mac
export SC2HEADLESS=1
python src/main.py --config=qmix --env-config=sc2 env_args.map_name=3m
```

### 3. 설정 파일에서 headless 모드 확인
`src/config/envs/sc2.yaml`에서 다음 설정들이 활성화되어 있는지 확인:

```yaml
env_args:
  window_size_x: 0
  window_size_y: 0
```

**주의**: `window_pos_x`와 `window_pos_y`는 StarCraft2Env에서 지원되지 않으므로 제거되었습니다.

### 4. render 설정 확인
`src/config/default.yaml`에서 render가 False로 설정되어 있는지 확인:

```yaml
render: False  # StarCraft II 화면이 나타나지 않도록 설정
```

### 5. 문제 해결
만약 여전히 화면이 나타난다면:

1. **환경 변수 설정**:
   ```bash
   set SC2HEADLESS=1
   set SC2HEADLESS_HOST=127.0.0.1
   ```

2. **설정 파일 수정**:
   ```yaml
   env_args:
     window_size_x: 0
     window_size_y: 0
     debug: False
   ```

3. **명령행에서 직접 설정**:
   ```bash
   python src/main.py --config=qmix --env-config=sc2 \
     env_args.map_name=3m \
     env_args.window_size_x=0 \
     env_args.window_size_y=0 \
     render=False
   ```

### 6. 테스트 시에만 화면 표시
테스트할 때만 화면을 보고 싶다면:

```bash
python src/main.py --config=qmix --env-config=sc2 \
  env_args.map_name=3m \
  evaluate=True \
  render=True
```

이제 학습할 때 StarCraft II 게임 화면이 나타나지 않습니다! 🎉
