import numpy as np

def compute_distances_no_loops(X_train, X_test):
    """
    for문 없이 Pure Numpy 행렬 연산만으로 L2 Distance Matrix를 계산합니다.

    Inputs:
    - X_train: (M, D) 형태의 numpy array (M개의 트레이닝 이미지, D차원)
    - X_test:  (N, D) 형태의 numpy array (N개의 테스트 이미지, D차원)

    Returns:
    - dists: (N, M) 형태의 numpy array
             dists[i, j]는 i번째 테스트 이미지와 j번째 트레이닝 이미지 간의 L2 거리
    """
    # 1. 테스트 데이터의 각 행(이미지)별 제곱합: (N, D) -> (N,) -> (N, 1) 차원 변경
    test_sum_sq = np.sum(np.square(X_test), axis=1, keepdims=True)  # Shape: (N, 1)

    # 2. 트레이닝 데이터의 각 행(이미지)별 제곱합: (M, D) -> (M,) -> (1, M) 차원 변경
    train_sum_sq = np.sum(np.square(X_train), axis=1, keepdims=True).T # Shape: (1, M)

    # 3. 행렬곱을 이용한 -2ab 계산: (N, D) @ (D, M) -> (N, M)
    # BLAS / SIMD 레벨에서 지극히 빠르게 처리됨
    cross_term = -2 * np.dot(X_test, X_train.T)  # Shape: (N, M)

    # 4. (a - b)^2 = a^2 + b^2 - 2ab 적용 (Broadcasting에 의해 (N, M) 크기로 자동 합성)
    dists_squared = test_sum_sq + train_sum_sq + cross_term

    # 부동소수점 오차로 인해 아주 미세한 음수(-0.00000001 등)가 나오는 것을 방지 (np.maximum)
    dists_squared = np.maximum(dists_squared, 0.0)

    # 5. 제곱근(Sqrt)을 씌워 최종 L2 Distance 완성
    dists = np.sqrt(dists_squared)

    return dists


# =====================================================================
# 🧪 검증 코드: 2중 for문 결과와 no_loops 결과가 완벽히 일치하는지 확인
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # N=50개의 테스트 이미지, M=100개의 트레이닝 이미지, D=3072 차원 (32x32x3)
    N, M, D = 50, 100, 3072
    X_test = np.random.randn(N, D)
    X_train = np.random.randn(M, D)

    # 1) 백엔드 가속 no_loops 함수 실행
    dists_no_loops = compute_distances_no_loops(X_train, X_test)

    # 2) 검증용 2중 for문 함수
    dists_two_loops = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dists_two_loops[i, j] = np.sqrt(np.sum(np.square(X_test[i] - X_train[j])))

    # 3) 두 방식의 차이(Difference) 계산
    difference = np.linalg.norm(dists_no_loops - dists_two_loops, ord='fro')

    print(f"Distance Matrix Shape: {dists_no_loops.shape}")
    print(f"Two-loop 방식과의 오차(Difference): {difference:.10f}")

    if difference < 1e-4:
        print("✅ 성공! for문 없이 완벽하게 동일한 L2 거리 행렬을 계산했습니다.")
    else:
        print("❌ 실패! 수식에 오차가 존재합니다.")