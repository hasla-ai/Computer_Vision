# [가장 직관적이고 정석적인 L2 거리 정답 코드]
dists_two_loops = np.zeros((N, M))
for i in range(N):          # 모든 테스트 이미지 i에 대해
    for j in range(M):      # 모든 트레이닝 이미지 j에 대해
        # i번째와 j번째 이미지 간의 픽셀 차이 제곱합의 루트 = L2 거리
        dists_two_loops[i, j] = np.sqrt(np.sum(np.square(X_test[i] - X_train[j])))