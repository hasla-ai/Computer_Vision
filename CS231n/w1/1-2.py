import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    2중 for문을 사용한 직관적인 Multiclass SVM Loss
    - W: (D, C)
    - X: (N, D)
    - y: (N,) 정답 클래스 인덱스 레이블 (예: [0, 3, 1, ...])
    - reg: L2 정규화 강도 (Hyperparameter)
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)               # (C,)
        correct_class_score = scores[y[i]] # 정답 클래스 점수
        
        for j in range(num_classes):
            if j == y[i]:
                continue # 정답 클래스는 비교 대상에서 제외
            
            margin = scores[j] - correct_class_score + 1.0 # Delta = 1.0
            if margin > 0:
                loss += margin

    # 배치 평균 손실
    loss /= num_train

    # L2 Regularization 추가 (Overfitting 방지)
    loss += 0.5 * reg * np.sum(W * W)

    return loss


def svm_loss_vectorized(W, X, y, reg):
    """
    for문 없이 초고속으로 계산하는 Vectorized Multiclass SVM Loss
    """
    num_train = X.shape[0]
    
    # 1. Score Matrix 계산: (N, D) @ (D, C) -> (N, C)
    scores = X.dot(W)
    
    # 2. 각 이미지별 정답 클래스의 점수만 추출: (N, 1)
    # np.arange(num_train)과 y를 이용해 행/열 인덱싱 수행
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    
    # 3. Margin Matrix 계산 (Broadcasting 활용)
    # (N, C) - (N, 1) + 1.0  -> (N, C)
    margins = np.maximum(0, scores - correct_class_scores + 1.0)
    
    # 4. 정답 클래스 위치(j == y_i)의 마진은 0으로 무효화
    margins[np.arange(num_train), y] = 0
    
    # 5. 전체 마진의 합으로 Loss 계산 후 배치 평균
    loss = np.sum(margins) / num_train
    
    # 6. L2 Regularization 손실 추가
    loss += 0.5 * reg * np.sum(W * W)
    
    return loss


# =====================================================================
# 🧪 검증 코드: Naive 방식과 Vectorized 방식의 결과 일치 여부 확인
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # N=500개 이미지, D=3072 차원, C=10개 클래스
    N, D, C = 500, 3072, 10
    X = np.random.randn(N, D)
    W = np.random.randn(D, C) * 0.001
    y = np.random.randint(0, C, size=N)
    reg = 0.1

    loss_naive = svm_loss_naive(W, X, y, reg)
    loss_vec = svm_loss_vectorized(W, X, y, reg)

    print(f"Naive Loss     : {loss_naive:.6f}")
    print(f"Vectorized Loss: {loss_vec:.6f}")
    print(f"두 방식 오차    : {abs(loss_naive - loss_vec):.10f}")
    
    if abs(loss_naive - loss_vec) < 1e-6:
        print("✅ 성공! Vectorized SVM Loss가 완벽히 동작합니다.")