import numpy as np


def svm_loss_vectorized(W, X, y, reg, delta=1.0):
    """
    다중 클래스 SVM Loss 및 Gradient(dW)의 벡터화 구현
    
    Inputs:
    - W: (D, C) 크기의 가중치 행렬
    - X: (N, D) 크기의 입력 데이터 (N개 샘플, D개 피처)
    - y: (N,) 크기의 정답 레이블 (0 <= y[i] < C)
    - reg: Regularization 강도 (lambda)
    - delta: 안전 마진 (기본값 1.0)
    
    Returns:
    - loss: 단일 스칼라 (Data Loss + Reg Loss)
    - dW: W에 대한 손실 함수의 기울기 (D, C)
    """
    num_train = X.shape[0]

    # 1. Forward Pass: Scores 계산 (N, C)
    scores = X.dot(W)

    # 2. 각 샘플의 정답 클래스 점수 추출 (N, 1)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)

    # 3. Margin 계산: max(0, S_j - S_yi + delta)
    margins = np.maximum(0, scores - correct_class_scores + delta)

    # 4. 정답 위치(j = y_i)의 margin은 0으로 지정 (Off-by-one 버그 방지)
    margins[np.arange(num_train), y] = 0.0

    # 5. Total Loss 계산 (Data Loss + L2 Reg Loss)
    data_loss = np.sum(margins) / num_train
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # -------------------------------------------------------------
    # 6. Backward Pass: Gradient (dW) 연산
    # -------------------------------------------------------------
    # 마진이 0보다 큰(위반이 발생한) 위치 개수 카운트
    binary_margins = np.zeros_like(margins)
    binary_margins[margins > 0] = 1

    # 오답 클래스에 빼앗긴 마진 합산 횟수만큼 정답 클래스 위치에서 감산
    row_sum = np.sum(binary_margins, axis=1)  # (N,)
    binary_margins[np.arange(num_train), y] = -row_sum

    # dW 연산: X.T (D, N) dot binary_margins (N, C) -> (D, C)
    dW = X.T.dot(binary_margins)

    # N으로 나누고 L2 Regularization의 미분항(reg * W) 추가
    dW /= num_train
    dW += reg * W

    return loss, dW


# =====================================================================
# 🧪 실제 작동 테스트
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 5개 샘플, 10개 피처, 3개 클래스 가짜 데이터 생성
    N, D, C = 5, 10, 3
    X = np.random.randn(N, D)
    W = np.random.randn(D, C) * 0.001
    y = np.random.randint(0, C, size=N)
    reg = 0.1

    # Loss 및 dW 계산
    loss, dW = svm_loss_vectorized(W, X, y, reg)

    print("=== Multiclass SVM Loss 테스트 ===")
    print(f"Calculated Loss: {loss:.4f}")
    print(f"dW Matrix Shape: {dW.shape}")