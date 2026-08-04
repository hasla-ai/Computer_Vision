import numpy as np


# 1. Softmax Loss 및 Gradient 계산 함수
def softmax_loss_vectorized(W, X, y, reg):
    """Softmax (Cross-Entropy) Loss 및 Analytical Gradient (dW) 구현"""
    num_train = X.shape[0]

    # Forward Pass: Scores -> Probabilities -> Loss
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  # Numeric Stability

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    data_loss = np.sum(correct_logprobs) / num_train
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # Backward Pass: Gradient (dW)
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1.0

    dW = X.T.dot(dscores)
    dW /= num_train
    dW += reg * W

    return loss, dW


# 2. 수치 미분(Numerical Gradient) 평가 함수
def eval_numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        # f(x + h)
        x[idx] = old_val + h
        fxph = f(x)

        # f(x - h)
        x[idx] = old_val - h
        fxmh = f(x)

        # 복원
        x[idx] = old_val

        # 중앙 차분 공식
        grad[idx] = (fxph - fxmh) / (2 * h)
        it.iternext()

    return grad


# 3. 상대 오차 계산 함수
def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# =====================================================================
# 🧪 Gradient Check 메인 실행 루틴
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 검증용 테스트 데이터 생성
    N, D, C = 5, 10, 3
    X = np.random.randn(N, D)
    W = np.random.randn(D, C) * 0.001
    y = np.random.randint(0, C, size=N)
    reg = 0.1

    # 1) 해석적 기울기 (Analytical Gradient) 계산
    loss, dW_analytical = softmax_loss_vectorized(W, X, y, reg)

    # 2) 수치 미분 (Numerical Gradient) 계산
    f = lambda w: softmax_loss_vectorized(w, X, y, reg)[0]
    dW_numerical = eval_numerical_gradient(f, W)

    # 3) 상대 오차 검증
    error = rel_error(dW_analytical, dW_numerical)

    print("=== Softmax Loss Gradient Check 결과 ===")
    print(f"Calculated Loss              : {loss:.4f}")
    print(f"Analytical Gradient (첫 3개) : {dW_analytical.flat[:3]}")
    print(f"Numerical Gradient  (첫 3개) : {dW_numerical.flat[:3]}")
    print(f"Relative Error (상대 오차)   : {error:.10e}")

    if error < 1e-6:
        print("\n✅ SUCCESS: Softmax dW 기울기 코드가 완벽하게 정답입니다!")
    else:
        print("\n❌ FAIL: dW 코드에 버그가 존재합니다.")