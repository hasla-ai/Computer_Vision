import numpy as np

# -------------------------------------------------------------------
# 1. 가상의 이미지 데이터 생성 (N=2장, H=32, W=32, C=3)
# 픽셀값은 0~255 사이의 uint8 정수 타입
# -------------------------------------------------------------------
np.random.seed(42)
images_hwc = np.random.randint(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)

print(f"[입력] Original Batch Shape (HWC): {images_hwc.shape}")


# -------------------------------------------------------------------
# 2. HWC -> CHW 변환 및 Data Type Normalization
# 딥러닝 연산을 위해 0~255 정수를 0.0~1.0 float32로 정규화합니다.
# -------------------------------------------------------------------
def preprocess_images(img_batch):
    # (N, H, W, C) -> (N, C, H, W) 축 변경
    img_chw = np.transpose(img_batch, (0, 3, 1, 2))
    
    # 0.0 ~ 1.0 스케일링 (float32 변환)
    img_normalized = img_chw.astype(np.float32) / 255.0
    return img_normalized

images_chw = preprocess_images(images_hwc)
print(f"[변환] PyTorch Style Shape (CHW): {images_chw.shape}")
print(f"[정규화] Min pixel: {images_chw.min():.1f}, Max pixel: {images_chw.max():.1f}")


# -------------------------------------------------------------------
# 3. Pure Numpy L1 / L2 Distance 계산 함수
# -------------------------------------------------------------------
def compute_l1_distance(img1, img2):
    """
    img1, img2: (C, H, W) 형태의 numpy array
    """
    # np.abs(): 픽셀별 절댓값 차이
    # np.sum(): 모든 픽셀(C*H*W)의 차이를 하나로 합산
    l1_dist = np.sum(np.abs(img1 - img2))
    return l1_dist

def compute_l2_distance(img1, img2):
    """
    img1, img2: (C, H, W) 형태의 numpy array
    """
    # np.square(): 픽셀별 차이의 제곱
    # np.sqrt(): 전체 합에 대한 제곱근
    l2_dist = np.sqrt(np.sum(np.square(img1 - img2)))
    return l2_dist


# 첫 번째 이미지와 두 번째 이미지 비교
img_A = images_chw[0]
img_B = images_chw[1]

l1_result = compute_l1_distance(img_A, img_B)
l2_result = compute_l2_distance(img_A, img_B)

print("\n--- 거리 계산 결과 ---")
print(f"L1 Distance (Manhattan): {l1_result:.4f}")
print(f"L2 Distance (Euclidean): {l2_result:.4f}")