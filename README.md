[1단계] CV 코어 & 딥러닝 원리 (Numpy 백엔드)
1주차: 이미지 데이터 구조 & Linear Classifier
개념: 픽셀 행렬 표현, Loss Function(Multiclass SVM Loss, Cross-Entropy Loss).
1-1회차: 이미지 픽셀 행렬(RGB, Channel-First vs Channel-Last) 및 L1/L2 거리 계산 코딩
https://blog.naver.com/revised_crybaby/224366944264

1-2회차: Linear Score Function 및 Multiclass SVM Loss 함수 Pure Numpy 구현
1-3회차: Softmax Loss (Cross-Entropy) 및 Data Loss + Regularization Loss 통합 계산기 작성
Loss Function(Multiclass SVM Loss, Cross-Entropy Loss).
-> CIFAR-10 데이터셋 로드 후 Linear Classifier 전방 전달(Forward Pass) 코딩.

2주차: 역전파(Backpropagation) 엔진 구현
개념: Computational Graph, Chain Rule, Analytic Gradient vs Numerical Gradient.
2-1회차: Computational Graph 구축 및 Simple Adder/Multiplier 역전파 Chain Rule 계산
2-2회차: 2-Layer Neural Network의 Forward & Backward Pass 함수 Pure Numpy 작성
2-3회차: Analytic Gradient와 Numerical Gradient 비교 검증(Gradient Check) 모듈 구현

3주차: Convolution & Pooling Layer 바닥부터 만들기
개념: Spatial Convolution, Stride, Padding, Receptive Field, Max Pooling.
3-1회차: 2D Spatial Convolution 동작 원리 및 Stride, Padding 수식 구현
3-2회차: Numpy im2col 기법을 활용한 Fast Matrix Convolution 연산기 작성
3-3회차: Max Pooling & Average Pooling Layer의 Forward/Backward 계산 구현
Numpy의 im2col 기법을 활용하여 Fast Convolution Layer 구현하기.

4주차: 최적화(Optimization) & 학습 루프
개념: SGD+Momentum, RMSprop, Adam, Batch Normalization, Dropout.
4-1회차: SGD+Momentum, RMSprop, Adam Optimizer 알고리즘 수식 및 코드 작성
4-2회차: Batch Normalization 및 Dropout Layer의 Train/Test 모드 동작 구현
4-3회차: Mini-batch 분할, Epoch iteration, Loss 시각화를 포함한 Full Training Loop 작성
Adam Optimizer와 Batch Normalization을 pure Numpy로 작성.
학습 루프(Epoch, Mini-batch iteration, Validation evaluation) 완제 작성.

[2단계] OpenCV 전처리 & Hugging Face PyTorch 모델링
5주차: OpenCV 영상 처리 & 파이프라인
개념: OpenCV 기반 전처리 및 전통적 영상 처리 이미지 색상 공간(BGR/HSV/RGB), Geometric Transformation, Filtering, Edge Detection.
5-1회차: cv2.imread, BGR/HSV 색상 공간 변환 및 cv2.inRange를 이용한 마스킹
5-2회차: Canny Edge Detection 및 cv2.findContours 기법 기반 경계선 추출
5-3회차: cv2.getPerspectiveTransform 기법 활용 문서/이미지 정방향 펴기(Rectification)
과제 [OpenCV]:
OpenCV 기본: cv2.imread, cv2.cvtColor, cv2.Canny, cv2.findContours를 사용해 이미지 내 특정 색상의 도형/물체 외곽선 검출 파이프라인 작성.
이미지 변환: Perspective Transform(cv2.getPerspectiveTransform)을 이용해 삐뚤어진 문서 이미지를 평평하게 펴는 Rectification 알고리즘 작성.

6주차: Hugging Face로 ResNet 다루기 & 전이학습
ResNet 파헤치기 & PyTorch 전이학습
개념: Vanishing Gradient Problem, Residual Connection, Fine-tuning.
6-1회차: transformers.AutoImageProcessor 및 AutoModelForImageClassification 이해
6-2회차: Hugging Face Hub에서 사전 학습된 microsoft/resnet-50 불러오기 및 추론
6-3회차: PyTorch Trainer 또는 Custom Loop로 사전 학습 모델 전이학습(Fine-tuning)
과제:
PyTorch로 ResNet-18 구조(Residual Block) 직접 구현해보기.
ImageNet으로 사전 학습된 모델(Pretrained Weights)을 가져와 커스텀 분류 데이터셋으로 전이학습(Transfer Learning) 실행.

[3단계] YOLO & SAM & ViT (Hugging Face / Ultralytics)
7주차: Object Detection (YOLOv8 & Hugging Face DETR)
7주차: Ultralytics YOLO 기반 실시간 Object Detection
개념: Bounding Box Regression, IoU (Intersection over Union), NMS (Non-Maximum Suppression), Anchor Box.
7-1회차: Bounding Box 포맷(XYXY, XYWH, YOLO format) 및 IoU(Intersection over Union) 구현
7-2회차: NMS(Non-Maximum Suppression) 알고리즘 직접 구현해보기
7-3회차: Ultralytics YOLOv8 모델로 동영상 내 객체 탐지 및 Bounding Box 시각화
7-4회차: Hugging Face facebook/detr-resnet-50을 활용한 Transformer 기반 객체 탐지 실습
과제 [YOLO]:
IoU & NMS 구현: Numpy/PyTorch로 IoU 계산 함수와 NMS 알고리즘 함수 직접 코딩해보기.
YOLOv8 파이프라인: ultralytics 라이브러리를 사용해 커스텀 데이터셋(예: 도로 위 차량/보행자)의 YOLO 포맷 라벨링 데이터 작성 후 학습(Fine-tuning) 및 cv2 영상 위에 Real-time Bounding Box 시각화.

8주차: Vision Transformer (ViT) 파헤치기
주차: Vision Transformer (ViT) & Patch Extraction
개념: Image Patches to Tokens, Position Embedding, Multi-Head Self-Attention in Vision.
8-1회차: 이미지를 16x16 Patch로 쪼개어 Linear Projection 및 Class Token 추가하는 코드 작성
8-2회차: Hugging Face google/vit-base-patch16-224 모델 구조 분석 및 Feature Extractor 실습
8-3회차: ViT의 Self-Attention Map을 추출하여 모델이 이미지의 어느 부위를 주시하는지 시각화
과제:
입력 이미지를 16x16 패치로 분할하여 선형 투영(Linear Projection)하는 층을 PyTorch nn.Conv2d 또는 einops로 구현하기.
간이 Vision Transformer(ViT) 모델 구조를 작성하고 CIFAR-10 학습시키기.

9주차: Meta SAM (Segment Anything) & Hugging Face
9주차: Meta SAM (Segment Anything) 기반 영구 세그멘테이션
개념: Promptable Segmentation, Image Encoder (ViT), Prompt Encoder, Mask Decoder.
9-1회차: Hugging Face facebook/sam-vit-base 모델 및 SamProcessor 로드
9-2회차: Point Prompt(마우스 클릭 좌표) 및 Box Prompt를 이용한 정밀 Mask 추출 실습
9-3회차: OpenCV 객체 탐지 결과(Bounding Box)를 SAM의 Prompt로 자동 전달하는 하이브리드 파이프라인 구축
과제 [SAM]:
segment-anything 또는 ultralytics SAM 2 라이브러리를 설치하고, 이미지 내 마우스 좌표(Point Prompt) 또는 Bounding Box를 기반으로 마스크(Mask)를 추출하는 파이프라인 작성.
OpenCV + SAM 연동: OpenCV로 검출한 물체 위치(Bounding Box)를 SAM의 Prompt 입력으로 전달하여 사물의 섬세한 경계(Polygon)를 자동 생성하는 하이브리드 자동 라벨러 작성.

[4단계] 멀티모달 파운데이션 & 통합 프로젝트
10주차: CLIP + SAM + LLM 비전 RAG 통합 파이프라인
10주차: CLIP + YOLO + SAM + LLM 통합 멀티모달 파이프라인
개념: Cross-Modal Alignment (Text-Image Joint Embedding), Zero-shot Visual Question Answering, Vision RAG.

10-1회차: CLIP 기반 Text-Image Zero-Shot 분류
세부 내용: Hugging Face openai/clip-vit-base-patch32 모델 및 CLIPProcessor 로드
실습: 텍스트 쿼리(예: "a photo of a dog")와 이미지 간의 Cosine Similarity를 계산하여 별도 학습 없이 이미지 분류하기

10-2회차: 크롭/세그멘테이션 및 CLIP 임베딩 DB화
세부 내용: YOLO/SAM으로 이미지 내 개별 물체들을 잘라내고(Crop), 각각을 CLIP Image Encoder에 통과시켜 벡터 DB(또는 Numpy 배열)에 저장하기
실습: "빨간색 차"라고 검색했을 때, 전체 이미지 중 해당 영역만 정확히 찾아내는 Visual Search 모듈 작성

10-3회차: Vision LLM(LLaVA/Qwen-VL) 연동 및 통합 비전 QA 시스템
세부 내용: Hugging Face의 llava-hf/llava-1.5-7b-hf 또는 경량 멀티모달 LLM 로드
실습: [OpenCV 전처리 → YOLO/SAM 객체 추출 → CLIP 검색 → Vision LLM 요약]으로 이어지는 End-to-End 멀티모달 비전 RAG 파이프라인 완성 및 결과 보고서 생성

과제 [통합 프로젝트]:
과제 목표: "이미지/동영상 질의응답 비전 파이프라인 구축"
세부 단계:
YOLO/SAM: 동영상 프레임에서 주요 객체를 YOLO로 탐지하고 SAM으로 정밀 마스킹/크롭.
CLIP: 크롭된 이미지 영역을 CLIP Image Encoder로 넘기고, 사용자 텍스트 쿼리와의 유사도(Cosine Similarity) 측정.
LLM: 탐지된 객체 정보와 수치화된 메타데이터를 정돈하여 LLM 프롬프트에 전달, 최종 분석 리포트 생성.
