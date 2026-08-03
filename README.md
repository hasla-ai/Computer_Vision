CS231n의 최신 공개 강의(Stanford CS231n: Deep Learning for Computer Vision) 기반
페이페이 리(Fei-Fei Li) 교수와 저스틴 존슨(Justin Johnson), 안드레이 카파시(Andrej Karpathy) 등이 정립한 스탠퍼드 대학의 시그니처 강의

 - 머신러닝 기초 이론부터 시작해 딥러닝 코어 엔진을 C++이나 PyTorch 없이 Numpy만으로 직접 작성(From Scratch)해보고, 현대 멀티모달(ViT, Diffusion)로 확장하는 로드맵

 기존 10주 차 주차별 과제 파트에 OpenCV, YOLO, SAM 실습 과제를 단계별로 배치

 - 가장 거대한 프로젝트 3개: ① OpenCV (Open Source Computer Vision Library) - 엔터프라이즈 가치: 자율주행, 로보틱스, CCTV 모니터링 등 실시간 엣지 디바이스(Edge Device) 상에서 '딥러닝 전후처리'를 신속하게 처리하기 위해 모든 실무 인프라에 반드시 포함되는 필수의 레이어

# 딥러닝 이전의 전통적인 영상 처리(Edge Detection, Filtering, Matrix Transformation, Geometric 3D)

# 엣지 디바이스(Edge Device) 상에서 '딥러닝 전후처리'

​

 - ② Ultralytics YOLO (You Only Look Once): 엔터프라이즈 가치: 공장 불량품 검사(Vision AI Inspections), 매장 내 행동 분석, 자율주행 차선/보행자 감지 등 "가장 빠르게 엔드투엔드로 서비스화할 수 있는 Vision ML 프로젝트"의 글로벌 표준 _ 제조

# 실시간 객체 탐지(Real-Time Object Detection) #객체의 위치(Bounding Box) #공장 불량품 검사(Vision AI Inspections)

​

 - ③ Meta Segment Anything Model (SAM / SAM 2): Promptable Visual Segmentation. 엔터프라이즈 가치: 멀티모달 LLM과 연동할 때 비주얼 레이아웃 분석, 의료 영상 자율 라벨링, 비디오 편집 자동화 등 '이미지를 구조화된 데이터로 빠르게 변환'하는 핵심 엔진

​

CS231n으로 비전의 입출력 구조와 특성 추출(Feature Extraction)의 수학적 원리를 익힌 뒤, OpenCV → YOLO → SAM으로 이어지는 실제 프레임워크.

Hugging Face ecosystem을 메인 프레임워크로 적극 채택하고, 혼자 공부하는 분이 하루에 1~2시간씩 '세부 차시(1회차, 2회차...)' 단위로 끊어서 독학.