## 프로젝트 목표

사용자의 본인 여부 판단을 위해 회원가입 시 인증사진과 프로필 사진을 비교하는 기존 얼굴인식 AI 기능 개선을 목표로 해당 기업 프로젝트를 진행하였습니다.
(사진 도용같은 문제를 해결)


## 팀프로젝트 역활 분담
김경수 :데이터 전처리, Arcface 모델 학습 및 구현

김영관 : 데이터 전처리, Arcface 모델 학습 및 구현 및 성능 테스트

염정헌: 데이터 전처리, 모델별 성능 확인, facenet 모델 학습

## 데이터 수집 
인증 사진과 프로필 사진이 같은 사람인지에 대한 정확성을 판단하기 위한 이미지 데이터가 필요하였습니다. 동양인으로써 각 사람에 대한 다양한 각도의 대량 이미지 데이터를 수집하는 것이 중요하다고 판단하였기에 아래의 데이터를 학습 데이터로 선정하였습니다.

가족관계가 알려진 얼굴 이미지 데이터 
소개
총 1,000 가계 (3000명)이상을 대상으로 80,700장(가족사진 : 6,900, 각도별
개인 사진 : 49,800, 기간별 나이 사진 : 24,000) 이상의 가족 관계 얼굴 이미지 데이터셋을 구축

## 모델 선택을 위한 모델 평가 
얼굴 인식 모델 선정을 위해 소량의 이미지 데이터셋을 가지고 각 모델에 인증 정확도와 인식 속도를 비교해 보았습니다. (우측 사진 참고)
테스트 결과 VGG-Face모델이 가장 높은 정확도를 보이지만 모델 크기가 530MB로써 계산 복잡도가 높아 속도가 느리다는 단점이 있습니다.
핸드폰 디바이스에서 작동되어 진다는 점에서 모델 크기가 작고(245MB) 속도가 빠르며 상대적으로 높은 정확성을 가지면서 대규모 얼굴 인식 시스템에 적합한 ArcFace모델을 1순위로 선정하는 것이 옳다고 판단하였고 2순위로 Facenet모델을 선정하였습니다.

## 전처리 과정

1. mediapipe를 이용하여 얼굴검출
2. mediapipe를 이용하여 얼굴 랜드마크 형성 후 양쪽눈의 좌표를 이용하여 얼굴 정렬
3. 그레이 스케일링
4. 리사이즈 (144,144)

## 모델 구현
1. 사진의 URL을 통해 이미지파일로 변화
2. 파일에 전처리 과정 진행
3. 학습한 가중치 파일을 통해 모델 로드
4. 특징정보를 가진 임베딩 벡터를 생성 후 유사도를 측정
5. 임계값을 기준으로 동일인물 여부를 판별

## 결과

전처리 이전 데이터

Accuracy 0.627

전처리 이후 데이터

Accuracy 0.947  0.25s

## 회고 

# 잘 한 점, 만족한 점은 무엇인가요? 

ArcFace모델을 이용하여 얼굴을 인식하여 본인인증 서비스를 완성하였다.

논문 구현하는 방법에 대해 자세히 알게되어서 좋았다.

# 아쉬운 점, 한계점은 무엇인가요? 

시간분배를 너무 잘못하여 초반에 무의미하게 시간을 보냈다. 그리고 팀에서 역활분배를 직렬적으로 진행하여 진행이 더뎠다

# 다시 한다면 어떤 점을 개선하고 싶은가요? 

멀리서 찍은 사람의 사진을 mediapipe에서 얼굴감지 부분에서 얼굴을 감지를 못하여 사진을 그냥 처리한 사진들이 많이 존재했다. 그래서 얼굴감지 부분을 다시 만들어 멀리 존재하는 사람의 얼굴도 감지할 수 있도록 수정이 필요할 것 같다.
