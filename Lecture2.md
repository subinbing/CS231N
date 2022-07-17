[CS231N] Lecture2 정리 >> Image Classification pipeline

CS231n: 시각적 인식을 위한 컨볼루션 신경망

http://cs231n.stanford.edu/

스탠포드 대학교 강의





Image Classification

- Core Task에 속함

ex) 고양이 사진 (입력 이미지)

<img width="316" alt="image" src="https://user-images.githubusercontent.com/87634136/179386578-48e8ec83-4499-4acf-932c-8b9daa8e7e47.png">

- 시스템 -> 카테고리 집합 있음

카테고리 : 개, 고양이, 트럭, 비행기 등

-> 컴퓨터가 해야 할 일은 이미지를 보고 어떤 카테고리에 속할지 고르는 것

우리 눈에는 고양이로 보이지만 컴퓨터는 그렇지 못함

<img width="322" alt="image" src="https://user-images.githubusercontent.com/87634136/179386582-d73e2689-17fc-4970-a014-24e5c8ed7a8c.png">

컴퓨터는 고양이가 아주 큰 격자 모양의 숫자 집합으로 보임

cf. 800x600 이미지

각 픽셀 : 세 개의 숫자로 표현 (red, green, blue)



-> 컴퓨터에게 이미지 : 단지 거대한 숫자 집합에 불과함

but 거대한 숫자 집합에서 "고양이"를 인식하는 것은 어려움

Because 의미론적인 차이 (Semantic gap)

"고양이"라는 레이블 -> 우리가 이미지에 붙인 의미상의 레이블



고양이 이미지가 고양이 사진이라는 사실과 실제 컴퓨터가 보는 픽셀 값과는 큰 차이가 있음



If

고양이 이미지에 아주 미묘한 변화를 주면 픽셀 값들은 모조리 변하게 될 것임

ex) 고양이 한마리가 가만히 앉아 있으면 아무 일도 일어나지 않음

but 카메라를 아주 조금만 옆으로 옮겨도 모든 픽셀 값들이 모조리 달라짐

but 달라진 픽셀 값은 고양이라는 사실이 바뀌지는 않음



바라보는 방향 뿐만 아니라 "조명" 또한 문제가 될 수 있음

-> 어떤 장면이냐에 따라서 조명은 달라짐

(고양이가 어두운 곳에 있던 밝은 곳에 있던 고양이는 고양이임 -> 알고리즘은 이것에 강해야함)

<img width="326" alt="image" src="https://user-images.githubusercontent.com/87634136/179386603-01e3b981-eb5e-4ba5-a94e-76c4e9aa641b.png">

객체 자체에 변형이 있을 수 있음

- 고양이의 자세

<img width="335" alt="image" src="https://user-images.githubusercontent.com/87634136/179386608-972f7d33-0c50-4339-b917-cbc7d1712cb3.png">

가려짐 (occlusion)의 문제

- 고양이의 일부밖에 볼 수 없는 상황이 있을 수 있음

- 고양이의 얼굴만 볼 수 있는 경우

- 소파에 숨어 들어간 고양이의 꼬리부분

<img width="422" alt="image" src="https://user-images.githubusercontent.com/87634136/179386618-501b37ab-958e-4437-8928-e88a1589829c.png">

배경과 비숫한 경우 (Background clutter)

- 고양이 배경과 거의 비슷하게 생긴 경우

<img width="320" alt="image" src="https://user-images.githubusercontent.com/87634136/179386624-a3fd49f1-7c56-4fca-9085-da582a0b0ab1.png">

Intraclass variation

- 고양이에 따라 생김새, 크기, 색, 나이가 각양 각색

<img width="359" alt="image" src="https://user-images.githubusercontent.com/87634136/179386628-5ce5f7ef-bcd4-498f-aab0-62a9db6c5c43.png">



Image Classifier API 코드 (함수 하나)

- Python 메서드를 작성

- 이미지 입력을 받아서, 어떤 "놀라운 마법"이 일어나고, 이 이미지가 "고양이다", "강아지이다" 라고 말해주는 것

​

고양이를 인식하고자 할 때

<img width="322" alt="image" src="https://user-images.githubusercontent.com/87634136/179386636-9f400514-eb2c-4ba7-8827-559bb7115562.png">

고양이는 두 개의 귀와 하나의 코가 있음

Edges 중요

1. 이미지에서 edges를 계산

2. 다양한 Corners와 Edges를 각 카테고리로 분류

(세 개의 선이 만나는 지점이면 corner라고 가정)

-> 귀는 "여기에 corner 하나" ,"저기에도 corner 하나", "저기에도 코나 하나" 가 있고

(고양이 인식을 위해 "명시적인 규칙 집합"을 써내려 가는 방법)



but 잘 동작하지 않음

Because1 > 알고리즘이 강인하지 못함

Because2 > 또 다른 객체를 인식해야 한다면, 트럭에 대해서도, 개에 대해서도 별도로 만들어야함

(즉, 처음부터 다시 시작해야함 --> 확장성 전혀 없는 방법)



So, 확장성 있는 알고리즘 만들어야 함

-> 데이터 중심 접근 방법 (Data-Driven Approach)

* 데이터 중심 접근 방법 중 ... (기능)

-> Global Function Approximater (이미지가 무엇인지 인식)



ex) 고양이는 무엇이다 하면서 손으로 직접 규칙을 써내려 하는 것 대신에

인터넷에 접속 -> 많은 고양이/비행기/사슴 데이터 수집



각 카테고리에서 다양한 많은 데이터들을 모으기 위해 -> Google Image Search 도구 이용



Machine Learning

- 방대한 데이터를 수집하려면 많은 시간과 노력 필요

-> 손쉽게 이용할 수 있는 방법이 "고퀄리티의 데이터셋"이 있음

(이 데이터 셋을 통해 Machine Learning Classifier를 학습)

- 데이터를 잘 요약해서 다양한 객체들을 인식할 수 있는 모델을 만들어냄 -> 고양이나 개를 잘 인식함



API 수정 (입력 이미지를 고양이로 인식 -> 2개의 함수(1개의 함수X))

- 함수1 >> Train 함수

입력은 이미지와 레이블, 출력은 우리의 모델

- 함수2 >> Predict 함수

입력이 모델, 출력은 이미지의 예측값



Nearest Neighbor (단순한 Classifier)

- Train Step

아무 일도 하지 않음

모든 학습 데이터를 기억

- Pridict Step

새로운 이미지가 들어오면 새로운 이미지와 기존의 학습 데이터를 비교해서 가장 유사한 이미지로 레이블링을 예측

아주 간단

but Data-driven Approach로서 아주 좋은 알고리즘



CIFAR-10 데이터셋

<img width="332" alt="image" src="https://user-images.githubusercontent.com/87634136/179386668-6bf1af4f-94d0-4513-ba67-34f93e418551.png">

- Machine Learning에서 자주 쓰는 연습용(테스트용) 데이터 셋

- 10가지 클래스 있음 (비행기, 자동차, 새, 고양이 등)

- 10가지 카테고리, 총 50,000여개의 학습용 이미지

- 50,000여개의 데이터는 각 카테고리에 균일하게 분포함

- 알고리즘 테스트 이미지 : 10,000

- 오른쪽 칸의 맨 왼쪽 열 : CIFAR-10 테스트 이미지

오른쪽 방향으로는 학습 이미지 중 테스트 이미지와 유사한 순으로 정렬

​

--> 이미지 쌍이 있을 때 어떻게 비교 할 것인가?

: "어떤 비교 함수를 사용할지"



>> L1 Distance (Manhattan distance)

- 아주 간단한 방법

<img width="326" alt="image" src="https://user-images.githubusercontent.com/87634136/179386679-5a4a62ef-7cdb-4456-af80-943eacb65791.png">

-> 이미지를 Pixel-wise로 비교

ex) 4x4 테스트 이미지

테스트/트레이닝 이미지의 같은 자리의 픽셀을 서로 빼고 절댓값을 취함

픽셀 간의 차이 값을 계산하고 모든 픽셀의 수행 결과를 모두 더함



"두 이미지 간의 차이를 어떻게 측정 할 것인가?"

위의 그림에서는 두 이미지 간에 "456"만큼 차이가 남



NN Classifier를 구현한 Python 코드

- 매우 짧고 간결

(Numpy에서 제공하는 Vectorized operations를 이용했기 때문)

<img width="350" alt="image" src="https://user-images.githubusercontent.com/87634136/179386687-05710369-2bae-488e-ad80-d955ee7f6c3e.png">

<img width="332" alt="image" src="https://user-images.githubusercontent.com/87634136/179386696-633a3a53-d54b-434f-b94d-4da70d79ae31.png">

<img width="333" alt="image" src="https://user-images.githubusercontent.com/87634136/179386702-56feae01-3fc5-420b-a588-5004a8a46a30.png">

- Train 함수

: 학습 데이터를 기억하는 것 (할 일이 크게 없음)

- Test 함수

: 이미지를 입력으로 받고 L1 Distance로 비교

: 학습 데이터들 중 테스트 이미지와 가장 유사한 이미지들을 찾아냄



=> Numpy의 Vectorized opeerations를 활용하면 구현은 Python code 1~2줄이면 충분



Simple Classifier에 대한 궁금증?

Q. Train 셋의 이미지가 총 N개라면 Train/Test 함수의 계산량은?

-> 학습(train) : O(1) / 예측(predict) : O(N)



* O(빅오) 표기법

: 점근적 실행 시간을 표기할 때 쓰임

- O(1) : 입력 값에 상관 없이 일정한 실행시간

- O(n) : 알고리즘을 수행하는 데 걸리는 시간은 입력값에 비례 (모든 입력 값을 적어도 한 번 이상은 살펴봐야함)

- O(n^2) : 버블 정렬 같은 비효율적인 알고리즘



-> 데이터를 기억만 하면 됨

-> 포인터만 잘 사용해서 복사를 하면, 데이터의 크기와 상관 없이 상수 시간으로 끝마칠 수 있음



but Test time에서는 N개의 학습 데이터 전부를 테스트 이미지와 비교해야 함

상당히 "뒤집어짐" (Train TIme < Test TIme)



Fact, "Train time"은 조금 느려도 되지만, "Test time"에서는 빠르게 동작하길 원함

<img width="428" alt="image" src="https://user-images.githubusercontent.com/87634136/179386718-002e636d-5eb2-4689-9ef2-6f9609d539a6.png">

NN(Nearest Neighbor)

- 훈련(train)이 빠른 이유 : 위의 소스에서 보는 것과 같이 Train 함수는

<img width="258" alt="image" src="https://user-images.githubusercontent.com/87634136/179386723-29d7b6a0-7c09-4793-8155-07e40e9aa988.png">

self.Xtr = X 처럼 데이터를 저장해두는 것 뿐이기 때문에 빠르다.

- Test time이 느린 이유 : train(훈련) 데이터 셋이랑 하나씩 비교해야하므로 느림

<img width="262" alt="image" src="https://user-images.githubusercontent.com/87634136/179386731-5b5ddacf-4535-433a-ac06-00816a78c43b.png">

==> 결론 : NN(Nearest Neighbor)은 테스트 시간이 오래 걸리기 때문에 효율적이지 않아서 사용하지 X



CNN

- 훈련(train)이 오래 걸리는 이유 : 데이터를 단순 저장만 하는 것이 아니라 학습 시키기 때문

(올바른 파라미터를 찾기 위해 하나하나 테스트 해봄)

=> NN(Nearest Neighbor)과 CNN은 Train과 Test가 반대이기 때문에 CNN이 더 효율적이고, 현재 CNN을 많이 사용함





NN의 "decision regions"

<img width="301" alt="image" src="https://user-images.githubusercontent.com/87634136/179386744-c603ce14-0bf4-499c-8bf7-72890281d6d4.png">

- 2차원 평면 상의 각 점 : 학습 데이터

- 점의 색 : 클래스 레이블 (카테고리)

- 위의 사진에서 클래스 : 5개

- 왼쪽 구석에는 파란색, 오른쪽 구석에는 보라색

- 2차원 평면 내의 모든 좌표에서 각 좌표가 어떤 학습 데이터와 가장 가까운지 계산

- 각 좌표를 해당 클래스로 칠함

- 위 사진의 분류기 좋지 않음



>> NN 분류기에서 발생 가능한 문제1 -> 가운데 부분에 대부분이 초록색 점인데, 중간에 노란 색 점이 포함되어 있음)

--> NN 알고리즘은 "가장 가까운 이웃" 만을 보기 때문에, 초록색 무리 한 가운데 노란색 영역이 생김



>> NN 분류기에서 발생 가능한 문제2 -> 유사하게 초록색 영역이 파란색 영역을 침범함

(초록색 점이 끼어들어서 -> 이 점은 잡음이거나 가짜임)



위의 문제들로 인해

NN ---> k-NN 알고리즘 탄생

- 단순하게 가장 가까운 이웃만 찾기보다는 조금 더 고급진 방법

- Distance metric을 이용하여 가까운 이웃을 k개 만큼 찾고, 이웃끼리 투표하는 방법

- 가장 많은 득표수를 획득한 레이블로 예측

- 투표를 하는 방법 -> 거리별 가중치 고려



but 가장 잘 동작하면서도 가장 쉬운 방법 -> 득표수만 고려하는 방법

<img width="323" alt="image" src="https://user-images.githubusercontent.com/87634136/179386757-2bceb275-7311-4bff-b733-bc91d3e08b27.png">
-> 동일한 데이터를 사용한 k-nn 분류기

각각 k =1, 2, 3에서의 결과



k = 3의 경우 : 초록색 영역에 자리 잡았던 노란색 점때문에 생긴 노란 지역이 깔끔하게 사라짐,

중앙은 초록색이 깔끔하게 점령, 왼쪽의 빨강/파랑 사이의 뾰족한 경계들도 점차 부드러워짐



k = 5의 경우 : 파란/빨강 영역의 경계가 아주 부드럽고 좋아짐

====> NN 분류기를 사용하면 k는 적어도 1보다는 큰 값으로 사용

(because k가 1보다 커야 결정 경계가 더 부드러워지고, 더 좋은 결과를 보임)



Q. 레이블링이 안 된 흰색 지역은 어떻게 처리하는지?

A. 흰색 영역은 k-nn이 "대다수"를 결정할 수 없는 지역임. 물론, 흰색 영역을 메꿀 수 있는 더 좋은 방법들도 있음

어떤 식으로든 추론을 해보거나, 임의로 정할 수도 있음. but 여기에서는 단순한 예제라서 가장 가까운 이웃이 존재하지 않으면 단순하게 흰색으로 칠함


k-nn을 사용할 때 결정해야 할 한 가지 사항

-> 서로 다른 점들을 어떻게 비교할 것인지

지금까지는 L1 Distance를 이용 ("픽셀 간 차이 절대값의 합")

but L2를 사용해도 됨 ("제곱 합의 제곱근"을 거리로 이용하는 방법)

어떤 "거리 척도"를 선택할 것인지?

because 서로 다른 척도에서는 해당 공간의 근본적인 기하학적 구조 자체가 서로 다르기 때문

<img width="278" alt="image" src="https://user-images.githubusercontent.com/87634136/179386765-7f6e0fd5-1585-4bef-aece-7948a02e8c48.png">

왼쪽에 보이는 사각형 -> L1 Distance의 관점에서는 원임

(생긴 모습은 원점을 기준으로 하는 사각형의 모양)

L1의 관점에서는 사각형 위의 점들이 모두 원점으로부터 동일한 거리만큼 떨어져 있음

오른쪽에 보이는 원 -> L2의 관점에서는 원

<img width="430" alt="image" src="https://user-images.githubusercontent.com/87634136/179386775-b344e1af-ef65-41bb-a64d-4e218f17416a.png">



-> k-nn 분류기로 이 문제를 다루려면 어떤 거리 척도를 사용할 지만 정해주면 됨

(두 문장 간의 거리를 측정할 수 있는 어떤 것이든 사용하면 됨)

<img width="322" alt="image" src="https://user-images.githubusercontent.com/87634136/179386781-71d3d5f4-d253-4ea2-981e-5777d520bf24.png">

<img width="436" alt="image" src="https://user-images.githubusercontent.com/87634136/179386785-91504022-44c3-4186-9934-82027e19e74e.png">



하이퍼 파라미터 (어떻게 하면 "내 문제"와 "데이터"에 꼭 맞는 모델을 찾을 수 있을까?)

: k와 거리척도

- Train time에 학습하는 것이 아님



하이퍼 파라미터를 어떻게 정의해야할까?

-> 문제 의존적 (problem-dependent)

데이터에 맞게 다양한 하이퍼 파라미터 값을 시도해 보고 가장 좋은 값을 찾음

여러가지 시도를 해보고 좋은 것을 선택하는 것이 좋음



but 하이퍼 파라미터 값들을 실험해보는 방법

방법1 >> 다양한 하이퍼 파라미터를 시도해 보는 것

방법2 >> 그 중 최고를 선택하는 것

<img width="316" alt="image" src="https://user-images.githubusercontent.com/87634136/179386793-f4bd032b-9e3f-4493-95cb-8d613d3aea06.png">

<img width="316" alt="image" src="https://user-images.githubusercontent.com/87634136/179386800-92466fd5-2515-4a7c-96a1-b76ebd4254a0.png">

<img width="316" alt="image" src="https://user-images.githubusercontent.com/87634136/179386802-f726289d-605a-4bae-adb5-8e0ea632ac88.png">

< 위의 3번째 사진 >

- 5-Fold Cross Validation을 사용하고 있음

- 처음 4개의 fold에서 하이퍼 파라미터를 학습시키고, 남은 한 fold에서 알고리즘을 평가함.

그리고 1,2,3,5 fold에서 다시 학습시키고 4 fold로 평가함. 이런식으로 계속 순환함.



=> 이런 방식으로 최적의 하이퍼파라미터를 확인할 수 있음

but 실제로는 잘 쓰지 않음



Q. 어떤 경우에 L1 Distance가 L2 Distance보다 더 좋은지?

A. 문제 의존적임. 어떤 경우에 L1/L2를 써야 하는지 결정하는 것은 어렵지만, L1은 좌표계에 의존적이므로 데이터가 좌표계에 의존적인지를 판단하는 것이 판단 기준이 될 수 있다.

어떤 특징 벡터가 있고, 각 요소가 어떤 특별한 의미를 지니고 있다면, 직원들을 분류하는 문제가 있을 때, 데이터의 각 요소가 직원들의 다양한 특징에 영향을 줄 수 있다. 예를 들어 봉급, 근속 년수가 있다.

각 요소가 특별한 의미를 가지고 있다면 L1을 사용하는것이 좀 더 괜찮음


<img width="424" alt="image" src="https://user-images.githubusercontent.com/87634136/179386809-f495be1d-8a84-45df-b493-49bbc4a544f4.png">


--> 공간을 조밀하게 덮으려면 충분한 량의 학습 데이터가 필요하고, 그 양은 차원이 증가함에 따라 기하급수 적으로 증가함 (아주 좋지 않은 현상)



Q. 그림의 초록 점과 파란 점은 무엇인지?

A. 각 점은 트레이닝 샘플들을 의미함. 점 하나하나가 트레이닝 샘플임. 그리고 각 점의 색은 트레이닝 샘플이 속한 카테고리를 나타낸다고 보면 됨. 맨 왼쪽의 1차원을 보면 이 공간을 조밀하게 덮으려면 트레이닝 샘플 4개면 충분함. 2차원 공간을 다 덮으려면 16개가 필요함. 1차원의 4배 임. 이렇게 3, 4, 5 차원 같이 고차원을 고려해보면 각 공간을 조밀하게 덮기 위해 필요한 트레이닝 샘플의 수는 차원이 늘어남에 따라 기하급수적으로 증가함.



Linear Classification

- 어떤 선을 구해서 클래스를 구분(선으로 구분)

(Linear : 선, Classification : 클래스 간 구분, class : 카테고리)

- 아주 간단한 알고리즘

but 아주 중요하고 NN과 CNN의 기반 알고리즘

- 일부 사람들은 Nerural Network를 레고 블럭에 비유

- NN을 구축할 때 다양한 컴포넌트들을 사용할 수 있고, 이 컴포넌트들을 한데 모아서 CNN이라는 거대한 타워를 지을 수 있음

- Linear Classifier : "parametric model"의 가장 단순한 형태

- Linear classifier는 행렬과 벡터 곱의 형태



Image Captioning (NN의 구조적 특성을 설명하는 예시)

: 이미지가 입력, 이미지를 설명하는 문장이 출력

- 이미지를 인식하기 위해 : CNN을 사용

- 언어를 인식하기 위해 : RNN을 사용

- 두 개(CNN + RNN)를 레고 블럭처럼 붙히고 한번에 학습시키면 어려운 문제도 해결할 수 있음.

<img width="328" alt="image" src="https://user-images.githubusercontent.com/87634136/179386823-8fb2fae6-2c99-461c-a0d4-c393aa9e6bb2.png">

-> 고양이 이미지 : 입력 이미지(X)

-> parameter(가중치) : "W", 세타(theta)

- data X와 parameter W를 가지고 10개의 숫자 출력

이 숫자는 CIRAR-10의 각 10개의 카테고리의 스코어

(고양이의 스코어가 높다 -> 입력 x가 고양이일 확률이 크다)

- b : Bias



가중치 W와 데이터 X를 조합하는 가장 쉬운 방법?

-> Linear classification

F(x,W) = Wx



-> 입력 이미지 : 32x32x3 = 3,072-dim(열 벡터)

3,072-dim 열 벡터가 10-classes 스코어가 되어야함 (10개의 카테고리에 해당하는 각 스코어를 의미하는 10개의 숫자를 얻고 싶다는 뜻) --> W = 10 x 3072

10 x 3072를 곱하면 10-classes 스코어를 의미하는 10 x 1짜리 하나의 열 벡터를 얻게 됨

bias term은 10-dim 열 벡터 (입력과 직접 연결되지 않음)




<img width="329" alt="image" src="https://user-images.githubusercontent.com/87634136/179386831-0f4b74b6-b6b8-4e17-afd5-7511ff9a1608.png">

-> 이 Linear classifier는 2x2 이미지를 입력으로 받고, 이미지를 4-dim 열 벡터로 쭉 폄

10개 클래스 대신 고양이, 개, 배의 세 가지 클래스를 보면

W = 4 x 3 행렬 (입력 픽셀 4개, 클래스 총 3개)

추가적으로 3-dim bias 벡터 있음 (bias는 데이터와 독립적으로 각 카테고리에 연결됨)



==> 고양이 스코어는?

: 입력 이미지의 픽셀 값들과 가중치 행렬을 내적한 값에 bias term을 더한 것임

(Linear classification은 템플릿 매칭과 거의 유사)




<img width="335" alt="image" src="https://user-images.githubusercontent.com/87634136/179386836-9af5dbb7-36fa-4293-a1e4-f54639a7ba4a.png">

=> y=wx만 표현하면 원점을 지난 직선이 표현이 됨 -> 그래서 평행이동하여 y=wx+b

* b(bias;편향)의 역할 : 원점에서 평행이동을 함으로써 데이터들이 골고루 분포되어 있지 않고, 한 쪽으로 치우쳐져 있는(데이터의 편향)것을 b라는 숫자 하나를 더해줘서 편향된 데이터들의 구분을 원활하게 함



내적?

: 클래스 간 탬플릿의 유사도를 측정하는 것과 유사함
