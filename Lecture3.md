CS231n: 시각적 인식을 위한 컨볼루션 신경망

http://cs231n.stanford.edu/

스탠포드 대학교 강의





>>> Lecture2 Review 

Recall from last time : Challenges of recognition

- 인식에서의 어려운 점을 살펴봄

- 데이터 중심의 방식에 초점을 맞춰봄 

- 이미지 분류에 대해서 배움

1. 왜 이미지 분류가 어려운지

2. 컴퓨터가 보는것과 사람이 보는 것의 차이가 있음

3. 분류를 어렵게하는 조명의 변화, 변형에 대해 다룸

그리고 인간의 시각체계는 이 일을 쉽게 하는데 컴퓨터는 왜 이것이 어려운지?



Recall from last time : data-driven approach, kNN

- kNN 분류기

데이터 중심 접근 방법 중에 가장 단순한 방법

- CIFAR-10

<img width="203" alt="image" src="https://user-images.githubusercontent.com/87634136/179699655-2ba91a1a-42fb-450a-8a9e-ce34c921aaea.png">

비행기, 자동차 등 10개의 카테고리가 있음

어떻게 kNN을 이용해서 학습 데이터를 가지고 각 카테고리 클래스를 분류하는 결정 경계를 학습시킬 수 있는지

- Cross Validation 

- Train Validation

- Test Set으로 나눠서 하이퍼파라미터를 찾는 법

<img width="391" alt="image" src="https://user-images.githubusercontent.com/87634136/179699705-387c23a1-8824-4ff4-a1f1-684f3f58ba39.png">



Recall from last time : Linear Classifier

- Linear classification 

: 뉴럴 넷의 기본 요소

: Linear classifier는 parametric classifier의 일종

-> parametric classifier : 트레이닝 데이터의 정보가 파라미터인 "행렬 W"로 요약된다는 것을 뜻함, 그리고 W가 학습됨

: Linear classifier -> 이미지를 입력 받으면 하나의 긴 벡터로 편다

<img width="389" alt="image" src="https://user-images.githubusercontent.com/87634136/179699773-1317de15-a131-414d-b7d8-1ea7dcd6c5f0.png">

-> 위에 고양이 이미지(X)가 있으며 32x32x3 픽셀임 -> 긴 열벡터로 펼치게 됨

(32x32는 이미지의 높이x너비 / 3은 red, green, blue)



-> 파라미터 "행렬 W"가 있음

행렬 W : 위에서 만들었던 이미지 픽셀과 연산 -> CIFAR-10의 각 10개의 클래스에 해당하는 클래스 score를 만들어줌 

- score가 더 큰 값(즉, 고양이 클래스의 score가 더 크다는 것)은 분류기가 이 이미지는 고양이일것 같다고 생각함)

On the contrary,,,

개 or 자동차 클래스의 score가 더 낮다는 것 -> 이 이미지가 개 or 자동차일 확률이 낮다는 것을 의미



-> Linear Classification (각 클래스 템플릿으로 보는 것)

위의 그림에 왼쪽 아래를 보면..

행렬 W에는 각 10개의 클래스와 이미지의 모든 픽셀에 대응되는 하나씩의 요소가 있음 -> "픽셀이 클래스를 결정하는데 얼마나 중요한 역할을 하는지" 의미

- 행렬 W의 각 행들이 해당하는 그 클래스의  템플릿이 되는 것)

- 행렬 W의 각 행은 "이미지의 픽셀 값"과 "해당 클래스" 사이의 가중치가 되기 때문에

각 행을 풀어서 다시 한 이미지로 재구성하면 각 클래스에 대응하는 학습된 템플릿을 볼 수 있음



-> 고차원 공간에서 일종의 "결정 경계"를 학습한다는 측면으로 볼 수 있음

(공간의 차원은 이미지의 픽셀 값에 해당하는 것)



Lecture2에서는?

- Linear classifier에 대한 간략한 아이디어만 보았고, 실제로 행렬 W를 어떻게 만든는지는 다루지 않음 

(가장 좋은 행렬 W를 구하는데 어떻게 training data를 활용해야하는지 언급하지 않음)

- 임의의 행렬 W를 사용했고, 그 행렬 W를 가지고 각 이미지에 해당하는 10개의 클래스 스코어를 계산하였음

이때의 클래스 score는 임의로 정했기 때문에 좋을수도 있고 나쁠수도 있음



ex) 세 개의 트레이닝 데이터에 대한 임의의 행렬 W를 가지고 예측한 10개의 클래스 score

<img width="248" alt="image" src="https://user-images.githubusercontent.com/87634136/179699880-ac682842-91e3-4d96-90a8-a39cc32d81b8.png">

-> 굵게 표시한 각 스코어를 보면 좋은것과 나쁜것을 볼 수 있음



 고양이 

-> 왼쪽 이미지는 고양이다 (인간이니까 쉽게 파악 가능)

but, 고양이에 부여된 score를 보면



분류기가 부여한 점수 ? 

cat -> 2.9점 

frog -> 3.78점

==> 이 분류기는 별로 좋지 않음 

why? 

정답 클래스(즉, 고양이)가 가장 높은 점수가 되는 분류기를 원하기 때문



 자동차 

자동차의 score가 가장 높음 -> 좋음



 개구리 

개구리의 score : -4 (오히려 다른 score 보다 훨씬 낮음) -> 매우 좋지 않음



BUT !!!  이렇게 분석하는 방법 좋지 않음

(score를 눈으로 훑으면서 어느것이 좋고, 어느것이 나쁜지 살펴보기만 하는 것은 좋은 생각이 아님)



Q. 어떤 W가 가장 좋은지를 결정하기 위해서는 ??

A. 지금 만든 W가 좋은지 나쁜지를 ​정량화 할 방법이 필요함



So, 손실함수 사용 !! 

* 손실함수 -> W를 입력으로 받아서 각 score를 확인하고, 이 W가 지금 얼마나 별로인지를 정량적으로 말해주는 것



>>> Lecture 3 

손실함수는 어떤 일을 해야할까?

- 임의의 값 W가 얼마나 좋은지 나쁜지를 정량화해 줘야 함

- 실제로 원하는 것은 ! 행렬 W가 될 수 있는 모든 경우의 수에 대해서 "그나마 괜찮은" W가 무엇인지 찾고 싶음.

==> 이런 과정?  "최적화 과정"



클래스를 3개로 하는 toy data set 사용

<img width="260" alt="image" src="https://user-images.githubusercontent.com/87634136/179699999-e76e413a-351c-4668-839b-4359912586e5.png">

위의 사진에서 ..

고양이 클래스는 잘 분류 X 

자동차 클래스는 잘 분류 O

개구리 클래스는 최악 (개구리 점수는 다른 것보다 더 낮음)



이것을 공식화 하면? (손실함수)

training data X와 Y

<img width="407" alt="image" src="https://user-images.githubusercontent.com/87634136/179700039-41424287-2037-4ffd-9575-f501822ac916.png">

- X(대부분) : 알고리즘의 입력

- Y : label or 타겟 (예측하고자 하는것)



- IF...Image classification 알고리즘 ?  

-> X : 이미지 / Y : 예측하고자 하는 것 

-> 각 이미지를 CIFAR-10의 10개의 카테고리 중 하나로 분류하는 것



-> 위의 사진에서 label y -> 1 ~ 10 사이의 정수 값 

(프로그래밍 언어에 따라 0 ~ 9일수도 있음)



So, y라는 정수값은 각 이미지 x의 정답 카테고리를 의미



-> 입력 이미지 x와 행렬 W를 입력으로 받아서 y를 예측하는것임

 IF...Image classification 문제라면 y => 10개가 됨 (CIFAR-10의 경우)



Next..

손실함수 Li를 정의

-> 예측함수 f와 정답 값 Y를 입력으로 받아서 이 training sample을 얼마나 구리게 예측하는지를 정량화시켜줌



And..

최종 Loss인 "L"은 ? -> data set에서 각 N개의 샘플들의 Loss의 평균이 됨

<img width="251" alt="image" src="https://user-images.githubusercontent.com/87634136/179700128-3781d898-f759-4ac8-a09c-ad5caeefdabf.png">

위의 함수는 일반적인 공식임



구체적인 한 손실함수의 예시

 multi-class SVM loss 

- multi-class SVM : 여러 클래스를 다루기 위한 이진 SVM의 일반화된 형태

(이진 SVM은 두 개의 클래스만 다룸 - 각 데이터는 Positive or Negative로 분류될 뿐)



But !

여러개의 클래스(예를들어 위의 예시처럼 10개의 클래스)를 다루려면 이 개념을 좀 더 일반화 시켜야함 



각각의 training data에서 Loss Li를 구하는 방법

<img width="389" alt="image" src="https://user-images.githubusercontent.com/87634136/179700229-25d2dc63-cc5a-4a9a-92e5-e58d89cee2c6.png">

1. Li를 구하기 위해 "True인 카테고리"를 제외한 "나머지 카테고리 Y"의 합을 구함

(맞지 않는 카테고리를 전부 합치는 것)



2. 올바른 카테고리의 score와 올바르지 않은 카테고리의 score를 비교

if (올바른 카테고리의 점수 > 올바르지 않은 카테고리의 점수) and 일정 마진(safety margin) 이상 

=> 마진을 1로 둠 

(이 경우는 True인 score가 다른 False 카테고리보다 훨씬 더 크다는 것을 의미 ==> Loss = 0)

- 이미지 내 정답이 아닌 카테고리의 모든 값들을 합치면 그 값이 바로 한 이미지의 최종 Loss가 됨



3. 전체 트레이닝 데이터 셋에서 그 Loss들의 평균을 구함



4. 수식화 시키면 if-then으로 표현할 수 있음

if -> 정답 클래스의 score의 점수가 가장 높으면

then -> max(0, s_j - s_yj + 1)



5. 0과 다른 값의 최댓값, Max(0, value)과 같은 식으로 손실 함수를 만든다

==> 이런 종류의 손실함수를 "hinge loss(경첩)"라고 부르게도 함 (그래프의 모양 때문에 붙혀진 이름임)

<img width="403" alt="image" src="https://user-images.githubusercontent.com/87634136/179700326-31291ca2-0676-448f-a647-effd9cffa031.png">

-> x축 : S_Yi (실제 정답 클래스의 score) / y축 : Loss​

* S : 분류기의 출력으로 나온 예측된 score

(예를 들어 1이 고양이이고, 2가 개이면 S_1은 고양이 score S_2는 개 score)

* Y_i : 이미지의 실제 정답 카테고리 (즉, 정수 값임) 

* S_Y_i : training set의 i번째 이미지의 정답 클래스의 score



-> 정답 카테고리의 점수가 올라갈수록 Loss가 선형적으로 줄어드는 것을 알 수 있음 

-> Loss는 0이 된 이후에도 Safety margin을 넘어설 때까지 더 줄어든다

-> Loss가 0이 됐다는 것은 클래스를 잘 분류했다는 뜻임



Q. 정확히 무엇을 계산하는 것인지 ?

A. 여기서 Loss가 말하고자 하는 것은 정답 score가 다른 score들보다 높으면 좋다는 것임

정답 score는 safty margin(여기에서 1)을 두고 다른 score들 보다 훨씬 더 높아야 함

충분히 높지 않으면 Loss가 높아지게 됨



>> 3개의 트레이닝 데이터 셋을 이용한 예시

<img width="401" alt="image" src="https://user-images.githubusercontent.com/87634136/179700393-e828d8be-b151-4e6c-b90d-902f65d0f124.png">


-> case space notation 제거 & zero one notation 사용



-> 사진의 맨 왼쪽 예시를 보면 multi-class SVM loss가 계산되는 과정을 보면...

정답이 아닌 클래스를 순회

cat은 정답 클래스이므로, car과 frog 클래스를 순회한다.



다시 Car의 경우를 살펴보면..

(Car 스코어) 5.1 - (Cat 스코어) 3.2 + 1(magin)를 구한다



Cat과 Car를 비교할 때 짐작할 수 있는 것?

-> Car가 Cat 보다 높으니까 Loss가 발생할 것이라는 것



Cat 이미지의 예시를 보면..

2.9의 손실이 발생함



Cat-Car 와 Car-Flog를 비교해보면..

Car는 3.2, Flog는 -1.7의 에러가 발생한다.



Cat 스코어는 Frog 스코어보다 훨씬 크므로 Loss는 0



고양이 이미지의 Multiclass-SVM Loss는 이런 클래스쌍의 Loss의 합이 된다

(즉,  2.9 + 0 = 2.9가 된다.)

* 여기에서 2.9라는 숫자가 "얼마나 분류기가 이 이미지를 안 좋게 분류하는지" 에 대한 척도가 된다.



===> 이러한 작업을 여러번 반복



Car 클래스에 다시한번 해보면..

<img width="394" alt="image" src="https://user-images.githubusercontent.com/87634136/179700495-43d902ce-5a82-4b0e-b38e-a601aaea0d94.png">

* Car와 Cat을 비교해 본다.

-> Car 스코어가 Cat 스코어보다 높기 때문에 여기에서 Loss는 0이 된다.

* Car 와 Frog도 비교해본다.

-> Car의 스코어가 Frog보다 훨씬 더 크다는 것을 알 수 있다.

(그러므로 Car 의 경우는 전체 Loss도 0이 된다)



Frog 클래스에 다시한번 해보면..

<img width="395" alt="image" src="https://user-images.githubusercontent.com/87634136/179700564-11471ecc-d8cf-49b6-b269-fc1df1095777.png">

* Frog와 Cat을 비교해 본다.

-> 엄청 큰 Loss가 발생함을 알 수 있다. (Flog score가 엄청 낮기 때문이다.)

* Frog와 Car을 비교해 본다.

-> 이 경우도 같다. ( Frog score 자체가 엄청 낮기 때문)

=> 전체 Loss = 12.9



===> 전체 training set의 최종 loss는 각 training 이미지의 loss들의 평균이 된다. (약 5.3)

(이것이 의미하는 것은, 분류기가 5.3점 만큼 이 training set을 안 좋게 분류하고 있다는 "정량적 지표"가 된다.)



Q. safty margin이 1이었는데, 위에서 계산하면서 1을 더하는것은 어떤 의미인지?

A. 이 숫자가 임의로 선택한 숫자같아 보이긴 하지만, 사실 손실함수의 "score가 정확이 몇인지"는 신경쓰지 않는다. 신경 써야할 것은 여러 score 간의 상대적인 차이이다. 우리가 관심있어 하는 것은 오로지 정답 score가 다른 score에 비해 얼마나 더 큰 스코어를 가지고 있는지이다. 행렬 W를 전체적으로 스케일링한다고 생각해 보면 결과 score도 이에 따라 스케일이 바뀔 것이다. 



Q. 만약 Car 스코어가 조금 변하면 Loss에는 무슨 일이 일어날까 ?

<img width="396" alt="image" src="https://user-images.githubusercontent.com/87634136/179700610-e879c736-089f-4c3c-a022-ca189ff77c9e.png">

A. Car의 스코어를 조금 바꾸더라도 Loss가 바뀌지 않을 것이다. 다시 SVM loss를 상기해보면 이 loss는 오직 정답 score와 그 외의 score와의 차이만 고려했다. 따라서 이 경우에는 Car score가 이미 다른 score들보다 엄청 높기 때문에 여기 score를 조금 바꾼다고 해도, 서로 간의 간격(Margin)은 여전히 유지될 것이고, 결국 Loss는 변하지 않는다. 계속 0일 것이다. 





Q. SVM Loss가 가질 수 있는 최댓/최솟값이 어떻게 될까 ?

<img width="392" alt="image" src="https://user-images.githubusercontent.com/87634136/179700650-f568f7ad-222a-4c63-aae0-347902d77f10.png"> 

A. 최솟값은 0이 된다. 왜냐하면 모든 클래스에 걸쳐서  정답 클래스의 스코어가 제일 크면 모든 트레이닝 데이터에서 loss가 0이 되기 때문이다. 그리고 다시 이 손실 함수가 hinge loss 모양이라는 점을 고려해 보면, 만약 정답 클래스 score가 엄청 낮은 음수 값을 가지고 있다고 생각해보면, 아마 Loss가 무한대 일 것이다. 그러니 최솟값은 0이고 최댓값을 무한대 일 것이다. 





Q. 파라미터를 초기화하고 처음부터 학습시킬 때, 보통은 행렬W를 임의의 작은 수로 초기화시키는데, 그렇게되면 처음 학습 시에는 결과 score가 임의의 일정한 값을 갖게 된다. 그렇다면 만약 모든 score S가 거의 "0에 가깝고", "값이 서로 거의 비슷하다면" Multiclass SVM에서 Loss가 어떻게 될까 ?

<img width="401" alt="image" src="https://user-images.githubusercontent.com/87634136/179700762-dac969eb-f75e-4f83-8dcf-76f583ad5613.png">

A. "클래스의 수 - 1" 이다. 왜냐하면 Loss를 계산할때 정답이 아닌 클래스를 순회합니다. 그러면 C - 1 클래스를 순회하게된다. 비교하는 두 score가 거의 비슷하니 Margin때문에 1 score를 얻게 된다. 그리고 전에 Loss는 C - 1을 얻게 되는 것이다. ==> "디버깅 전략"

* C - 1(정답 클래스 제외해야해서)



"디버깅 전략"

이 전략을 가지고 training을 시작하면 Loss가 어떻게 될 지를 짐작할 수 있게 됨

(training을 처음 시작할 때 Loss가 C-1이 아니라면 아마 버그가 있는 것이고, 고쳐야 함)





Q. SVM Loss는 정답인 클래스는 빼고 다 더했다. 그렇다면 정답인 것도 같이 더하면 어떻게 될까 ?

<img width="403" alt="image" src="https://user-images.githubusercontent.com/87634136/179700821-ac858bc1-ccfb-48ad-9896-cfd40824881b.png">

A. Loss에 1이 더 증가한다는 것이다. 정답 클래스만 빼고 계산하는 이유는, 일반적으로 Loss가 0이 되야지만 "아무것도 잃는 것이 없다"고 쉽게 해석할 수 있으며. Loss에 모든 클래스를 다 더한다고 해서 다른 분류기가 학습되는 것은 아니다. 하지만 관례상 정답 클래스는 빼고 계산을 하며, 그렇게 되면 최소 Loss는 0이 된다. 





Q. Loss에서 전체 평균을 쓰는게 아니라 합을 쓰면 어떻게 될까 ?

<img width="403" alt="image" src="https://user-images.githubusercontent.com/87634136/179700875-1f9719d7-1d73-4b29-a6ec-a7195ea5631d.png">

A. 영향을 미치지 않는다. 클래스의 수는 어차피 정해져 있으니 상관이 없을 것입니다. 단지 스케일만 변할 뿐이다.

왜냐하면 score 값이 몇인지는 신경쓰지 않기 때문이다.





Q. 만약 손실함수를 아래와 같이 제곱 항으로 바꾸면 어떻게 될까요?

<img width="401" alt="image" src="https://user-images.githubusercontent.com/87634136/179700910-1a027651-e944-44a3-99ef-bea5cadd0643.png">

A. 결과는 달라질 것이다. 좋은것 과 나쁜것 사이의 트레이드 오프를 비 선형적인 방식으로 바꿔주는 것이다. 그렇게 되면 손실함수의 계산 자체가 바뀌게 된다. 실제로도 squared hinge loss를 종종 사용한다. 이것은 손실함수를 설계할때 쓸 수 있는 한가지 방법이 될 수 있다.



 *** 왜 굳이 제곱 항을 고려 해야하나 ?

--> 손실함수의 요지는 "얼마나 안 좋은지"를 정량화 하는 것이다. 그리고 분류자가 다양한 종류의 실수를 저지르고 있다면, 어떻게 해야 이 분류기가 만드는 다양한 Loss들 마다 상대적으로 패널티를 부여할 수 있을까 ? 만약 Loss에 제곱을 한다면 이제 "엄청 엄청 안좋은 것들" 은 정말로 "배로 안좋은 것" 이 된다. 결국 매우 나빠지게 된다. 



but hinge loss를 사용하게 되면..?

실제로 "조금 잘못된 것" 과 "많이 잘못된 것" 을 크게 신경쓰지 않게 되는 것이다.

학습을 통해 Loss를 줄일 것인데, 그 줄어드는 Loss의 량이 "조금 잘못된 것"이던 "많이 잘못된 것" 이던 큰 차이가 없을 것이다. 



--> 둘 중 어떤 loss를 선택하느냐는 에러에 대해 얼마나 신경쓰고 있고, 그것을 어떻게 정량화 할 것인지에 달려있다. 

그리고 이 문제는 실제 손실함수를 만들때 고려해야만 하는 것이다.

Because, 

손실 함수라는 것이 여러분이 여러분들의 알고리즘에게 "어떤 에러를 내가 신경쓰고 있는지", 그리고 "어떤 에러가 트레이드오프 되는 것인지" 를 알려주는 것이다.



thus, 실제로는 문제에 따라서 손실함수를 잘 설계하는 것은 엄청 중요하다고 할 수 있다.

<img width="386" alt="image" src="https://user-images.githubusercontent.com/87634136/179700987-2fccb279-e230-4614-887f-a644523f254b.png"> 

-> numpy 코드이다.

-> numpy를 사용하면 코드 몇 줄이면 이 손실 함수를 코딩할 수 있다는 것이다.



-> max로 나온 결과에서 정답 클래스만 0으로 만들어준다. (margin[y] = 0)

(굳이 전체를 순회할 필요가 없게 해주는 일종의 vectorized 기법임)

전체 합을 구할 때는 제외하고 싶은 부분만 0으로 만들어주면 된다.

<img width="386" alt="image" src="https://user-images.githubusercontent.com/87634136/179701029-be5ed362-9cee-4b21-b7db-d0bac9f22e41.png">

-> 만약 W가 0인 정답을 찾았다고 하면, 잃은 것이 전혀 없고, 이긴 것이다.

이렇게 Loss가 0이 되게 하는 W가 유일하게 하나만 존재하는 것일까? -> No, 다른 W도 존재한다.

because, W의 스케일은 변한다. 하지만 W에 두 배를 한다고 해도 Loss의 값은 변하지 않는다.

<img width="397" alt="image" src="https://user-images.githubusercontent.com/87634136/179701465-36b73c43-9dad-4161-b81a-1afdc8c332be.png">


-> If, W와 2W가 있다면, 정답 score와 정답이 아닌 스코어의 차이의 마진(margins) 또한 두 배가 될 것이다.

Thus, 모든 마진(margins)이 이미 1보다 더 크면, 두배를 한다고 해도 여전히 1보다 클 것이고, Loss가 0일 것이다. 



-> 손실함수 = 분류기에게 어떤 W를 찾고 있고, 어떤 W에 신경쓰고 있는지를 말해주는 것이라면 

조금 이상함

why? -> "불일치 하는 점이 있음"

다양한 W중 Loss가 0인 것을 선택하는 것은 모순이다.

<img width="288" alt="image" src="https://user-images.githubusercontent.com/87634136/179701513-0dbe9535-1e68-4021-a4a4-46acffd2068d.png">

because, 여기에서는 오직 데이터의 loss에 대해서만 신경을 쓰고 있고, 분류기에게 training data에 꼭 맞는 W를 찾으라고 말하는 것과 같다.



but, 실제로 training data를 이용해서 어떤 분류기를 찾는 것인데, 그 분류기는 test data에 적용할 것이기 때문



thus, training data의 성능에 관심있는 것이 아니라 test data의 성능에 관심이 있는 것이다. 

분류기에게 training data의 Loss에만 신경쓰라고 하면 좋지 않은 상황이 나타남




<img width="314" alt="image" src="https://user-images.githubusercontent.com/87634136/179701577-0cea29a4-1dba-40e9-b9b0-c80b9e06457b.png">

-> 파란 점의 데이터 셋이 있다.

-> 할 일 : 어떤 곡선을 가지고 저 파란색 점들에 fitting 시키는 것

분류기야 training data에 fit하게 해 !

그려면 분류기는 모든 트레이닝 데이터를 완벽하게 분류해내기 위해 구불구불한 곡선을 만든다.

but, 이러한 것은 좋지 않음



because, "성능"에 대해서 전혀 고려하지 않았기 때문

So, 항상 test data의 성능을 고려해야함



If, 새로운 데이터가 들어오게 된다면 -> 앞에서 만든 구불구불한 곡선은 잘못된 것이 된다.

<img width="389" alt="image" src="https://user-images.githubusercontent.com/87634136/179701634-0814d9ee-f4c8-4326-acdf-0baddffe9dc7.png">

-> 원래 의도했던 것은 초록색 점이다.

So, 완벽하게 training data에 fit한 복잡하고 구불구불한 곡선을 원한 것이 아니다.

(기계학습에서 가장 중요한 문제)



--> 이것을 해결하는 방법 : Regularization(조직화, 규칙화) 

Regulation : 모델이 training dataset에 완벽히 fit 하지 못하도록 모델의 복잡도에 패널티를 부여하는 방법

1. 손실함수에 항을 하나 추가한다.



2. "Data Loss Term"에서는 분류기가 training data에 fit하게 한다. 

(보통 손실 함수에 "Regularization term"을 추가하는데 이것은 모델이 좀 더 단순한 W를 선택하도록 도와준다.)



3. "simple" 개념은 해결해야할 문제나 모델에 따라 조금씩 달라짐



4. 과학 계에서 쓰이는 "Occam's Razor"이라는 말이 있음

if, 다양한 가설들을 가지고 있고, 그 가설들 모두가 어떤 현상에 대해 설명이 가능하다면, 일반적으로 "more simple"을 선호해야한다. 

because, "more simple"이 미래에 일어날 현상을 잘 설명할 가능성이 더 높기 때문이다.



And, 기계학습에서는 이러한 것을 사용하기 위해 "Regularization penalty"라는 것을 만들어 냈다.

Regularization은 보통 R로 표기함



일반적인 손실 함수의 형태는 두 가지 항을 가지게 된다.

Data loss / Regularization loss

and 하이퍼파라미터인 람다(lambda)도 생겼다.

(Regularization의 하이퍼파라미터 람다는 실제로 모델을 훈련시킬 때 고려해야 할 중요한 요소 중 하나이다.)



Regularization의 두 가지 역할 ?

1. 모델이 더 복잡해 지지 못하도록 하는 것

2. 모델에 soft penalty를 추가하는 것으로 보는 것

=> "만약 너가 복잡한 모델을 계속 쓰고싶으면, 이 penalty를 감수해야 할 거야!" 

(모델은 여전히 더 복잡한 모델이 될 가능성이 있다.)

<img width="394" alt="image" src="https://user-images.githubusercontent.com/87634136/179701749-4fb97d87-731d-4613-8a55-61b5b8e97859.png">

Regularization의 종류 ?

1. L2 Regularization (= Weight decay)

- 가중치 행렬W에 대한 Euclidean Norm

- 가끔 squared norm라고도 함 or 1/2 * squared norm 을 사용하기도 함 (미분이 더 깔끔해진다)

but, L2 Regularization의 주요 아이디어는가중치 행렬 W의 euclidean norm에 패널티를 주는 것



2. L1 regularization

- L1 norm으로 W에 패널티를 부과하는 것

- L1 Regularization을 하면 행렬 W가 희소행렬이 되도록 한다. 



* 희소 - 비어있다 (0이 많은 것 - 의미 있는 값이 많지 않다.)



3. Elastic net regularization (L1 + L2)



4. Max norm regularization (L1, L2 대신에 max norm 사용)

<img width="398" alt="image" src="https://user-images.githubusercontent.com/87634136/179701788-d87337de-5644-4cfd-a851-17c1a9a32dd2.png">



Fei-Fei Li & Justin Johnson & Serena Yeung 

-> training data x와, 서로 다른 두개의 W가 있다.



-> x는 4줄짜리 벡터이고, 여기에 두개의 서로 다른 W에 대해 생각해 볼 수 있다. 

x = [1 1 1 1] 



W중 하나는 처음에만 1이 있고 나머지 세 원소는 0이다.

w1 = [1 0 0 0]



다른 하나는 원소가 모두 0.25이다.

w2 = [0.25 0.25 0.25 0.25]



=> Linear classification을 구할 때 ?

x와 w의 내적(dot product)을 구한다.



Linear classification의 관점에서 w1와 w2는 같다.

because, x와의 내적이 서로 같기 때문



Q. w1, w2중 L2 regression이 더 선호하는것은 어떤 것일까 ?

A. L2 regression은 w2를 더 선호할 것이다. 

because, L2 regression에서는 w2가 더 norm이 작기 때문이다. (1보다 0.25가 더 작기 때문)

Thus, L2 Regression은 분류기의 복잡도를 상대적으로 w1와 w2중 어떤 것이 더 값이 매끄러운지를 측정한다.



Linear classification에서 W가 의미하는 것 ? 

-> "얼마나 x가 Output Class와 닮았는지"

​

L2 Regularization ->  x의 모든 요소가 영향을 줬으면 함

So, 변동이 심한 어떤 입력 x가 있고, 그 x의 특정 요소에만 의존하기 보다 모든 x의 요소가 골고루 영향을 미치기를 원한다면, L2 Regularization를 통해 더 강해질 수 있음



but, L1 Regularization은 L2 Regularization와는 정반대임



L1 Regularization ?

-> "복잡도"를 다르게 정의

-> 가중치 W에 0의 갯수에 따라 모델의 복잡도를 다룬다.



So, "복잡도" 을 어떻게 정의하느냐, L2 Regularization은 "복잡도"을 어떻게 측정하느냐는 어떤 문제를 가지고 있는지에 따라 다름

<img width="409" alt="image" src="https://user-images.githubusercontent.com/87634136/179701895-597c5cf5-7276-4073-99fa-1f5a851c1082.png">



Multinomial logistic regression (= softmax)

- 딥러닝에서 이것을 더 많이 씀



multi-class SVM loss에서는 score 자체에 대한 해석은 고려하지 않았다.

but, Multinomial Logistic regression의 손실함수는 score 자체에 추가적인 의미를 부여한다.



and, 수식을 이용해서 스코어를 가지고 클래스 별 확률 분포를 계산하게 될 것이다.



softmax라고 불리는 함수를 쓸 것임.

-> score를 전부 이용하는데,  score들에 지수를 취해서 양수가 되게 만듭니다.

그리고 그 지수들의 합으로 다시 정규화 시킨다. 



So, softmax 함수를 거치게 되면 확률 분포를 얻을 수 있고 그것은 바로 해당 클래스일 확률이 되는 것이다. 



확률이기 때문에 0 ~1 사이의 값이고,  모든 확률들의 합은 1이 된다.

​

식을 해석하면 ?

- score를 가지고 계산한 확률 분포가 있다.

- 이 확률 값을 실제 값과 비교

If, 이미지가 고양이라면, 실제 고양이 일 확률 = 1, 나머지 클래스의 확률 = 0

<img width="391" alt="image" src="https://user-images.githubusercontent.com/87634136/179701944-7bffab8b-66b1-492b-9f89-64046d730a16.png">



- softmax에서 나온 확률이 정답 클래스에 해당하는 클래스의 확률을 1로 나타내게 하는것임

<img width="401" alt="image" src="https://user-images.githubusercontent.com/87634136/179701979-89b1cbf3-9f53-4d85-b32f-76811deac29c.png">

-> 위의 식은 다양하게 사용될 수 있다. 

ex) 두 분포 간의 KL divergence or MLE로도 볼 수 있음



eventually, 

정답 클래스에 해당하는 클래스의 확률이 1에 가깝게 계산되는 것이다.

그렇게 되면 Loss는 "-log(정답클래스확률)"이 될 것임



* log -> 단조 증가 함수, log를 최대화시키는 것이 그냥 확률값을 최대화 시키는 것보다 쉽다. SO, log 사용함



log가 단조 증가 함수이기 때문에 정답 클래스인, log P를 최대화 시키는 것은, log P가 높았으면 좋겠다는 것이다.

그런데 손실 함수는 "얼마나 좋은지"가 아니라 "얼마나 안좋은지"를 측정하는 것이기 때문 -> log에 (-)를 붙임



==> SVM의 손실함수는 -log(P(정답클래스))로 나타내게 된다.

<img width="230" alt="image" src="https://user-images.githubusercontent.com/87634136/179702024-7365b75f-489a-4ac5-80db-f2bbe8ae4dfe.png">

-> score가 있으면, softmax를 거치고, 나온 확률 값이 -log를 추가하면 된다.





위에 식에 대한 구체적인 예시 

<img width="376" alt="image" src="https://user-images.githubusercontent.com/87634136/179702097-acd5ccbd-bf73-427f-a205-b84ed25d1b27.png">

Linear Classifier의 출력으로 나온 score가 있다.

이 score는 이전에 SVM Loss에서의 예제와 같은 값임

<img width="392" alt="image" src="https://user-images.githubusercontent.com/87634136/179702118-7deb242c-62a6-4b1a-be94-b3720fdea931.png">

but, 이젠 score 자체를 Loss로 쓰기보다는 score를 지수화 시킨다.

<img width="379" alt="image" src="https://user-images.githubusercontent.com/87634136/179702143-ba98e71e-76bc-4825-a4d2-caf772a7a660.png">

and, 합이 1이 되도록 정규화시켜 준다. 

<img width="404" alt="image" src="https://user-images.githubusercontent.com/87634136/179702184-c5ab4ea7-ad92-4170-a66b-f5222781ad97.png">

and, 정답 score에만 -log를 씌워준다. (= softmax, 다항 로지스틱 회귀(multinomial logistic regression)) 



Q1. softmax loss의 최댓값과 최솟값은 얼마일까 ?

A. loss의 최솟값은 0이고, 최댓값은 무한대

확률 분포를 한번 생각해 보자면, 우리는 정답 클래스의 확률은 1이 되길 원하고 정답이 아닌 클래스는 0이 되길 원한다. 이렇게 생각하면, log 안에 있는 어떤 값은 1이 되어야 한다. 

정답 클래스에 대한 log 확률이기 때문에 log(1)=0이고, -log(1)=0이다.

So, 고양이를 완벽하게 분류했다면 Loss는 0이 된다.



Q. Loss가 0이 되려면 실제 score는 어떤 값이어야 할까 ?

A. 정답 score는 거의 무한대에 가깝게 극단적으로 높아야 한다.

지수화를 하고 정규화를 하기 때문에, 확률 1(정답)과 0(그 외)를 얻으려면, 정답 클래스의 score는 (+)무한대가 되어야 하고, 나머지는 (-)무한대가 되어야 한다.

컴퓨터는 무한대 계산을 잘 못하기 때문에, Loss가 0인 경우는 절대 없음.(유한 정밀도 때문)

but, 이론적으로 보면 0은 "이론적으로 최소 Loss이다." 



최대 손실은 최댓값이 없다.

만약, 정답 클래스의 확률이 0이고, 거기 -log를 취하면 [-log(0)]

log(0)는 음의 무한대가 되고 따라서 -log(0)는 양의 무한대가 된다. (정말 안좋은 경우)



but, 이러한 경우 또한 발생하지는 않을 것이다. 

because, 확률이 0이 되려면 e^syi가 0이 되어야 하는데 그게 가능한 경우는 정답 클래스의 스코어가 음의 무한대일 일 때 뿐이기 떄문이다.



==> "유한 정밀도" 를 가지고는 최댓값(무한대) 최솟값(0)에 도달할 수는 없습니다.



Q. 만약, S가 모두 0 근처에 모여있는 작은 수일 때 Loss는 어떻게 될까 ?

A. -log(1/C) 

log는 분모와 분자를 뒤집을 수 있으니까 -log(1/C)는 log(C)가 된다.

softmax를 사용할 때 첫 번째 interation에서 해볼만한 아주 좋은 디버깅 전략이다.




두 손실함수 비교

<img width="402" alt="image" src="https://user-images.githubusercontent.com/87634136/179702250-fe5b6be7-3983-493c-aac3-f34309801200.png">

<img width="402" alt="image" src="https://user-images.githubusercontent.com/87634136/179702338-e7bb4ff5-f7e2-4f0f-93d2-73d83b688276.png">

<img width="398" alt="image" src="https://user-images.githubusercontent.com/87634136/179702403-615bba27-bc07-4e5c-bb0d-3c32b03d68e8.png">

- 데이터 셋 x, y

- 입력 x로부터 score를 얻기 위해 -> Linear classifier 사용

- softmax, SVM 과 같은 손실함수 이용 => 모델의 예측 값이 정답 값에 비해 얼마나 안좋은지 측정

and, 모델의 복잡함과 단순함을 통제하기 위해 손실함수에 regularization term 추가

and, 그것을 모두 합쳐서, 최종 손실 함수가 최소가 되게 하는 가중치 행렬이자 파라미터인 행렬W를 구하게 되는 것




어떻게 실제 Loss를 줄이는 W를 찾을 수 있는 걸까? ====> "최적화"

<img width="353" alt="image" src="https://user-images.githubusercontent.com/87634136/179702447-a1e24f7f-a99b-43d8-a067-24f3c0b4979a.png">

 예시) 

사람 -> 산, 계곡 등의 골짜기



산, 계곡 => 파라미터 (W)

사람이 있는 곳의 높이 => Loss

(Loss는 W에 따라 변하고, W를 찾아야함)



사람이 해야할 것 -> 골짜기의 밑바닥 찾기

1. 단순한 방법 -> 임의의 탐색(random search)

<img width="393" alt="image" src="https://user-images.githubusercontent.com/87634136/179702517-e0730fda-038d-4505-b3b6-58e7aa41c9cd.png">

->임의로 샘플링한 W들을 많이 모아놓고 Loss를 계산해서 어떤 W가 좋은지 살펴보는 것

(이 방법 사용 X)



2. 지역적인 기하학적 특성을 이용 (local geometry)

- 경사(slope)란 ?

: 어떤 함수에 대한 미분값



ex) 1차원 함수 f -> f(x) = y

x : 입력 / y : (출력) 어떤 커브의 높이

=> 곡선의 일부를 구하면 기울기를 계산 할 수 있음



- 어떤 점(x)에서의 경사인 도함수 계산 ?

<img width="205" alt="image" src="https://user-images.githubusercontent.com/87634136/179702588-9b8113b1-7847-4b43-a358-175542896c1c.png">

 작은스텝 h가 있고, 이 스텝간의 함수차이를 비교해보면 f(x+h) - f(x)

스텝 사이즈를 0으로 만들면 h->0

==> 어떤 점에서의 이 함수의 경사



위의 수식 -> 확장 가능 -> 다변수 함수 (multi-variable functions)

x -> 스칼라X , 벡터 O

벡터이기 때문에 다변수로 확장해야 함



다변수 상황에서의 미분으로 일반화시켜보면 gradient

gradient -> 벡터 x의 각 요소를 편도함수들의 집합

gradient의 모양은 x와 같음 (입력 : 3개 -> gradient : 3개)



gradient의 각 요소가 알려주는 것 -> 우리가 그쪽으로 갈때 함수 f의 경사가 어떤지

gradient -> 편도함수들의 벡터



특정 방향에서 얼마나 가파른지 알고 싶을 때 

-> 그 방향의 유닛 벡터와 gradient 벡터 내적 (dot)



급강하 방향은 음의 경사도이다.





Summary,

수치적인 gradient는 간단하고 그럴듯 하다.

but, 실제로는 사용X



실제 gradient 계산 구현 -> 분석적인 방법

(수치적 gradient도 디버깅 툴로 유용)



but, 수치적인 방법 -> 느림, 부정확 

디버깅에 이용하고 싶으면 파라미터의 스케일 줄이기 (so, 디버깅 시간 단축)





Gradient Descent

<img width="401" alt="image" src="https://user-images.githubusercontent.com/87634136/179702656-841fd462-58a4-40bf-8434-18b63bc16093.png">

1. W를 임의의 값으로 초기화

2. Loss와 gradient를 계산

3. 가중치를 gradient의 반대 방향으로 업데이트

(gradient가 함수에서 증가하는 방향이기 때문에 -gradient를 해야 내려가는 방향이 됨)

--> -gradient 방향으로 아주 조금씩 이동할 것이고, 반복하다 보면 결국에는 수렴



but, 스텝 사이즈는 하이퍼 파라미터 

스텝사이즈는 -gradient 방향으로 얼마나 나아가야 하는지를 알려줌

스텝 사이즈 -> Learning rate라고도 함 

실제 학습을 할때 직접 정해줘야 하는 가장 중요한 하이퍼파라미터 중 하나





2차원 공간의 간단한 예시

<img width="389" alt="image" src="https://user-images.githubusercontent.com/87634136/179702697-12709a95-cd7b-4cd7-94e6-b156382760b4.png">

- 그릇처럼 보이는 것이 손실합수

- 가운데 빨간 부분이 낮은 Loss

- 테두리의 파란영역과 초록 영역은 Loss가 더 높은 곳 (피해야함)



임의의 점에 W를 설정

-> -gradient를 계산할 것이고 이를 통해 결국 가장 낮은 지점에 도달할 것

and, 이 과정 반복 -> 정확한 최저점 도달





stochastic gradient descent (SGD)

<img width="380" alt="image" src="https://user-images.githubusercontent.com/87634136/179702747-a7cbee15-66c7-4f76-b561-c9409acc3842.png">

SGD : 전체 데이터 셋의 gradient와 loss를 계산하기 보다는 Minibatch라는 작은 트레이닝 샘플 집합으로 나눠서 학습하는 것

- Minibatch는 보통 2의 승수로 정하며 32, 64, 128 을 보통 씀

minibatch를 이용 -> Loss의 전체 합의 "추정치"와 실제 gradient의 "추정치"를 계산하는 것

- 거의 모든 DNN 알고리즘에 사용되는 기본적인 학습 알고리즘임

​
