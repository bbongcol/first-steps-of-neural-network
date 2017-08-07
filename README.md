# 신경망 첫걸음 Python Code

책 "신경망 첫걸음"의 Python Code입니다.

책에서는 iPython notebook으로 코드를 작성하고 있고, 저자의 [Github](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)에서도 iPython으로 코드가 업로드 되어 있는데, 책 스터디 할때는 python3로 작성해서 테스트 해봤습니다.

책은 신경망에 대한 기본적인 이론과 이 이론을 바탕으로 간단한 3 Laye를 구성해 보고, MNIST를 실습해 보는 내용입니다. 책 제목 그대로 신경망을 이해하기 위한 첫걸음에 좋은 책인것 같습니다.

신경망 구성 내용은 아래와 같습니다.
- 3 Layer Neural Network : Input(784) - Hidden(200) - Output(10)
- Activation Function : Sigmoid
- Deavtivation Function : Logit
- Learning Rate : 0.01
- Additional training : rotating each original by +/- 10 degrees

신경망 학습
- Mnist 학습 Data : 60000
- 주기(epoch) : 10

실행 결과
- Mnist 테스트 Data : 10000
- 정답 비율 출력
- 0에 대한 Network backwards 이미지 출력

performance =  0.9766 total 10000