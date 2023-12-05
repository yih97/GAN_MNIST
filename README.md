# GAN을 이용한 MNIST 이미지 생성 프로젝트

## 프로젝트 소개
이 프로젝트는 GAN(Generative Adversarial Networks)을 사용하여 MNIST 손글씨 숫자 데이터를 기반으로 새로운 이미지를 생성하는 것을 목표로 합니다. GAN은 생성자(Generator)와 판별자(Discriminator) 두 부분으로 구성된 심층 학습 모델입니다. 생성자는 진짜와 유사한 이미지를 생성하려고 시도하고, 판별자는 이미지가 진짜인지 생성된 것인지를 판별합니다. 이 경쟁적인 과정을 통해 생성자는 점점 더 실제와 유사한 이미지를 만들어냅니다.

## 사용 기술
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## 설치 방법
이 코드를 실행하기 위해서는 Python과 필요한 라이브러리들이 설치되어 있어야 합니다. 다음 명령어를 통해 필요한 라이브러리들을 설치할 수 있습니다:

## 사용 방법
1. 코드를 다운로드하고 프로젝트 폴더로 이동합니다.
2. `main.py` 파일을 실행합니다. 이 파일은 GAN 모델을 훈련시키고, 생성된 이미지를 `gan_images` 폴더에 저장합니다.
3. `gan_images` 폴더를 확인하여 생성된 이미지를 볼 수 있습니다.

## 코드 구조

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os

# 데이터 로드 및 전처리
# MNIST 데이터셋을 불러오고, 이미지들을 신경망에 맞는 형태로 변환합니다.
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255

# 생성자 모델 정의
# 100차원의 랜덤 노이즈를 받아 28x28 크기의 이미지를 생성합니다.
generator = Sequential([
    Dense(256, activation='relu', input_shape=(100,)),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# 판별자 모델 정의
# 생성된 이미지 또는 실제 이미지를 받아, 그것이 실제인지 가짜인지를 판별합니다.
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 판별자 모델 컴파일
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

# GAN 모델 정의 및 컴파일
# 생성자와 판별자를 결합하여 GAN 모델을 구성합니다.
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')

# 이미지 저장을 위한 폴더 생성
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

# 훈련 과정
epochs = 10000
batch_size = 32

for epoch in range(epochs):
    # 임의의 노이즈에서 이미지 생성
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise)

    # 실제 이미지와 결합
    real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
    x = np.concatenate([real_images, generated_images.reshape(batch_size, 28, 28)], axis=0)

    # 레이블 생성
    y = np.zeros(2 * batch_size)
    y[:batch_size] = 1

    # 판별자 훈련
    discriminator.trainable = True
    discriminator.train_on_batch(x, y)

    # 생성자 훈련
    noise = np.random.normal(0, 1, (batch_size, 100))
    y2 = np.ones(batch_size)
    discriminator.trainable = False
    gan.train_on_batch(noise, y2)

    # 에포크마다 이미지 저장
    if epoch % 100 == 0:
        noise = np.random.normal(0, 1, (10, 100))
        generated_images = generator.predict(noise)
        generated_images = generated_images * 255
        generated_images = generated_images.astype('uint8')

        fig, axs = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            axs[i].imshow(generated_images[i], cmap='gray')
            axs[i].axis('off')

        fig.savefig("gan_images/generated_{}.png".format(epoch))
        plt.close()
```

## 기여 방법
이 프로젝트에 기여하고 싶은 개발자들은 GitHub를 통해 Pull Request를 보내거나 Issue를 오픈할 수 있습니다. 모든 기여와 피드백은 환영합니다.

## 라이선스
이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 저자
- [윤이현]
