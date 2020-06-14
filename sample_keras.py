import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


if __name__ == '__main__':
  np.random.seed(123)
  tf.random.set_seed(123) #tensorflow用の乱数シード

  # 1.データの準備
  
  N = 300
  ## sklearnライブラリによるデータ生成 (月のような形のグラフデータが生成される。)
  x, t = datasets.make_moons(N, noise=0.3)

  # print(x.shape) #: (300,2)
  # print(t.shape) #: (300,)

  t = t.reshape(N,1)
  # print(t.shape) #: (300,1)
  x_train,x_test,t_train,t_test = train_test_split(x,t, test_size=0.2)

  # 2.モデルの構築
  model = Sequential()
  model.add(Dense(3,activation='sigmoid'))
  model.add(Dense(1,activation='sigmoid'))

  # 3.モデルの学習

  ##最適化アルゴリズムを指定。(SGD: 確率的勾配降下法)
  optimizer = optimizers.SGD(learning_rate=0.1)
  ## model.compileで設定する。# metrics: 評価関数を指定。(accuracyは正解率)
  model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

  ## verbose: ログの出力を指定。デフォルトが1.
  model.fit(x_train,t_train, epochs=100, batch_size=10, verbose=1)

  # 4.モデルの評価

  ## モデルの評価の場合:                      model.evaluate()
  ## モデルの用いてテストデータの出力を表示:   model.predict()
  loss,acc = model.evaluate(x_test,t_test,verbose=0)    # model.evaluate() モデルの評価
  print("")
  print("test_loss: {:.3f}, test_acc: {:.3f}".format(loss,acc))