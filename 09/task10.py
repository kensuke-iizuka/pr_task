# -*- coding: utf-8 -*-
# Auto Encoder(Cifar 10)
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb
# クラス数
class_num = 10
# 画像の大きさ
size = 32
feature = size * size
# 学習データ数
train_num = 200 
# データ
data_vec = np.zeros((class_num, train_num, 3, feature), dtype=np.float64)
# 学習係数
alpha = 0.1

# シグモイド関数
def Sigmoid( x ):
  return 1 / ( 1 + np.exp(-x) )

# シグモイド関数の微分
def Sigmoid_( x ):
  return ( 1-Sigmoid(x) ) * Sigmoid(x)

# ReLU関数
def ReLU( x ):
  return np.maximum( 0, x )

# ReLU関数の微分
def ReLU_( x ):
  return np.where( x > 0, 1, 0 )

# ソフトマックス関数
def Softmax( x ):
  return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

# 出力層
class Outunit:
  def __init__(self, m, n):
    # 重み
    self.w = np.random.uniform(-0.5,0.5,(m,n))
    # 閾値
    self.b = np.random.uniform(-0.5,0.5,n)

  def Propagation(self, x):
    self.x = x
    # 内部状態
    self.u = np.dot(self.x, self.w) + self.b
    # 出力値（恒等関数）
    self.out = self.u
    self.out = Sigmoid(self.u)

  def Error(self, t):
    # 誤差
    f_ = Sigmoid_(self.u) 
    delta = ( self.out - t ) * f_
    # 重み，閾値の修正値
    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    # 前の層に伝播する誤差
    self.error = np.dot(delta, self.w.T) 

  def Update_weight(self):
    # 重み，閾値の修正
    self.w -= alpha * self.grad_w
    self.b -= alpha * self.grad_b

  def Save(self, filename):
    # 重み，閾値の保存
    np.savez(filename, w=self.w, b=self.b)
      
  def Load(self, filename):
    # 重み，閾値のロード
    work = np.load(filename)
    self.w = work['w']
    self.b = work['b']

# 中間層
class Hunit:
  def __init__(self, m, n):
    # 重み
    self.w = np.random.uniform(-0.5,0.5,(m,n))
    # 閾値
    self.b = np.random.uniform(-0.5,0.5,n)
      
  def Propagation(self, x):
    self.x = x
    # 内部状態
    self.u = np.dot(self.x, self.w) + self.b
    # 出力値（ソフトマックス関数）
    # self.out = Softmax( self.u )
    self.out = Sigmoid( self.u )

  def Error(self, p_error):
    # 誤差
    f_ = Sigmoid_( self.u )
    delta = p_error * f_
    # 重み，閾値の修正値
    self.grad_w = np.dot(self.x.T, delta)
    self.grad_b = np.sum(delta, axis=0)
    # 前の層に伝播する誤差
    self.error = np.dot(delta, self.w.T) 

  def Update_weight(self):
    # 重み，閾値の修正
    self.w -= alpha * self.grad_w
    self.b -= alpha * self.grad_b

  def Save(self, filename):
    # 重み，閾値の保存
    np.savez(filename, w=self.w, b=self.b)

  def Load(self, filename):
    # 重み，閾値のロード
    work = np.load(filename)
    self.w = work['w']
    self.b = work['b']


# データの読み込み
def Read_data( flag ):
  dir = [ "train" , "test" ]
  class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
  for i in range(class_num):
    for j in range(train_num):
      # グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
      train_file = "../data/cifar-10/" + dir[ flag ] +  "/" + class_list[i] + "/" + str(j) + ".png"
      work_img = Image.open(train_file).convert('RGB')
      resize_img = work_img.resize((size, size))
      work_array = np.resize(np.asarray(resize_img).astype(np.float64), (size, size, 3))
      data_vec[i][j][0] = work_array[:,:,0].flatten() # Red image
      data_vec[i][j][1] = work_array[:,:,1].flatten() # Green image
      data_vec[i][j][2] = work_array[:,:,2].flatten() # Blue image
      # 入力値の合計を1とする
      data_vec[i][j][0] = data_vec[i][j][0] / 255 
      data_vec[i][j][1] = data_vec[i][j][1] / 255 
      data_vec[i][j][2] = data_vec[i][j][2] / 255 

# Visualize weights
def Visualize_weights(hunit, img_name):
  g_size = 10
  plt.figure(figsize=(g_size,g_size))
  
  count = 1
  for i in range(100):
    plt.subplot(g_size,g_size,count)
    plt.imshow(np.reshape(hunit.w[:,i],(size,size)),cmap='gray')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    count += 1
  file = "fig/" + img_name + "-weight.png"
  plt.savefig(file)
  plt.close()

# Train 
def Train():
  # エポック数
  epoch = 1000
  for e in range( epoch ):
    r_error = 0.0
    g_error = 0.0
    b_error = 0.0
    for i in range(class_num):
      for j in range(0,train_num):
        rnd_c = np.random.randint(class_num)
        rnd_n = np.random.randint(train_num)
        # Input data
        r_input_data = data_vec[rnd_c][rnd_n][0].reshape(1,feature)
        g_input_data = data_vec[rnd_c][rnd_n][1].reshape(1,feature)
        b_input_data = data_vec[rnd_c][rnd_n][2].reshape(1,feature)
        # Propagation 
        r_hunit.Propagation( r_input_data )
        r_outunit.Propagation( r_hunit.out )
        g_hunit.Propagation( g_input_data )
        g_outunit.Propagation( g_hunit.out )
        b_hunit.Propagation( b_input_data )
        b_outunit.Propagation( b_hunit.out )
        # teacher signal 
        r_teach = data_vec[rnd_c][rnd_n][0].reshape(1,feature)
        g_teach = data_vec[rnd_c][rnd_n][1].reshape(1,feature)
        b_teach = data_vec[rnd_c][rnd_n][2].reshape(1,feature)
        # Caliculate error 
        r_outunit.Error( r_teach )
        r_hunit.Error( r_outunit.error )
        g_outunit.Error( g_teach )
        g_hunit.Error( g_outunit.error )
        b_outunit.Error( b_teach )
        b_hunit.Error( b_outunit.error )
        # Update weight
        r_outunit.Update_weight()
        r_hunit.Update_weight()
        g_outunit.Update_weight()
        g_hunit.Update_weight()
        b_outunit.Update_weight()
        b_hunit.Update_weight()

        r_error += np.dot( ( r_outunit.out - r_teach ) , ( r_outunit.out - r_teach ).T )
        g_error += np.dot( ( g_outunit.out - g_teach ) , ( g_outunit.out - g_teach ).T )
        b_error += np.dot( ( b_outunit.out - b_teach ) , ( b_outunit.out - b_teach ).T )
    print( e , "R: ->" , r_error , "G: ->" , b_error, "B: ->" , g_error)

  # Save weights
  r_outunit.Save( "dat/task10-r-out.npz" )
  r_hunit.Save( "dat/task10-r-hunit.npz" )
  g_outunit.Save( "dat/task10-g-out.npz" )
  g_hunit.Save( "dat/task10-g-hunit.npz" )
  b_outunit.Save( "dat/task10-b-out.npz" )
  b_hunit.Save( "dat/task10-b-hunit.npz" )


def Predict():
  class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
  # Load weights
  r_outunit.Load( "dat/task10-r-out.npz" )
  r_hunit.Load( "dat/task10-r-hunit.npz" )
  g_outunit.Load( "dat/task10-g-out.npz" )
  g_hunit.Load( "dat/task10-g-hunit.npz" )
  b_outunit.Load( "dat/task10-b-out.npz" )
  b_hunit.Load( "dat/task10-b-hunit.npz" )
  # 混合行列
  result = np.zeros((class_num,class_num), dtype=np.int32)
  
  for i in range(class_num):
    for j in range(0,train_num):
      # Input 
      r_input_data = data_vec[i][j][0].reshape(1,feature)
      g_input_data = data_vec[i][j][1].reshape(1,feature)
      b_input_data = data_vec[i][j][2].reshape(1,feature)
      # Propagation
      r_hunit.Propagation( r_input_data )
      r_outunit.Propagation( r_hunit.out )
      g_hunit.Propagation( g_input_data )
      g_outunit.Propagation( g_hunit.out )
      b_hunit.Propagation( b_input_data )
      b_outunit.Propagation( b_hunit.out )
      # teacher signal
    #   teach = data_vec[i][j].reshape(1,feature)
      # Convert numpy to image(RGB)
      output = np.zeros((3, feature), dtype=np.float64)
      output[0] = (r_outunit.out / r_outunit.out.max() * 255).astype(np.uint8)
      output[1] = (g_outunit.out / g_outunit.out.max() * 255).astype(np.uint8)
      output[2] = (b_outunit.out / b_outunit.out.max() * 255).astype(np.uint8)

      img_vec = np.resize(output, (3, size, size))
      img_vec = np.transpose(img_vec, (1, 2, 0))
      img_vec = (img_vec / img_vec.max() * 255).astype(np.uint8)
      output_img = Image.fromarray(np.uint8(img_vec))


      orig_img_vec = np.resize(data_vec[i][j], (3, size, size))
      orig_img_vec = np.transpose(orig_img_vec, (1, 2, 0))
      orig_img_vec = (orig_img_vec / orig_img_vec.max() * 255).astype(np.uint8)
      orig_img = Image.fromarray(np.uint8(orig_img_vec))

      # pdb.set_trace()
      
      if j < 1:
        # 画像の描画
        plt.figure()
        # 元画像の表示
        plt.subplot(1,2,1)
        plt.imshow(np.reshape(orig_img,(size,size,3)))
        plt.title( "Original Image" )
        # 復元画像の表示
        plt.subplot(1,2,2)
        plt.imshow(np.reshape(output_img,(size,size,3)))
        # 画像の保存
        plt.title( "Decode Image(" + class_list[i] + "," + str(j) + ")" )
        file = "fig/decode-" + class_list[i] + "-" + str(j) + "-result.png"
        plt.savefig(file)
        plt.close()

  Visualize_weights(r_hunit, "r_hunit")
  Visualize_weights(g_hunit, "g_hunit")
  Visualize_weights(b_hunit, "b_hunit")
        

if __name__ == '__main__':

  hunit_num = 100 
  # R channel
  r_hunit = Hunit( feature , hunit_num )
  r_outunit = Outunit( hunit_num , feature )
  # G channel
  g_hunit = Hunit( feature , hunit_num )
  g_outunit = Outunit( hunit_num , feature )
  # B channel
  b_hunit = Hunit( feature , hunit_num )
  b_outunit = Outunit( hunit_num , feature )

  argvs = sys.argv

  # 引数がtの場合
  if argvs[1] == "t":
    # 学習データの読み込み
    flag = 0
    Read_data( flag )
    # 学習
    Train()
  # 引数がpの場合
  elif argvs[1] == "p":
    os.system( "rm -f fig\*.png" )
    # テストデータの読み込み
    flag = 1
    Read_data( flag )
    # テストデータの予測
    Predict()
