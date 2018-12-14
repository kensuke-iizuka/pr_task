# -*- coding: utf-8 -*-
import sys
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter

# Input from user
K = int( input( " Please input K > " ) )
N = int( input( " Please input N > " ) )
train_num = K
train_img = np.zeros((10,K,28,28), dtype=np.float32)

# Generate K proto types from N images of each number type
for i in range(10):
  for j in range(K):
    temp_file = np.zeros((28,28), dtype=np.float32)
    for k in range(N):
      num = random.randint(1,100)
      train_file = "mnist/train/" + str(i) + "/" + str(i) + "_" + str(num) + ".jpg"
      temp_file += np.asarray(Image.open(train_file).convert('L')).astype(np.float32)
    # Calculate arithmetic mean of each image
    train_img[i][j] = temp_file / N 

## 混合行列
result = np.zeros((10,10), dtype=np.int32)
for i in range(10):
  for j in range(1,101):
    # 未知パターンの読み込み
    pat_file = "mnist/train/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
    pat_img = np.asarray(Image.open(pat_file).convert('L')).astype(np.float32)

    # Nearest Neibhor 
    min_val = float('inf')
    ans = 0
    for k in range(10):
      for l in range(1,train_num+1):
        # Calculate SSD
        t = train_img[k][l-1].flatten()
        p = pat_img.flatten()
        dist = np.dot( (t-p).T , (t-p) )
        # Explore minimum value
        if dist < min_val:
          min_val = dist
          ans = k

    # Print result 
    result[i][ans] +=1
    print( i , j , "->" , ans )

print( "\n [混合行列]" )
print( result )
print( "\n 正解数 ->" ,  np.trace(result) )