# -*- coding: utf-8 -*-

# 最近傍法

import sys
import os
import numpy as np
from numpy import random
from PIL import Image
from collections import Counter

# プロトタイプを格納する配列
train_num = 100
train_img = np.zeros((10,train_num,28,28), dtype=np.float32)

# ユーザーの入力
K = int( input( " Please input K > " ) )

# プロトタイプの読み込み
for i in range(10):
  for j in range(1,train_num+1):
    train_file = "../data/mnist/train/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
    train_img[i][j-1] = np.asarray(Image.open(train_file).convert('L')).astype(np.float32)

# 混合行列
result = np.zeros((10,10), dtype=np.int32)
for i in range(10):
  for j in range(1,101):
    # 未知パターンの読み込み
    pat_file = "../data/mnist/test/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
    pat_img = np.asarray(Image.open(pat_file).convert('L')).astype(np.float32)

    min_val = np.full(K, np.inf)
    ans = np.zeros(K)
    # k最近傍法
    for k in range(10):
      for l in range(1,train_num+1):
        # SSDの計算
        t = train_img[k][l-1].flatten()
        p = pat_img.flatten()
        dist = np.dot( (t-p).T , (t-p) )
        # 最小値の探索
        for m in range(K):
          if dist < min_val[m]:
            min_val[m] = dist
            ans[m] = int(k)
            break
          else:
            continue
  print(ans)
  c = Counter(ans)
  mode = c.most_common(1)
  # count = np.array([v for key,v in sorted(c.items())])
  # print(count)
  final_ans = int(mode[0][0])
  print(final_ans)
  #  結果の出力
  result[i][final_ans] +=1
  print( i , j , "->" , ans, final_ans )
  
print( "\n [ 混合行列 ]" )
print( result )
print( "\n 正解数 -> " , np.trace(result) )

