# -*- coding: utf-8 -*-

# ステレオマッチング
#
# > pip install numpy
# > pip install pillow

import sys
import os
import numpy as np
from PIL import Image
import pdb

# 左画像の読み込み
left_file = "left1.jpg"
left_img = Image.open(left_file).convert('L')

# numpyに変換 -> (Y,X,channel)
left = np.asarray(left_img).astype(np.float32)
print("left shape: ", left.shape )
left_width = left.shape[1]
left_height = left.shape[0]

# 右画像の読み込み
right_file = "right1.jpg"
right_img = Image.open(right_file).convert('L')

# numpyに変換 -> (Y,X,channel)
right = np.asarray(right_img).astype(np.float32)
print("right shape: ", right.shape )
right_width = right.shape[1]
right_height = right.shape[0]

# 視差マップ
result = np.ones((left_height, left_width))
# 探索領域の大きさ
search_size = 21 // 2
# テンプレートの大きさ
template_size = 21 // 2 
for y in range(template_size-1,left_height-template_size,1):
#   print(y)
  for x in range(template_size-1,left_width-template_size,1):
    ans_x = 0
    ans_y = 0
    min_val = float( 'inf' )
    # 左画像の座標(x,y）を中心としたテンプレートに類似した領域を
    # 右画像から検索し，その座標（ans_x,ans_y）を求めなさい
    left_reg = left[y-template_size+1:y+template_size, x-template_size+1:x+template_size].flatten()
    for gy in range(0, search_size*2, 1):
      for gx in range(0, search_size*2, 1):
        right_reg = right[y+gy-search_size-1:y+gy+search_size, x+gx-search_size-1:x+gx+search_size].flatten()
        # pdb.set_trace()
        # if x == 611:
        if left_reg.shape != right_reg.shape:
          print(gy,gx,y,x)
          print("left:",left_reg.shape)
          print("right",right_reg.shape)
        # sum = np.dot((left_reg-right_reg).T, (left_reg-right_reg))
        sum = 0
        if min_val > sum:
          min_val = sum
          ans_x = x
          ans_y = y
            
    result[y,x]=(x-ans_x)*(x-ans_x)+(y-ans_y)*(y-ans_y)

min = np.min( result )
max = np.max( result )
result = (result-min)/(max-min) * 255
result_img = Image.fromarray(np.uint8(result))
