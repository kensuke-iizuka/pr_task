import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb

size = 32
feature = size * size

data_vec = np.zeros((3, 1024), dtype=np.float64)
convert_vec = np.zeros((3, 1024), dtype=np.float64)
dir = [ "train" , "test" ]
class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
i = 1
flag = 1
j = 1
train_file = "../data/cifar-10/" + dir[ flag ] +  "/" + class_list[i] + "/" + str(j) + ".png"
work_img = Image.open(train_file).convert('RGB')
resize_img = work_img.resize((size, size))
work_array = np.resize(np.asarray(resize_img).astype(np.float64), (size, size, 3))
r_image = np.zeros((3, 1024))
g_image = np.zeros((3, 1024))
b_image = np.zeros((3, 1024))
r_image[0] = work_array[:,:,0].flatten()
g_image[1] = work_array[:,:,1].flatten()
b_image[2] = work_array[:,:,2].flatten()
data_vec[0] = work_array[:,:,0].flatten() # Red image
data_vec[1] = work_array[:,:,1].flatten() # Green image
data_vec[2] = work_array[:,:,2].flatten() # Blue image
# 入力値の合計を1とする (これ実行すると画像が黒く塗りつぶされてしまう...)

# convert_vec[0] = data_vec[0]
# convert_vec[1] = data_vec[1]
# convert_vec[2] = data_vec[2]
convert_vec[0] = data_vec[0] / np.sum( data_vec[0] )
convert_vec[1] = data_vec[1] / np.sum( data_vec[1] )
convert_vec[2] = data_vec[2] / np.sum( data_vec[2] )
# convert_vec[0] = data_vec[0] * np.sum( data_vec[0] )
# convert_vec[1] = data_vec[1] * np.sum( data_vec[1] )
# convert_vec[2] = data_vec[2] * np.sum( data_vec[2] )
convert_vec[0] = (convert_vec[0] / convert_vec[0].max() * 255).astype(np.uint8)
convert_vec[1] = (convert_vec[1] / convert_vec[1].max() * 255).astype(np.uint8)
convert_vec[2] = (convert_vec[2] / convert_vec[2].max() * 255).astype(np.uint8)

print("Arraies are equal?", np.array_equal(data_vec, convert_vec))

orig_img_vec = np.resize(data_vec, (3, size, size))
orig_img_vec = np.transpose(orig_img_vec, (1, 2, 0))
orig_img = Image.fromarray(np.uint8(orig_img_vec))

r_img_vec = np.resize(r_image, (3, size, size))
r_img_vec = np.transpose(r_img_vec, (1, 2, 0))
r_img = Image.fromarray(np.uint8(r_img_vec))

g_img_vec = np.resize(g_image, (3, size, size))
g_img_vec = np.transpose(g_img_vec, (1, 2, 0))
g_img = Image.fromarray(np.uint8(g_img_vec))

b_img_vec = np.resize(b_image, (3, size, size))
b_img_vec = np.transpose(b_img_vec, (1, 2, 0))
b_img = Image.fromarray(np.uint8(b_img_vec))

conv_img_vec = np.resize(convert_vec, (3, size, size))
conv_img_vec = np.transpose(conv_img_vec, (1, 2, 0))
conv_img = Image.fromarray(np.uint8(conv_img_vec))


plt.figure(figsize=(1,5))
plt.subplot(1,5,1)
plt.imshow(np.reshape(orig_img,(size,size,3)))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.subplot(1,5,2)
plt.imshow(np.reshape(conv_img,(size,size,3)))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.subplot(1,5,3)
plt.imshow(np.reshape(r_img,(size,size,3)))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.subplot(1,5,4)
plt.imshow(np.reshape(g_img,(size,size,3)))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.subplot(1,5,5)
plt.imshow(np.reshape(b_img,(size,size,3)))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
file = "fig/show-data-" + class_list[i] + "-" + str(j) + "-result.png"
plt.savefig(file)
plt.close()
