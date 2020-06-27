# -*- coding: UTF-8 -*- 
width = 400
height = 32
image_shape = [height, width]
channels = 1
max_text_len = 12  # fix to 10 chars
num_classes = 3756  # 6128 + blank
ctc_blank = num_classes - 1

batch_size = 360
epochs = 6

# TEST_IMG_ROOT = './test_images'      # the root of images that you want to infer