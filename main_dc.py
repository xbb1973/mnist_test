import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import datetime
import random

from crnn.model import CRNN
from crnn import Config

import torch
import torch.nn.functional as F
from torchvision import transforms


app = Flask(__name__)

# use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

# global variable
model = CRNN(Config.num_classes)


# 读取字典文件 用来翻译结果
fp = open('./crnn/Idx2Char.txt', 'r', encoding='utf-8-sig')
dictionary = fp.read()
fp.close()
char_dict = eval(dictionary)


# 根据字典将序号转换为文字
def Idx2Word(labels, dict=char_dict):
    texts = []
    for label in labels[0]:
        texts.append(dict[label])
    return texts


# 贪心策略解码函数，解码预测标签
def ctc_greedy_decoder(outputs, ctc_blank=Config.ctc_blank):
    output_argmax = outputs.permute(1, 0, 2).argmax(dim=-1)
    output_argmax = output_argmax.cpu().numpy()
    # assert output_argmax.shape = (batch_size, int(sequence_length[0]))
    output_labels = output_argmax.tolist()
    pred_labels = []

    # 删除ctc_blank
    for label in output_labels:
        pred_label= []
        preNum = label[0]
        for curNum in label[1: ]:
            if preNum == ctc_blank:
                pass
            elif curNum == preNum:
                pass
            else:
                pred_label.append(preNum)
            preNum = curNum
        if preNum != ctc_blank:
            pred_label.append(preNum)
        pred_labels.append(pred_label)

    return pred_labels


def load_model():
    """
    Load the pre-trained model.

    """
    global model
    # model = resnet50(pretrained=True)
    # 读取参数
    if os.path.exists('crnn/checkpoint.pth.tar'):
        checkpoint = torch.load('crnn/checkpoint.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('model has restored')

    model.eval()
    model = model.to(device)


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    return cv_img


def prepare_image(image):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """
    # # 将图片读取进来
    # image = cv_imread(image_path)
    
    # 改变图片大小以适应网络结构要求
    scale = float(Config.height / image.shape[0])
    # 将彩色图片转化为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    image = transform(image)

    # add batch_size axis.
    image = image.unsqueeze(0)
    image = image.to(device)

    return image


class Pic_str:
    def create_uuid(self): #生成唯一的图片的名称字符串，防止图片显示时的重名问题
        nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
        randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
        if randomNum <= 10:
            randomNum = str(0) + str(randomNum)
        uniqueNum = str(nowTime) + str(randomNum)
        return uniqueNum


app.config['JSON_AS_ASCII'] = False

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('up.html')


# 上传文件
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['photo']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        print(fname)
        ext = fname.rsplit('.', 1)[1]
        new_filename = Pic_str().create_uuid() + '.' + ext
        f.save(os.path.join(file_dir, new_filename))

        # ocr = CnOcr()
        img_fp = os.path.join(file_dir, new_filename)
        # img = mx.image.imread(img_fp, 1)
        # res = ocr.ocr(img)
        # print("Predicted Chars:", res)
        # 将图片读取进来
        image = cv_imread(img_fp)

        # Preprocess the image and prepare it for classification.
        image = prepare_image(image)

        # 读取模型
        load_model()
        # infer
        # 将图片数据输入模型
        output = model(image)
        output_log_softmax = F.log_softmax(output, dim=-1)
        
        # 对结果进行解码
        pred_labels = ctc_greedy_decoder(output_log_softmax)
        pred_texts = ''.join(Idx2Word(pred_labels))

        print('predict result: {}\n'.format(pred_texts))
        return jsonify({"success": 1, "msg": pred_texts})
        # return img_fp
        # return jsonify({"success": 0, "msg": "上传成功"})
    else:
        return jsonify({"error": 1001, "msg": "上传失败"})


# # 预测
# @app.route("/predict", methods=["POST"])
# def predict():
#     # FIXME: 
#     # Initialize the data dictionary that will be returned from the view.
#     data = {"success": False}

#     # Ensure an image was properly uploaded to our endpoint.
#     if request.method == 'POST':
#         if request.files.get("image"):
#             # Read the image in PIL format
#             image = request.files["image"]
#             image = PIL.Image.open(io.BytesIO(image))

#             # TODO: image_path
#             # FIXME
#             # BUG
#             # image_path = os.path.join(IMG_ROOT, image.filename)
#             # image_path = image.filename
#             # 将图片读取进来
#             image = cv_imread(image_path)

#             # Preprocess the image and prepare it for classification.
#             image = prepare_image(image)

#             # 读取模型
#             # load_model() 
#             # infer
#             # 将图片数据输入模型
#             output = model(image)
#             output_log_softmax = F.log_softmax(output, dim=-1)
            
#             # 对结果进行解码
#             pred_labels = ctc_greedy_decoder(output_log_softmax)
#             pred_texts = ''.join(Idx2Word(pred_labels))
#             # print('predict result: {}\n'.format(pred_texts))
            
#             # FIXME:
#             data['predictions'] = list()
#             data['predictions'].append({'result': pred_texts})
            
#             # Indicate that the request was a success.
#             data["success"] = True

#     # Return the data dictionary as a JSON response.
#     return jsonify(data)


# # 路由
# @app.route("/api/mnist", methods=['post'])
# def mnist():
#     input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(
#         1, 784)
#     output1 = regression(input)
#     output2 = convolutional(input)
#     return jsonify(results=[output1, output2])


# @app.route('api/CRNN', methods=['post'])
# def infer():
#     input
#     output = model(input)
#     return jsonify(results=output)


@app.route("/")
def main():
    return render_template("index.html")


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8000)