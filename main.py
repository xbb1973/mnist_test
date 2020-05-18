import numpy as np
import tensorflow as tf
from mnist import module as model

from cnocr import CnOcr
import mxnet as mx
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
import datetime
import random
import os
import base64

# import os


class Pic_str:
    def create_uuid(self): #生成唯一的图片的名称字符串，防止图片显示时的重名问题
        nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
        randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
        if randomNum <= 10:
            randomNum = str(0) + str(randomNum);
        uniqueNum = str(nowTime) + str(randomNum);
        return uniqueNum;



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# # 0.9表示可以使用GPU 90%的资源进行训练，可以任意修改
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto(allow_soft_placement=True)

# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

x = tf.placeholder("float", [None, 784])
sess = tf.Session(config=config)


with tf.variable_scope("regression"):
    print(model.regression(x))
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
regression_file = tf.train.latest_checkpoint("mnist/data/regreesion.ckpt")
if regression_file is not None:
    saver.restore(sess, regression_file)

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(variables)
convolutional_file = tf.train.latest_checkpoint(
    "mnist/data/convolutional.ckpt")
if convolutional_file is not None:
    saver.restore(sess, convolutional_file)


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(
        y2, feed_dict={
            x: input,
            keep_prob: 1.0
        }).flatten().tolist()


app = Flask(__name__)
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

        ocr = CnOcr()
        img_fp = os.path.join(file_dir, new_filename)
        img = mx.image.imread(img_fp, 1)
        res = ocr.ocr(img)
        print("Predicted Chars:", res)

        return jsonify({"success": 0, "msg": res})
        # return jsonify({"success": 0, "msg": "上传成功"})
    else:
        return jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass


# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

# 路由
@app.route("/api/mnist", methods=['post'])
def mnist():

    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(
        1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


@app.route("/")
def main():
    return render_template("index.html")


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
