import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort, url_for
from mnist import module as model
from flask_json import as_json
from cnocr import CnOcr
import mxnet as mx
from werkzeug.utils import secure_filename
# from time import strftime
import datetime
import random
import shutil
import os
import cv2
import base64
import json
# 异步任务
from concurrent.futures import ThreadPoolExecutor

# 模型
from crnn.model import DenseNet_BLSTM_CTC_MODEL
from crnn import Config
from crnn import dictionary
import torch
import torch.nn.functional as F
from torchvision import transforms

executor = ThreadPoolExecutor(max_workers=2)

from task import long_task
from main_task import conv_fc

from celery import Celery

app = Flask(__name__)
# 配置
# from kombu import Exchange, Queue
# default_exchange = Exchange('default', type='direct')
# concurrency_exchange = Exchange('concurrency', type='direct')
#
# app.config.task_queues = (
#     Queue('default', default_exchange, routing_key='default'),
#     Queue('concurrency', concurrency_exchange, routing_key='concurrency'),
# )
# app.config.task_default_queue = 'default'
# app.config.task_default_exchange = 'default'
# app.config.task_default_routing_key = 'default'

CELERY_TASK_RESULT_EXPIRES = 60 * 60 * 24  # 任务过期时间
app.config['CELERY_TASK_RESULT_EXPIRES'] = CELERY_TASK_RESULT_EXPIRES
# 配置消息代理的路径，如果是在远程服务器上，则配置远程服务器中redis的URL
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# 要存储 Celery 任务的状态或运行结果时就必须要配置
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# celery worker的并发数，默认是服务器的内核数目,也是命令行-c参数指定的数目
app.config['CELERYD_CONCURRENCY'] = 12
# celery worker 每次去rabbitmq预取任务的数量
app.config['CELERYD_PREFETCH_MULTIPLIER'] = 4

# 初始化Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# 将Flask中的配置直接传递给Celery
celery.conf.update(app.config)

# mysql
import pymysql
from DBUtils.PooledDB import PooledDB

# 建立数据库连接池
dbPool = PooledDB(pymysql, 5, host='127.0.0.1', user='root', passwd='123', db='hwr', port=3306)  # 5为连接池里的最少连接数

# 本地主机监控
import psutil


def getLocalInfo():
    mem = psutil.virtual_memory()
    mem_total = mem.total  # 总内存,单位为byte
    mem_available = mem.available
    mem_percent = int(round(mem.percent))
    mem_used = mem.used
    cpu = int(round(psutil.cpu_percent(1)))

    data_dict = {
        'mem_total': mem_total,
        'mem_used': mem_used,
        'mem_available': mem_available,
        'mem_percent': mem_percent,
        'cpu': cpu
    }
    return data_dict


# use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

# global variable
model = DenseNet_BLSTM_CTC_MODEL(num_classes=Config.num_classes)

# 读取字典文件 用来翻译结果
inverse_char_dict = dictionary.char_dict
char_dict = dict([val,key] for key,val in inverse_char_dict.items())


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
        pred_label = []
        preNum = label[0]
        for curNum in label[1:]:
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
    def create_uuid(self):  # 生成唯一的图片的名称字符串，防止图片显示时的重名问题
        nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
        randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
        if randomNum <= 10:
            randomNum = str(0) + str(randomNum);
        uniqueNum = str(nowTime) + str(randomNum);
        return uniqueNum;


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 推理时的文件路径
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

# 训练时的文件路径
# 进行验证时上传图片的路径
DATA_UPLOAD_FOLDER = 'static/data/upload/'
# 数据源储存路径，存储所有数据，内部按数据源Id分文件夹,
DATA_SET_FOLDER = 'static/data/images/'
# CKPT 保存路径，内部按Train id分文件夹
CKPT_FOLDER = 'static/data/ckpt'
app.config['DATA_UPLOAD_FOLDER'] = DATA_UPLOAD_FOLDER
app.config['DATA_SET_FOLDER'] = DATA_SET_FOLDER
app.config['CKPT_FOLDER'] = CKPT_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

FORBIDDEN_LABEL = ['/', '.', '*', '@', '-', '&', '?', '=', ' ']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def allowed_label(label):
    for i in FORBIDDEN_LABEL:
        if i in label:
            return False
    return True


# @app.route('/upload')
# def upload_test():
#     return render_template('up.html')


## 上传 文件 前后端已经联通(可以实现批量上传,只是前端禁用了批量上传, 这里逻辑是单个和批量上传都可以) by hly
# 上传文件
@app.route('/validate', methods=['POST'], strict_slashes=False)
def upload_validate():
    file_dir = os.path.join(basedir, app.config['DATA_UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    choose_ckpt = request.values.get("ckpt")
    print(choose_ckpt)
    f_list = request.files.getlist('photo')
    for f in f_list:
        print("filename:" + f.filename)
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            print(fname)
            ext = fname.rsplit('.', 1)[1]
            new_filename = Pic_str().create_uuid() + '.' + ext
            f.save(os.path.join(file_dir, new_filename))

            # ocr = CnOcr()
            # img_fp = os.path.join(file_dir, new_filename)
            # img = mx.image.imread(img_fp, 1)
            # res = ocr.ocr(img)
            # print("Predicted Chars:", res)

            # return jsonify({"success": 0, "msg": res})
            # return jsonify({"success": 0, "msg": "success"})
            # return jsonify({"success": 0, "msg": "上传成功"})
        # else:
        # return jsonify({"error": 1001, "msg": "上传失败"})
    return jsonify({"success": 0, "msg": "洒水大"})


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


###############################################################################
## 开始训练
# 待完善
@app.route("/start_train", methods=['POST'])
def start_train():
    #  dataSourceId是必须存在的,  根据dataSourceId 得到训练的数据集Id
    #  而 ckptId可以没有(前端传来<0，说明不要ckpt)，不需要ckpt就是从新开始，如果有就是从这个ckpt开始训练。
    dataset_id = request.values.get("train_dataset_id")
    ckpt_id = request.values.get("train_ckpt_id")
    train_modal = request.values.get("train_modal")
    result = start_task(None, dataset_id, ckpt_id, train_modal)
    return jsonify(result)


class MyTrainTask(celery.Task):
    def on_success(self, retval, task_id, args, kwargs):
        print('MyTrainTask  ' + task_id + ' success!')
        try:
            # 调用连接池
            conn = dbPool.connection()
            # 获取执行查询的对象
            cursor = conn.cursor()
            # 执行那个查询，这里用的是 select 语句
            sql = "select * from train where train_task_id=%s"
            rows = cursor.executemany(sql, (task_id))
            # 提交
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            for data in data_dict:
                values = task_info_tool(task_id)
                if data['train_status'] != values['train_status']:
                    train_id = data['train_id']
                    values = task_info_tool(task_id)
                    # 停止后更新数据库
                    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # sql = 'update train set train_took_seconds = %s train_update_time =%s train_status = %s where train_id = %s'
                    values['train_took_seconds'] = values['train_took_seconds'] + data['train_took_seconds']
                    values['train_update_time'] = dt
                    values['train_comment'] = 'Task done'

                    sql = 'UPDATE train SET {}'.format(', '.join('{}=%s'.format(k) for k in values.keys()))
                    sql = sql + " where train_id='" + str(train_id) + "'"
                    print(sql)
                    print(values.values())
                    cursor.execute(sql, list(values.values()))
                    conn.commit()
        except IOError:
            conn.rollback()  # 出现异常 回滚事件
            print("Error: on_failure happen Error")
        finally:
            cursor.close()
            conn.close()
        return super(MyTrainTask, self).on_success(retval, task_id, args, kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print('MyTrainTask  ' + task_id + ' fail!')
        try:
            # 调用连接池
            conn = dbPool.connection()
            # 获取执行查询的对象
            cursor = conn.cursor()
            # 执行那个查询，这里用的是 select 语句
            sql = "select * from train where train_task_id=%s"
            rows = cursor.executemany(sql, (task_id))
            # 提交
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            for data in data_dict:
                values = task_info_tool(task_id)
                if data['train_status'] != values['train_status']:
                    train_id = data['train_id']
                    values = task_info_tool(task_id)
                    # 停止后更新数据库
                    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # sql = 'update train set train_took_seconds = %s train_update_time =%s train_status = %s where train_id = %s'
                    values['train_took_seconds'] = values['train_took_seconds'] + data['train_took_seconds']
                    values['train_update_time'] = dt
                    values['train_comment'] = 'Task fail'

                    sql = 'UPDATE train SET {}'.format(', '.join('{}=%s'.format(k) for k in values.keys()))
                    sql = sql + " where train_id='" + str(train_id) + "'"
                    print(sql)
                    print(values.values())
                    cursor.execute(sql, list(values.values()))
                    conn.commit()
        except IOError:
            conn.rollback()  # 出现异常 回滚事件
            print("Error: on_failure happen Error")
        finally:
            cursor.close()
            conn.close()

        return super(MyTrainTask, self).on_failure(exc, task_id, args, kwargs, einfo)


@celery.task(base=MyTrainTask, bind=True)
def simple_train(self, train_id, train_modal, ds_data, ckpt_data):
    # To be continue
    # ⚠️此处调用自定义的模型训练方式！！！
    long_task(self, train_id, train_modal, ds_data, ckpt_data)
    # conv_fc(self, train_id, train_modal, ds_data, ckpt_data)


## 前往 某个训练记录 详情页面
# 待完善
@app.route("/record/<recordId>")
def record_page(recordId):
    ## recordId是 某个训练记录Id
    print("recordId:" + str(recordId))
    ## 前端显示 这个训练 的 状态 , 所用数据集, 总运行时间,训练进度，准确率, 每轮chart 等。  详见前端显示
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "select * from train where train_id=%s"
        rows = cursor.execute(sql, (recordId))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        if (data_dict[0] is not None and len(data_dict) >= 1):
            task_id = data_dict[0]['train_task_id']
            data_dict[0]['task_info_url'] = url_for('get_task_info', taskId=task_id, trainId=recordId)
            data_dict[0]['task_took_seconds'] = int(
                (datetime.datetime.now() - data_dict[0]['train_update_time']).total_seconds() + data_dict[0][
                    'train_took_seconds'])
            print(data_dict[0])
            return render_template("record.html", **data_dict[0])
        else:
            return render_template("error.html")

    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function record_page happen Error")
    finally:
        cursor.close()
        conn.close()
    ## 模拟 生成 训练记录 详情


@app.route("/task", methods=['POST'])
def get_task_info():
    ## recordId是 某个训练记录Id
    trainId = request.json.get("train_id")
    taskId = request.json.get("task_id")
    last_update_time = request.json.get("last_update_time")
    last_update_time = try_time_format(last_update_time, "%Y-%m-%d %H:%M:%S")

    print("get_task_info:" + str(taskId))
    response = task_info_tool(taskId)
    show_chart_info = get_task_statistics_info(last_update_time, trainId)
    train_info_dict = {}
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_info_dict['show_chart_info'] = show_chart_info
    train_info_dict['update_time'] = dt
    if response is not None:
        # 更新数据库
        now = datetime.datetime.now()
        values_list = []
        # values = {
        #     'train_status': response['train_status'],
        #     'train_update_time': dt,
        #     'train_ckpt_id': response['train_ckpt_id'],
        #     'train_dataset_id': response['train_dataset_id'],
        #     'train_took_seconds': int(response['train_took_seconds']),
        #     'train_task_id': taskId,
        #     'train_epochs': response['train_epochs'],
        #     'train_epoch': response['train_epoch'],
        #     'train_steps': response['train_steps'],
        #     'train_step': response['train_step'],
        #     'train_acc': response['train_acc'],
        #     'train_loss': response['train_loss'],
        #     'train_comment': response['train_comment']
        # }
        values = {}
        values['train_update_time'] = dt
        values['train_task_id'] = taskId
        response.update(values)
        r_keys = response.keys()
        if 'train_task_id' in r_keys:
            response.pop('train_task_id')
        if 'train_id' in r_keys:
            response.pop('train_id')
        values_list.append((trainId, response))
        update_batch_train_task_info(values_list)

        if 'train_epochs' not in r_keys:
            response['train_epochs'] = 1
        if 'train_epoch' not in r_keys:
            response['train_epoch'] = 0
        if 'train_steps' not in r_keys:
            response['train_steps'] = 1
        if 'train_step' not in r_keys:
            response['train_step'] = 0
        if 'train_acc' not in r_keys:
            response['train_acc'] = 0
        if 'train_loss' not in r_keys:
            response['train_loss'] = 0

        train_info_dict['task_running_info'] = response
        return jsonify({"success": 1, "msg": "获取成功!", "train": train_info_dict})
    else:
        return jsonify({"success": 2, "msg": "无此taks运行信息!", 'train': train_info_dict})


@app.route("/get_recent_task_list")
def get_recent_task_list():
    print("get_recent_task_list:")
    task_list = get_priority_task_info_list()
    activity_task_status = ('NEW', 'PENDING', 'PREPARE', 'PROGRESS')
    values_list = []
    for task in task_list:
        if task['train_status'] in activity_task_status:
            response = task_info_tool(task['train_task_id'])
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d %H:%M:%S")
            if response is None:  # 此线程不存在
                values = {
                    'train_status': 'NONE',
                    'train_update_time': dt,
                    'train_took_seconds': int(task['train_took_seconds']) + int(
                        (now - datetime.datetime.strptime(task['train_update_time'],
                                                          "%Y-%m-%d %H:%M:%S")).total_seconds()),
                    'train_comment': 'This thread is not exists by train_task_id'
                }
            else:
                values = {
                    'train_update_time': dt,
                    'train_took_seconds': int(task['train_took_seconds']) + int((now - datetime.datetime.strptime(
                        task['train_update_time'], "%Y-%m-%d %H:%M:%S")).total_seconds()),
                }
                r_keys = response.keys()
                if 'train_task_id' in r_keys:
                    response.pop('train_task_id')
                if 'train_id' in r_keys:
                    response.pop('train_id')
                if 'train_epochs' not in r_keys:
                    response['train_epochs'] = 1
                if 'train_epoch' not in r_keys:
                    response['train_epoch'] = 0
                if 'train_steps' not in r_keys:
                    response['train_steps'] = 1
                if 'train_step' not in r_keys:
                    response['train_step'] = 0
                if 'train_acc' not in r_keys:
                    response['train_acc'] = 0
                if 'train_loss' not in r_keys:
                    response['train_loss'] = 0
                values.update(response)

                # values = {
                #     'train_status': response['train_status'],
                #     'train_update_time': dt,
                #     'train_ckpt_id': response['train_ckpt_id'],
                #     'train_dataset_id': response['train_dataset_id'],
                #     'train_took_seconds': int(task['train_took_seconds']) + int((now - datetime.datetime.strptime(
                #         task['train_update_time'], "%Y-%m-%d %H:%M:%S")).total_seconds()),
                #     'train_epochs': response['train_epochs'],
                #     'train_epoch': response['train_epoch'],
                #     'train_steps': response['train_steps'],
                #     'train_step': response['train_step'],
                #     'train_acc': response['train_acc'],
                #     'train_loss': response['train_loss'],
                #     'train_comment': response['train_comment']
                #
                # }
            values_list.append((task['train_id'], values))
    update_batch_train_task_info(values_list)

    return jsonify({"success": 1, "msg": "获取成功!", "task_list": task_list})


@app.route("/get_statistics", methods=['POST'])
def get_statistics():
    print("get_statistics:")
    last_update_time = request.json.get("last_update_time")
    this_main_chart_train_id = request.json.get("this_main_chart_train_id")
    last_update_time = try_time_format(last_update_time, "%Y-%m-%d %H:%M:%S")
    if this_main_chart_train_id is None or not is_int_number(this_main_chart_train_id):
        this_main_chart_train_id = None
    statistics = get_statistics_info(last_update_time, this_main_chart_train_id)

    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    statistics.update({
        'update_time': dt
    })
    localHostStat = getLocalInfo();
    statistics.update(localHostStat)
    print(statistics)

    return jsonify({"success": 1, "msg": "获取成功!", "statistics": statistics})


@app.route("/stop/<trainId>")
def stop_train(trainId):
    print("stop_train  trainId:" + str(trainId))
    ## 前端显示 这个训练 的 状态 , 所用数据集, 总运行时间,训练进度，准确率, 每轮chart 等。  详见前端显示
    result = stop_task_with_save(trainId)
    return jsonify(result)


@app.route("/restart/<trainId>")
def restart_train(trainId):
    print("restart_train  trainId:" + str(trainId))
    ## 前端显示 这个训练 的 状态 , 所用数据集, 总运行时间,训练进度，准确率, 每轮chart 等。  详见前端显示
    result = restart_task(trainId, None, None)
    return jsonify(result)


@app.route("/reset/<trainId>")
def reset_train(trainId):
    print("reset_train  trainId:" + str(trainId))
    ## 前端显示 这个训练 的 状态 , 所用数据集, 总运行时间,训练进度，准确率, 每轮chart 等。  详见前端显示
    result = restart_task(trainId, -1, None)
    return jsonify(result)


def get_priority_task_info_list():
    print('get_priority_task_info_list')
    activity_task_status1 = ('NEW', 'PREPARE', 'PROGRESS')
    activity_task_status3 = ('PENDING', 'REVOKED')
    task_list = []
    try:

        conn = dbPool.connection()
        cursor = conn.cursor()
        sql = "select * from train where train_status in (%s,%s,%s)"
        rows = cursor.execute(sql, activity_task_status1)
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        for data in data_dict:
            print(data)
            data['train_create_time'] = data['train_create_time'].strftime("%Y-%m-%d %H:%M:%S")
            data['train_update_time'] = data['train_update_time'].strftime("%Y-%m-%d %H:%M:%S")
            task_list.append(data)

        sql = "select * from train where train_status = %s order by train_update_time desc limit 5"
        rows = cursor.execute(sql, ('SUCCESS'))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        for data in data_dict:
            data['train_create_time'] = data['train_create_time'].strftime("%Y-%m-%d %H:%M:%S")
            data['train_update_time'] = data['train_update_time'].strftime("%Y-%m-%d %H:%M:%S")
            task_list.append(data)
        if len(task_list) <= 5:
            sql = "select * from train where train_status not in (%s,%s) order by train_update_time desc limit 5"
            rows = cursor.execute(sql, activity_task_status3)
            # 提
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            for data in data_dict:
                data['train_create_time'] = data['train_create_time'].strftime("%Y-%m-%d %H:%M:%S")
                data['train_update_time'] = data['train_update_time'].strftime("%Y-%m-%d %H:%M:%S")
                task_list.append(data)
        return task_list

    except IOError:
        print("Error: Function get_priority_task_info_list happen Error")
        return None
    finally:
        cursor.close()
        conn.close()
    return None


def get_task_statistics_info(last_update_time, train_id):
    print('get_task_statistics_info')
    show_data = {}
    try:
        conn = dbPool.connection()
        cursor = conn.cursor()
        sql = "select * from train_record where train_id = %s order by record_id desc limit 15"
        rows = cursor.execute(sql, (train_id))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        recent_all_data = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        for data in recent_all_data:
            data['update_time'] = data['update_time'].strftime("%Y-%m-%d %H:%M:%S")
        show_data.update({
            'recent_all_data': recent_all_data
        })
        # print("train recrod, train id", train_id)
        last_data = []
        # 提交
        sql = "select record_id,train_id,update_time,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss from train_record where train_id = %s order by record_id desc limit 2"
        rows = cursor.execute(sql, (train_id))
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        for index, data in enumerate(data_dict):
            # item = {}
            # item['record_id'] = data['record_id']
            # item['train_id'] = data['train_id']
            # item['update_time'] = data['update_time'].strftime("%H-%M-%S")
            # item['train_dataset_id'] = data['train_dataset_id']
            # item['train_ckpt_id'] = data['train_ckpt_id']
            # item['train_epochs'] = data['train_epochs']
            # item['train_epoch'] = data['train_epoch']
            # item['train_steps'] = data['train_steps']
            # item['train_step'] = data['train_step']
            # item['train_acc'] = data['train_acc']
            # item['train_loss'] = data['train_loss']
            # item['train_modal'] = data['train_modal']
            # item['train_status'] = data['train_status']

            # last_update_time = try_time_format(last_update_time,"%Y-%m-%d %H:%M:%S")
            if last_update_time is not None:
                if data['update_time'] > last_update_time:
                    if index < len(data_dict) - 1:
                        data['took_time'] = int((data['update_time'] - data_dict[index + 1]['update_time']).total_seconds())
                    else:
                        data['took_time'] = 0
                        data['update_time'] = data['update_time'].strftime("%H-%M-%S")
                    last_data.append(data)

        print(last_data)

        show_data.update({'show_chart': last_data})

        return show_data

    except IOError:
        print("Error: Function get_task_statistics_inf happen Error")
        return None
    finally:
        cursor.close()
        conn.close()
    return None


def get_statistics_info(last_update_time, this_main_chart_train_id):
    print('get_statistics_info')
    activity_task_status1 = ('NEW', 'PREPARE', 'PROGRESS')
    activity_task_status3 = ('SUCCESS', 'REVOKED', 'NEW', 'PENDING', 'PREPARE', 'PROGRESS')
    stat_dict = {}
    try:

        conn = dbPool.connection()
        cursor = conn.cursor()
        sql = "select convert(count(train_id),CHAR) as train_count, convert(sum(train_took_seconds),CHAR) as train_took from train where train_status in (%s,%s,%s,%s,%s,%s)"
        rows = cursor.execute(sql, activity_task_status3)
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        stat_dict.update(data_dict[0])

        sql = "select convert(count(dataset_id),CHAR) as ds_count, convert(sum(dataset_size),CHAR) as ds_size from dataset "
        rows = cursor.execute(sql)
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        stat_dict.update(data_dict[0])
        show_data = []
        if this_main_chart_train_id is not None:
            sql = "select * from train where train_id=%s   order by train_create_time desc"
            rows = cursor.execute(sql, this_main_chart_train_id)
            # 提交
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            if len(data_dict) > 0:
                data = data_dict[0]
                if data['train_status'] not in activity_task_status1:
                    stat_dict.update({'this_main_chart_train_id_need_change': True})
                else:
                    stat_dict.update({'this_main_chart_train_id_need_change': False})
                    print("train recrod, train id", data['train_id'])
                    if last_update_time is None:
                        sql = "select record_id,train_id,update_time,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss from train_record where train_id = %s order by update_time desc limit 1"
                        rows = cursor.execute(sql, (data['train_id']))
                    else:
                        sql = "select record_id,train_id,update_time,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss from train_record where train_id = %s  and  update_time>%s order by update_time desc limit 1"
                        rows = cursor.execute(sql, (data['train_id'], last_update_time))
                    # 提交
                    conn.commit()
                    desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
                    data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
                    for data in data_dict:
                        data['update_time'] = data['update_time'].strftime("%H-%M-%S")
                    print(data_dict)
                    show_data.append(data_dict)
            else:
                stat_dict.update({'this_main_chart_train_id_need_change': True})
        else:
            stat_dict.update({'this_main_chart_train_id_need_change': True})
        stat_dict.update({'show_chart': show_data})

        return stat_dict

    except IOError:
        print("Error: Function get_priority_task_info_list happen Error")
        return None
    finally:
        cursor.close()
        conn.close()
    return None


def start_task(train_id, dataset_id, ckpt_id, train_modal):
    if train_id is None or not is_int_number(train_id):
        train_id = 0
    train_id = int(train_id)
    if ckpt_id is None or not is_int_number(ckpt_id):
        ckpt_id = 0
    if dataset_id is None or not is_int_number(dataset_id):
        if train_id <= 0:
            return {"success": 0, "msg": "无法开始训练! DS Id错误!"}
        else:
            dataset_id = 0
    dataset_id = int(dataset_id)
    ckpt_id = int(ckpt_id)
    print("start_train-- dataset_id:" + str(dataset_id) + "  ckpt_id:" + str(ckpt_id))
    if train_id <= 0:
        if train_modal is None:
            train_modal = 'A'
    ## 模拟开始训练过程
    # 交由线程去执行耗时任务
    # executor.submit(task.mnist_train, 'hello', 123)
    ckpt_data = None
    ds_data = None
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()

        if train_id > 0:  # 此Train已有说明是继续开始的任务
            sql = "select * from train where train_id=%s"
            rows = cursor.execute(sql, (train_id))
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            train_data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            if len(train_data_dict) >= 1:
                train_data = train_data_dict[0]
                dataset_id = train_data['train_dataset_id']
                train_modal = train_data['train_modal']
                if ckpt_id != -1:  ## ckpt_id = -1 说明是重新开始
                    ckpt_id = train_data['train_ckpt_id']

        ## 检测ckpt并获取
        # 获取ckpt和ds地址
        if ckpt_id > 0:
            sql = "select * from ckpt where ckpt_id=%s"
            cursor.execute(sql, (ckpt_id))
            # 提交
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            ckpt_data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            if len(ckpt_data_dict) <= 0:
                if train_id <= 0:
                    return {"success": 0, "msg": "无法开始训练! Ckpt Id不存在!"}
            else:
                ckpt_data = ckpt_data_dict[0]
                ckpt_path = ckpt_data['ckpt_path']
                if not os.path.exists(ckpt_path):
                    sql = "delete from ckpt where ckpt_id=%s"
                    rows = cursor.execute(sql, (ckpt_data['ckpt_id']))
                    # 提交
                    conn.commit()
                    if train_id <= 0:
                        return {"success": 0, "msg": "无法开始训练! 此Ckpt不存在!"}
                    else:
                        ckpt_id = -1
                        ckpt_data = None
                        ckpt_path = None

        ## 检测dataset并获取
        sql = "select * from dataset where dataset_id=%s"
        cursor.execute(sql, (dataset_id))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        ds_data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        if len(ds_data_dict) <= 0:
            return {"success": 0, "msg": "无法开始训练! DS Id不存在!"}
        else:
            ds_data = ds_data_dict[0]
            dataset_path = ds_data['dataset_path']
            if not os.path.exists(dataset_path):
                ## 注释代码是检测到数据集不存在时就会删除数据库中的记录，但是一般不会不存在，所以注释掉
                # sql = "delete from dataset where dataset_id=%s"
                # rows = cursor.execute(sql, (dataset_id))
                # # 提交
                # conn.commit()
                return {"success": 0, "msg": "无法开始训练! 此数据集不存在!"}

        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if train_id > 0:  # 继续已有任务开始
            task = simple_train.apply_async(args=[train_id, train_modal, ds_data, ckpt_data])
            values = {
                'train_task_id': task.id,
                'train_status': 'NEW'
            }
            if ds_data is not None:
                values['train_dataset_id'] = ds_data['dataset_id']
            else:
                values['train_dataset_id'] = 0
            if ckpt_data is not None:
                values['train_ckpt_id'] = ckpt_data['ckpt_id']

            update_train_task_info(train_id, values)
        else:  ## 是从零开始的任务，无train_id开启
            # 执行那个查询，这里用的是 select 语句
            sql = "insert into train(train_name,train_create_time,train_update_time,train_record_num,train_status,train_dataset_id,train_ckpt_id,train_task_id,train_took_seconds,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss,train_comment,train_modal) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            cursor.execute(sql,
                           ("Train", dt, dt, 0, 'NEW', dataset_id, ckpt_id, 0, 0, 0, 0, 0, 0, 0, 0, '', train_modal))
            # 提交
            conn.commit()
            id = cursor.lastrowid  # 获取自增ID号
            task = simple_train.apply_async(args=[id, train_modal, ds_data, ckpt_data])
            values = {
                'train_task_id': task.id
            }
            update_train_task_info(id, values)
            train_id = id
        return {"success": 1, "msg": "接收到参数,开始训练!", 'train_id': train_id, 'train_task_id': task.id}
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function start_task happen Error")
        return {"success": 0, "msg": "start_task Mysql error!"}
    finally:
        cursor.close()
        conn.close()
    return {"success": 0, "msg": "start_task fail!"}


def restart_task(trainId, train_ckpt_id, train_dataset_id):
    msg = "restart_task  trainId:" + str(trainId)
    if train_ckpt_id is not None:
        msg = msg + ' train_ckpt_id:' + str(train_ckpt_id)
    else:
        msg = msg + ' train_ckpt_id: None'
    if train_dataset_id is not None:
        msg = msg + ' train_dataset_id:' + str(train_dataset_id)
    else:
        msg = msg + ' train_dataset_id: None'
    print(msg)
    return start_task(trainId, train_dataset_id, train_ckpt_id, None)
    # try:
    #     # 调用连接池
    #     conn = dbPool.connection()
    #     # 获取执行查询的对象
    #     cursor = conn.cursor()
    #     # 执行那个查询，这里用的是 select 语句
    #     sql = "select * from train where train_id=%s"
    #     rows = cursor.execute(sql, (trainId))
    #     # 提交
    #     conn.commit()
    #     desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
    #     data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
    #     if (data_dict[0] is not None and len(data_dict) >= 1):
    #         task_id = data_dict[0]['train_task_id']
    #         # 先停止任务
    #         stop_task_with_save(trainId)
    #
    #         dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         # 开始任务
    #         reulst = start_task(trainId,data_dict[0]['train_dataset_id'])
    #         task = simple_train.delay(trainId, data_dict[0]['train_modal'], data_dict[0]['train_dataset_id'],
    #                                   data_dict[0]['train_ckpt_id'])
    #         values = {
    #             'train_task_id': task.id,
    #             'train_update_time': dt,
    #             'train_status': task.state,
    #             'train_ckpt_id': data_dict[0]['train_ckpt_id'],
    #             'train_dataset_id': data_dict[0]['train_dataset_id'],
    #             'train_comment': 'Task restart'
    #         }
    #
    #         if train_ckpt_id == -1:  # -1代表 无ckpt 从零开始
    #             values['train_ckpt_id'] = -1
    #             train_ckpt_id = None
    #         else:
    #             if data_dict[0]['train_ckpt_id'] > 0:  # 使用原先数据库记录的最新ckpt
    #                 values['train_ckpt_id'] = data_dict[0]['train_ckpt_id']
    #                 train_ckpt_id = data_dict[0]['train_ckpt_id']
    #             else:  # 说明数据库里的ckpt也是默认值(不存在)，那就从零开始
    #                 values['train_ckpt_id'] = -1
    #                 train_ckpt_id = None
    #
    #         if train_dataset_id is not None and train_dataset_id > 0:  # 说明使用新指定的数据集
    #             values['train_dataset_id'] = train_dataset_id
    #         else:  # 说明使用原先的数据集
    #             values['train_dataset_id'] = data_dict[0]['train_dataset_id']
    #             train_dataset_id = data_dict[0]['train_dataset_id']
    #
    #         sql = 'UPDATE train SET {}'.format(', '.join('{}=%s'.format(k) for k in values.keys()))
    #         sql = sql + " where train_id='" + str(trainId) + "'"
    #         cursor.execute(sql, list(values.values()))
    #         conn.commit()
    #         return {"success": 1, "msg": trainId + " restarted!", "taskId": task.id}
    # except IOError:
    #     conn.rollback()  # 出现异常 回滚事件
    #     print("Error: Function restart_task happen Error")
    #     return {"success": 0, "msg": trainId + " restart error!"}
    # finally:
    #     cursor.close()
    #     conn.close()
    # return {"success": 0, "msg": trainId + " restart fail!"}


def stop_task_with_save(trainId):
    print("stop_task_with_save  trainId:" + str(trainId))
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "select * from train where train_id=%s"
        rows = cursor.execute(sql, (trainId))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        if (data_dict[0] is not None and len(data_dict) >= 1):
            task_id = data_dict[0]['train_task_id']

            # 停止任务
            response = task_info_tool(task_id)
            celery.control.revoke(task_id, terminate=True)
            # 停止后更新数据库
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # sql = 'update train set train_took_seconds = %s train_update_time =%s train_status = %s where train_id = %s'
            response['train_update_time'] = dt
            response['train_took_seconds'] = response['train_took_seconds'] + data_dict[0]['train_took_seconds']
            response['train_status'] = 'TERMINATED'
            response['train_comment'] = 'Task is terminated!'

            sql = 'UPDATE train SET {}'.format(', '.join('{}=%s'.format(k) for k in response.keys()))
            sql = sql + " where train_id='" + str(trainId) + "'"
            cursor.execute(sql, list(response.values()))
            conn.commit()
            return {"success": 1, "msg": trainId + "is terminated!"}
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function stop_task_with_save happen Error")
        return {"success": 0, "msg": trainId + " terminate error!"}
    finally:
        cursor.close()
        conn.close()
    return {"success": 0, "msg": trainId + " terminate fail!"}


# @celery.task(bind=True)
def update_batch_train_task_info(values_list):
    for train_id, values in values_list:
        update_train_task_info(train_id, values)


def update_train_task_info(train_id, values):
    print("update_train_task_info")
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        sql = 'UPDATE train SET {}'.format(', '.join('{}=%s'.format(k) for k in values.keys()))
        sql = sql + " where train_id='" + str(train_id) + "'"
        rows = cursor.execute(sql, list(values.values()))
        conn.commit()
        if rows > 0:
            return True
        else:
            return False
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function update_train_task_info happen Error")
        return False
    finally:
        cursor.close()
        conn.close()
    return False


# @celery.task(bind=True)
def insert_new_train_record(values):
    print('insert_new_train_record')
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "insert into train_record({}".format(
            ', '.join('{}'.format(k) for k in values.keys())) + ") values({}".format(
            ', '.join('%s' for k in values.keys())) + ")"
        # sql = "insert into train(train_name,train_create_time,train_update_time,train_record_num,train_status,train_dataset_id,train_ckpt_id,train_task_id,train_took_seconds,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss,train_comment ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        cursor.execute(sql, list(values.values()))
        # 提交
        conn.commit()
        id = cursor.lastrowid  # 获取自增ID号
        return id
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function insert_dataset_info happen Error")
        return -1
    finally:
        cursor.close()
        conn.close()
    return -1


# ******   还未设置某个dataSource、某个data、某个ckpt的 CRUD 的route，  前端已经实现了crud效果，这里看情况是否添加


# 上传训练数据集
# 已实现 批量上传数据 到指定 数据集
@app.route('/upload_image', methods=['POST'], strict_slashes=False)
def upload_image():
    print("upload_image")
    file_dir = os.path.join(basedir, app.config['DATA_SET_FOLDER'])
    # 上传同时，前端会发送数据源Id,根据不同Id，保存到不同数据源
    # dataSourceId=-1，说明是新建的数据源
    # **** 还  需要实现  保存已有数据源的信息，有哪些数据源，
    # 有了数据源后，新建训练就可以选择不同数据源来训练
    dataset_name = request.values.get("dataset_name")
    dataset_id = request.values.get("dataset_id")
    dataset_tag = request.values.get("dataset_tag")
    dataset_comment = request.values.get("dataset_comment")
    if ((not is_int_number(dataset_id)) or (not is_available(dataset_name)) or (not is_available(dataset_tag))):
        return jsonify({"success": 0, "msg": "上传图片失败,参数不正确"})
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_id = int(dataset_id)
    if (dataset_id > 0):  # 获取数据库原有的数据集
        dataset_info = get_dataset_info_by_id(dataset_id)
        if dataset_info is None:  # 说明数据库中不存在此id的数据集的信息
            return jsonify({"success": 0, "msg": "上传图片失败,参数数据集Id不存在"})
        else:  # 说明数据库有此id的数据集信息，则更新
            print("update ds,  dataset_id:" + str(dataset_info['dataset_id']))
            f_list = request.files.getlist('upData')
            print("f_list lenght:" + str(len(f_list)))
            file_dir = os.path.join(file_dir, str(dataset_info['dataset_id']))
            # executor.submit(save_image_and_update_db,dataset_info['dataset_id'], file_dir, f_list)
            save_image_and_update_db(dataset_info['dataset_id'], file_dir, f_list)
            return jsonify({"success": 1, "msg": "插入数据，更新数据集成功!"})
    else:  # 说明是新建数据集
        print("new ds,  dataset_tag:" + dataset_tag + "  dataset_name:" + dataset_name)
        if not (dataset_tag == 'trainset' or dataset_tag == 'testset' or dataset_tag == 'validset'):
            return jsonify({"success": 0, "msg": "上传图片失败,参数Tag不正确"})
        values = {
            'dataset_name': dataset_name,
            'dataset_size': 0,
            'dataset_update_time': dt,
            'dataset_create_time': dt,
            'dataset_comment': dataset_comment,
            'dataset_tag': dataset_tag,
            'dataset_version': 0,
            'dataset_status': 'NEW'
        }
        dataset_id = insert_dataset_info(values)
        if dataset_id > 0:
            file_dir = os.path.join(file_dir, str(dataset_id))
            f_list = request.files.getlist('upData')
            print("f_list lenght:" + str(len(f_list)))
            # print("f_list type:"+str(type(f_list)))
            # executor.submit(save_image_and_update_db, dataset_id, file_dir, f_list)
            save_image_and_update_db(dataset_id, file_dir, f_list)
            return jsonify({"success": 1, "msg": "插入数据，新建数据集成功!"})
        else:
            return jsonify({"success": 0, "msg": "新建数据集失败!"})


@app.route("/update_ds_info", methods=['POST'])
def update_dataSource_info():
    print("update_dataSource_info")
    dataset_name = request.json.get("dataset_name")
    dataset_id = request.json.get("dataset_id")
    dataset_tag = request.json.get("dataset_tag")
    dataset_comment = request.json.get("dataset_comment")
    if ((not is_int_number(dataset_id)) or int(dataset_id) <= 0):
        return jsonify({"success": 0, "msg": "更新失败,id不正确"})
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_id = int(dataset_id)
    values = {
        'dataset_update_time': dt,
    }
    if is_available(dataset_name):
        values['dataset_name'] = dataset_name
    if is_available(dataset_tag) and (
            dataset_tag == 'trainset' or dataset_tag == 'testset' or dataset_tag == 'validset'):
        values['dataset_tag'] = dataset_tag
    if is_available(dataset_comment):
        values['dataset_comment'] = dataset_comment
    result = update_dataset_info_by_id(dataset_id, values)
    if result:
        return jsonify({"success": 1, "msg": "更新成功"})
    else:
        return jsonify({"success": 0, "msg": "更新失败"})


@app.route("/remove_ds/<dataset_id>")
def remove_dataSource(dataset_id):
    print("remove_dataSource")
    if ((not is_int_number(dataset_id)) or int(dataset_id) <= 0):
        return jsonify({"success": 0, "msg": "删除失败,id不正确"})
    dataset_id = int(dataset_id)
    result = delete_dataset_info_by_id_and_delete_files(dataset_id)
    if result:
        return jsonify({"success": 1, "msg": "删除成功"})
    else:
        return jsonify({"success": 0, "msg": "删除失败"})


## 前端ajax传递获取 数据集信息列表请求
@app.route("/get_ds_info_list")
def get_dataSource_info_list():
    print("get_dataSource_info_list")
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "select * from dataset;"
        rows = cursor.execute(sql)
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict_list = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        for data in data_dict_list:
            data['dataset_update_time'] = data['dataset_update_time'].strftime("%Y-%m-%d %H:%M:%S")
            data['dataset_create_time'] = data['dataset_create_time'].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"success": 1, "msg": "get datalist success!", "datalist": data_dict_list})
    except IOError:
        # conn.rollback()  # 出现异常 回滚事件
        print("Error: Function get_dataSource_list happen Error")
        return jsonify({"success": 0, "msg": "get datalist error!"})
    finally:
        cursor.close()
        conn.close()
    return jsonify({"success": 0, "msg": "get datalist fail!"})


## 前端ajax传递获取 数据集信息列表请求
@app.route("/get_data_url")
def get_data_url():
    print("get_dataSource_list")
    dataset_id = request.args.get('dataset_id')
    limit = request.args.get('limit')
    page = request.args.get('page')
    print(dataset_id)
    if not is_int_number(dataset_id) and int(dataset_id) <= 0:
        return jsonify({"success": 0, "msg": "获取图片url失败,dataset_id不正确!"})
    dataset_id = int(dataset_id)

    data = get_dataset_info_by_id(dataset_id)
    file_url_list = []
    if not is_int_number(limit) or int(limit) > 100:  # 说明获取不限个数图片url，但是这里限制为100个
        limit = 100
    limit = int(limit)
    if not is_int_number(page):  # 说明获取不限个数图片url，但是这里限制为100个
        page = 1
    elif int(page) > 0:
        limit = 40  # 说明是某个数据集展示页面需求url list,这里固定一页显示40个图片
        page = int(page)
    else:
        page = 1

    start = (page - 1) * limit
    end = page * limit
    count = 0
    if data is not None:
        for file_name in os.listdir(data['dataset_path']):
            print(file_name)
            if allowed_file(file_name):
                if count >= start and count < end:
                    file_url_list.append('/static/data/images/' + str(dataset_id) + '/' + str(file_name))
                elif count >= end:
                    break;
                count += 1
        return jsonify({"success": 1, "msg": "获取图片url成功!", 'url_list': file_url_list})
    else:
        return jsonify({"success": 0, "msg": "获取图片url失败!"})


## 前往 某个数据集详情页面
@app.route("/dataset/<dataset_id>")
def data_set_page(dataset_id):
    print("data_set_page")
    if ((not is_int_number(dataset_id)) or int(dataset_id) <= 0):
        return jsonify({"success": 0, "msg": "删除失败,id不正确"})
    dataset_id = int(dataset_id)
    data = get_dataset_info_by_id(dataset_id)
    if data is None:
        return render_template("error.html")
    else:
        return render_template("data.html", **data)


@app.route("/update_image", methods=['POST'])
def update_image():
    print("update_image_label")
    dataset_id = request.json.get('dataset_id')
    image_orig_name = request.json.get('image_orig_name')
    newLabel = request.json.get('newLabel')
    if not is_int_number(dataset_id) and int(dataset_id) <= 0:
        return jsonify({"success": 0, "msg": "dataset_id不正确!"})
    if not is_available(image_orig_name) or not allowed_file(image_orig_name):
        return jsonify({"success": 0, "msg": "image_orig_name不正确!"})
    if not is_available(newLabel) or not allowed_label(newLabel.strip()):
        return jsonify({"success": 0, "msg": "newLabel不正确!"})
    newLabel = newLabel.strip() + ""
    dataset_id = int(dataset_id)
    data = get_dataset_info_by_id(dataset_id)
    update_image_label.delay(data['dataset_path'], image_orig_name, newLabel)
    return jsonify({"success": 1, "msg": "image label更新成功!"})


@app.route("/delete_image", methods=['POST'])
def delete_image():
    print("delete_image")
    dataset_id = request.json.get('dataset_id')
    image_orig_name = request.json.get('image_orig_name')
    if not is_int_number(dataset_id) and int(dataset_id) <= 0:
        return jsonify({"success": 0, "msg": "dataset_id不正确!"})
    if not is_available(image_orig_name) or not allowed_file(image_orig_name):
        return jsonify({"success": 0, "msg": "image_orig_name不正确!"})

    dataset_id = int(dataset_id)
    data = get_dataset_info_by_id(dataset_id)
    orig_file_path = os.path.join(data['dataset_path'], image_orig_name)
    if os.path.exists(orig_file_path):
        os.remove(orig_file_path)
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        values = {
            'dataset_update_time': dt
        }
        count_dataset_size_and_upadte_db.delay(data['dataset_id'], values, data['dataset_path'])
        return jsonify({"success": 1, "msg": "image删除成功!"})
    else:
        return jsonify({"success": 0, "msg": "image不存在!"})


@celery.task(bind=True)
def update_image_label(self, file_dir, orig_file_name, new_file_label):
    orig_file_path = os.path.join(file_dir, orig_file_name)
    if not os.path.exists(orig_file_path):
        self.update_state(state='FAILURE',
                          meta={
                              'file_dir': file_dir,
                              'orig_file_name': orig_file_name,
                              'new_file_name': new_file_name,
                              'comment': 'file_dir not exists!'
                          })
        return False;
    else:
        file = os.path.splitext(orig_file_path)
        filename, type = file
        new_file_name = new_file_label + type
        new_name_path = os.path.join(file_dir, new_file_name)
        os.rename(orig_file_path, new_name_path)
        self.update_state(state='SUCCESS',
                          meta={
                              'file_dir': file_dir,
                              'orig_file_name': orig_file_name,
                              'new_file_name': new_file_name,
                              'comment': 'file_dir not exists!'
                          })
        return True;


## 前端ajax传递获取 数据集信息列表请求
@app.route("/get_ckpt_list")
def get_ckpt_list():
    print("get_ckpt_list")
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        find_num = 15
        max_find_count = 3
        min_num = 10
        result = []
        while max_find_count > 0:
            mis_ckpt_list_id = []
            result = []
            sql = "select * from ckpt  order by create_time desc limit %s;"
            rows = cursor.execute(sql, (find_num))
            # 提交
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            data_dict_list = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            if (len(data_dict_list) == 0):
                break
            for index, data in enumerate(data_dict_list):
                if not os.path.exists(data['ckpt_path']):
                    mis_ckpt_list_id.append(data['ckpt_id'])
                else:
                    data['create_time'] = data['create_time'].strftime("%m-%d %H:%M:%S")
                    data['ckpt_path'] = ''
                    result.append(data)
                    if len(result) >= min_num:
                        break
            # 删除不存在的ckpt
            if len(mis_ckpt_list_id) > 0:
                sql = "delete from ckpt where ckpt_id in ( {}".format(','.join(str(k) for k in mis_ckpt_list_id)) + ")"
                print(sql)
                rows = cursor.execute(sql)
                # 提交
                conn.commit()
            if len(result) >= min_num:
                break
            max_find_count -= 1

        return jsonify({"success": 1, "msg": "get ckpt list success!", "ckptlist": result})
    except IOError:
        # conn.rollback()  # 出现异常 回滚事件
        print("Error: Function get_ckpt_list happen Error")
        return jsonify({"success": 0, "msg": "get ckpt list error!"})
    finally:
        cursor.close()
        conn.close()
    return jsonify({"success": 0, "msg": "get ckpt list fail!"})


## 前往 控制面板页面
# 待完善
@app.route("/dashboard")
def dashboard():
    ## 控制面板显示 总的 训练概览   详见前端显示

    return render_template("dashboard.html")


## 前往 生成训练页面
@app.route("/train")
def train():
    return render_template("train.html")


## 前往 数据集列表页面
@app.route("/datalist")
def datalist():
    return render_template("datalist.html")


## 开启异步线程保存图片，
# @celery.task(bind=True)
def save_image_and_update_db(dataset_id, file_dir, f_list):
    if dataset_id is not None and dataset_id > 0:  # 说明是更新已有数据集
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        for f in f_list:
            print("filename:" + f.filename)
            if f and allowed_file(f.filename):
                fname = secure_filename(f.filename)
                print(fname)
                ext = fname.rsplit('.', 1)[1]
                new_filename = Pic_str().create_uuid() + '.' + ext
                f.save(os.path.join(file_dir, new_filename))

        # 上传完毕后 更新数据库信息
        values = {
            'dataset_update_time': dt
        }
        count_dataset_size_and_upadte_db.delay(dataset_id, values, file_dir)


@celery.task(bind=True)
def count_dataset_size_and_upadte_db(self, dataset_id, values, file_dir):
    # 查看该路径下有多少文件
    size = len(os.listdir(file_dir))
    # 上传完毕后 更新数据库信息
    values['dataset_size'] = size
    values['dataset_path'] = file_dir
    update_dataset_info_by_id(dataset_id, values)


def get_dataset_info_by_id(dataset_id):
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "select * from dataset where dataset_id=%s"
        rows = cursor.execute(sql, (dataset_id))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict_list = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        if len(data_dict_list) >= 1:
            data_dict = data_dict_list[0]
            data_dict['dataset_update_time'] = data_dict['dataset_update_time'].strftime("%Y-%m-%d %H:%M:%S")
            data_dict['dataset_create_time'] = data_dict['dataset_create_time'].strftime("%Y-%m-%d %H:%M:%S")
            print(data_dict)
            return data_dict
        else:
            return None
    except IOError:
        print("Error: Function get_dataset_by_id happen Error")
        return None
    finally:
        cursor.close()
        conn.close()
    return None


# 更新
def update_dataset_info_by_id(dataset_id, values):
    print('update_dataset_info_by_id')
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = 'UPDATE dataset SET {}'.format(', '.join('{}=%s'.format(k) for k in values.keys()))
        sql = sql + " where dataset_id='" + str(dataset_id) + "'"
        rows = cursor.execute(sql, list(values.values()))
        conn.commit()
        if rows > 0:
            return True
        else:
            return False
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function update_dataset_info_by_id happen Error")
        return False
    finally:
        cursor.close()
        conn.close()
    return False


# 删除
def delete_dataset_info_by_id_and_delete_files(dataset_id):
    print('delete_dataset_info_by_id')
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "select * from dataset where dataset_id=%s"
        rows = cursor.execute(sql, (dataset_id))
        # 提交
        conn.commit()
        desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
        data_dict_list = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
        if (len(data_dict_list) >= 1):
            data_dict = data_dict_list[0]
            files_dir = data_dict['dataset_path']
            # 开启一个异步线程去删除文件夹
            delete_file.delay(files_dir)
            sql = 'delete from dataset where dataset_id = %s'
            rows = cursor.execute(sql, (dataset_id))
            conn.commit()
            if rows > 0:
                return True
            else:
                return False
        else:
            return False
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function delete_dataset_info_by_id happen Error")
        return False
    finally:
        cursor.close()
        conn.close()
    return False


# 更新
def insert_dataset_info(values):
    print('insert_dataset_info_by_id')
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 执行那个查询，这里用的是 select 语句
        sql = "insert into dataset({}".format(', '.join('{}'.format(k) for k in values.keys())) + ") values({}".format(
            ', '.join('%s' for k in values.keys())) + ")"
        # sql = "insert into train(train_name,train_create_time,train_update_time,train_record_num,train_status,train_dataset_id,train_ckpt_id,train_task_id,train_took_seconds,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss,train_comment ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        cursor.execute(sql, list(values.values()))
        # 提交
        conn.commit()
        id = cursor.lastrowid  # 获取自增ID号
        return id
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function insert_dataset_info happen Error")
        return -1
    finally:
        cursor.close()
        conn.close()
    return -1


# 删除文件夹
@celery.task(bind=True)
def delete_file(self, filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)
        return True
    else:
        return False


def task_info_tool(task_id):
    task = None
    try:
        task = simple_train.AsyncResult(task_id)
    except TypeError:
        print("Error: Function task_info_tool happen TypeError")
        return None

    # print('task type:',type(task))
    # print('task state:', str(task))
    print('task_info_tool')
    print(task.state)
    if task is None or is_int_number(str(task)):  # 说明这个线程并不存在
        response = None
    elif task.state == 'PREPARE':  # 准备任务
        response = {
            'train_task_id': task_id,
            'train_status': task.state,
            'train_ckpt_id': task.info.get('train_ckpt_id', 0),
            'train_dataset_id': task.info.get('train_dataset_id', 0),
            'train_took_seconds': task.info.get('train_took_seconds', 0),
            'train_epochs': task.info.get('train_epochs', 0),
            'train_epoch': task.info.get('train_epoch', 0),
            'train_steps': task.info.get('train_steps', 0),
            'train_step': task.info.get('train_step', 0),
            'train_acc': task.info.get('train_acc', 0),
            'train_loss': task.info.get('train_loss', 0),
            'train_comment': task.info.get('train_comment', 0)
        }
    elif task.state == 'PROGRESS':  # 正在进行
        response = {
            'train_task_id': task_id,
            'train_status': task.state,
            'train_ckpt_id': task.info.get('train_ckpt_id', 0),
            'train_dataset_id': task.info.get('train_dataset_id', 0),
            'train_took_seconds': task.info.get('train_took_seconds', 0),
            'train_epochs': task.info.get('train_epochs', 0),
            'train_epoch': task.info.get('train_epoch', 0),
            'train_steps': task.info.get('train_steps', 0),
            'train_step': task.info.get('train_step', 0),
            'train_acc': task.info.get('train_acc', 0),
            'train_loss': task.info.get('train_loss', 0),
            'train_comment': task.info.get('train_comment', 0)
        }
    elif task.state == 'SUCCESS':  # 任务完毕
        response = {
            'train_task_id': task_id,
            'train_status': task.state,
            'train_ckpt_id': task.info.get('train_ckpt_id', 0),
            'train_dataset_id': task.info.get('train_dataset_id', 0),
            'train_took_seconds': task.info.get('train_took_seconds', 0),
            'train_epochs': task.info.get('train_epochs', 0),
            'train_epoch': task.info.get('train_epoch', 0),
            'train_steps': task.info.get('train_steps', 0),
            'train_step': task.info.get('train_step', 0),
            'train_acc': task.info.get('train_acc', 0),
            'train_loss': task.info.get('train_loss', 0),
            'train_comment': task.info.get('train_comment', 0)
        }
    elif task.state == 'PENDING':  # 在等待
        response = {
            'train_task_id': task_id,
            'train_status': task.state,
            # 'train_ckpt_id': 1,
            # 'train_dataset_id': 2,
            'train_took_seconds': 0,
            # 'train_epochs': 1,
            # 'train_epoch': 0,
            # 'train_steps': 1,
            # 'train_step': 0,
            # 'train_acc': 0,
            # 'train_loss': 0,
            'train_comment': 'Task is pending!'
        }
    elif task.state == 'FAILURE':  # 没有失败
        response = {
            'train_task_id': task_id,
            'train_status': task.state,
            # 'train_ckpt_id': 1,
            # 'train_dataset_id': 2,
            'train_took_seconds': 0,
            # 'train_epochs': 1,
            # 'train_epoch': 0,
            # 'train_steps': 1,
            # 'train_step': 0,
            # 'train_acc': 0,
            # 'train_loss': 0,
            'train_comment': 'Task is fail!'
        }
    else:
        response = {
            'train_task_id': task_id,
            'train_status': task.state,
            # 'train_ckpt_id': 0,
            # 'train_dataset_id': 0,
            'train_took_seconds': 0,
            # 'train_epochs': 1,
            # 'train_epoch': 0,
            # 'train_steps': 1,
            # 'train_step': 0,
            # 'train_acc': 0,
            # 'train_loss': 0,
            'train_comment': 'Task error:' + str(task.info)
        }

    return response


# 判断是否是整数
def is_int_number(s):
    if s is None:
        return False

    try:
        int(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def is_available(s):
    if s is None:
        return False

    if is_int_number(s):
        return True
    else:
        try:
            if str(s).strip() == '':
                return False
            else:
                return True
        except ValueError:
            return False


def try_time_format(date_text, format):
    try:
        if date_text is None:
            return None
        else:
            dt = datetime.datetime.strptime(date_text, format)
            return dt
    except ValueError:
        print("Incorrect data format, should be ", str(format))
        return None
    return None


#####################################################################################


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
        return jsonify({"success": 1, "msg": format(pred_texts)})
        # return img_fp
        # return jsonify({"success": 0, "msg": "上传成功"})
    else:
        return jsonify({"error": 1001, "msg": "上传失败"})


@app.route("/")
def main():
    return render_template("index.html")


###############################################################################


# 上传训练数据集
# 已实现 批量上传数据 到指定 数据集
@app.route('/up_train_data', methods=['POST'], strict_slashes=False)
def api_upload_train_data():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    # 上传同时，前端会发送数据源Id,根据不同Id，保存到不同数据源
    # dataSourceId=-1，说明是新建的数据源
    # **** 还  需要实现  保存已有数据源的信息，有哪些数据源，
    # 有了数据源后，新建训练就可以选择不同数据源来训练
    dataSourceName = request.values.get("dataSourceName")
    dataSourceId = request.values.get("dataSourceId")
    print('dataSourceName:' + dataSourceName)
    print('dataSourceId:' + dataSourceId)
    print('file_dir:' + file_dir)
    file_dir = os.path.join(file_dir, 'dataSourceId', dataSourceId)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    f_list = request.files.getlist('upData')
    for f in f_list:
        print("filename:" + f.filename)
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            print(fname)
            ext = fname.rsplit('.', 1)[1]
            new_filename = Pic_str().create_uuid() + '.' + ext
            f.save(os.path.join(file_dir, new_filename))

            # ocr = CnOcr()
            # img_fp = os.path.join(file_dir, new_filename)
            # img = mx.image.imread(img_fp, 1)
            # res = ocr.ocr(img)
            # print("Predicted Chars:", res)

            # return jsonify({"success": 0, "msg": res})
            # return jsonify({"success": 0, "msg": "success"})
            # return jsonify({"success": 0, "msg": "上传成功"})
        # else:
        # return jsonify({"error": 1001, "msg": "上传失败"})
    return jsonify({"success": 0, "msg": "success"})


#####################################################################################



if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
    celery.control.purge()
