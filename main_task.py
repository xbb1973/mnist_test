import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#### hly celery  begin
from celery import Celery
import datetime
import time
from celery import Celery
import pathlib
import os

# mysql
import pymysql
from DBUtils.PooledDB import PooledDB

# 建立数据库连接池
dbPool = PooledDB(pymysql, 5, host='127.0.0.1', user='root', passwd='123', db='hwr', port=3306)  # 5为连接池里的最少连接数

CKPT_FOLDER = 'static/data/ckpt'
basedir = os.path.abspath(os.path.dirname(__file__))


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


def insert_new_ckpt_record(ckpt_values):
    print('insert_new_ckpt_record')
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "insert into ckpt({}".format(
            ', '.join('{}'.format(k) for k in ckpt_values.keys())) + ") values({}".format(
            ', '.join('%s' for k in ckpt_values.keys())) + ")"
        # sql = "insert into train(train_name,train_create_time,train_update_time,train_record_num,train_status,train_dataset_id,train_ckpt_id,train_task_id,train_took_seconds,train_epochs,train_epoch,train_steps,train_step,train_acc,train_loss,train_comment ) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        cursor.execute(sql, list(ckpt_values.values()))
        # 提交
        conn.commit()
        id = cursor.lastrowid  # 获取自增ID号
        return id
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function insert_new_ckpt_record happen Error")
        return -1
    finally:
        cursor.close()
        conn.close()
    return -1


def get_dataset_info_by_id(dataset_id):
    print('get_dataset_info_by_id  dataset_id:' + str(dataset_id))
    if dataset_id is None or not is_int_number(dataset_id):
        return {"success": 0, "msg": "无法开始训练! DS Id不正确!"}
    dataset_id = int(dataset_id)
    ds_data = None
    try:
        conn = dbPool.connection()
        cursor = conn.cursor()
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
        return {"success": 1, "msg": "找到dataset,开始训练!", 'dataset_data': ds_data}
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function get_dataset_info_by_id happen Error")
        return {"success": 0, "msg": "get_dataset_info_by_id Mysql error!"}
    finally:
        cursor.close()
        conn.close()
    return {"success": 0, "msg": "get_dataset_info_by_id fail!"}


def get_ckpt_info_by_id(ckpt_id):
    print('get_ckpt_info_by_id  ckpt_id:' + str(ckpt_id))
    if ckpt_id is None or not is_int_number(ckpt_id):
        return None
    ckpt_id = int(ckpt_id)
    ckpt_data = None
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        ## 检测ckpt并获取
        if ckpt_id > 0:
            sql = "select * from ckpt where ckpt_id=%s"
            cursor.execute(sql, (ckpt_id))
            # 提交
            conn.commit()
            desc = cursor.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
            ckpt_data_dict = [dict(zip([col[0] for col in desc], row)) for row in cursor.fetchall()]  # 列表表达式把数据组装起来
            if len(ckpt_data_dict) <= 0:
                return None
            else:
                ckpt_data = ckpt_data_dict[0]
                ckpt_path = ckpt_data['ckpt_path']
                if not os.path.exists(ckpt_path):
                    sql = "delete from ckpt where ckpt_id=%s"
                    rows = cursor.execute(sql, (ckpt_data['ckpt_id']))
                    # 提交
                    conn.commit()
                    return None

        return ckpt_data
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function get_ckpt_info_by_id happen Error")
        return None
    finally:
        cursor.close()
        conn.close()
    return None


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


#### hly celery end

# 定义初始化权重的函数
def weight_variavles(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return w


# 定义一个初始化偏置的函数
def bias_variavles(shape):
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b


def model():
    # 1.建立数据的占位符 x [None, 784]  y_true [None, 10]
    with tf.variable_scope("date"):
        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.float32, [None, 10])

    # 2.卷积层1  卷积:5*5*1,32个filter,strides= 1-激活-池化
    with tf.variable_scope("conv1"):
        # 随机初始化权重
        w_conv1 = weight_variavles([5, 5, 1, 32])
        b_conv1 = bias_variavles([32])

        # 对x进行形状的改变[None, 784] ----- [None,28,28,1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])  # 不能填None,不知道就填-1

        # [None,28, 28, 1] -------- [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        # 池化 2*2，步长为2，【None, 28,28, 32]--------[None,14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3.卷积层2  卷积:5*5*32,64个filter,strides= 1-激活-池化
    with tf.variable_scope("conv2"):
        # 随机初始化权重和偏置
        w_conv2 = weight_variavles([5, 5, 32, 64])
        b_conv2 = bias_variavles([64])

        # 卷积、激活、池化
        # [None,14, 14, 32]----------【NOne, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # 池化 2*2，步长为2 【None, 14,14，64]--------[None,7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4.全连接层 [None,7, 7, 64] --------- [None, 7*7*64] * [7*7*64, 10]+[10] = [none, 10]
    with tf.variable_scope("fc"):
        # 随机初始化权重和偏置:
        w_fc = weight_variavles([7 * 7 * 64, 1024])
        b_fc = bias_variavles([1024])

        # 修改形状 [none, 7, 7, 64] ----------[None, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(x_fc_reshape, w_fc) + b_fc)

        # 在输出之前加入dropout以减少过拟合
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc1 = weight_variavles([1024, 10])
        b_fc1 = bias_variavles([10])

        # 进行矩阵运算得出每个样本的10个结果[NONE, 10]，输出
        y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc1) + b_fc1)

    return x, y_true, y_predict, keep_prob


# hly celery  参数得加入 self
# def conv_fc()
def conv_fc(self, train_id, train_modal, dataset_data, ckpt_data):
    ## hly celery begin
    ckpt_id = 0
    ckpt_path = None
    if ckpt_data is not None:
        print('ckpt_data:', ckpt_data)
        ckpt_id = int(ckpt_data['ckpt_id'])
        ckpt_path = ckpt_data['ckpt_path']
    dataset_id = 0
    dataset_path = None
    if dataset_data is not None:
        print('dataset_data:', dataset_data)
        dataset_id = int(dataset_data['dataset_id'])
        dataset_path = dataset_data['dataset_path']
    else:
        self.update_state(state='FAILURE',
                          meta={
                              'train_ckpt_id': ckpt_id,
                              'train_dataset_id': dataset_id,
                              'train_took_seconds': 0,
                              'train_epochs': 1,
                              'train_epoch': 0,
                              'train_steps': 1,
                              'train_step': 0,
                              'train_acc': 0,
                              'train_loss': 0,
                              'train_comment': 'dataset_data is None!'
                          })
        return
    print("simple_train-- dataset_id:" + str(dataset_id) + "  ckpt_id:" + str(ckpt_id))
    self.update_state(state='PREPARE',
                      meta={
                          'train_ckpt_id': ckpt_id,
                          'train_dataset_id': dataset_id,
                          'train_took_seconds': 0,
                          'train_epochs': 1,
                          'train_epoch': 0,
                          'train_steps': 1,
                          'train_step': 0,
                          'train_acc': 0,
                          'train_loss': 0,
                          'train_comment': 'Prepare for new train!'
                      })
    ## hly celery end

    # 获取数据，MNIST_data是楼主用来存放官方的数据集，如果你要这样表示的话，那MNIST_data这个文件夹应该和这个python文件在同一目录
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    ## hly celery begin
    self.update_state(state='PREPARE',
                      meta={
                          'train_ckpt_id': ckpt_id,
                          'train_dataset_id': dataset_id,
                          'train_took_seconds': 0,
                          'train_epochs': 1,
                          'train_epoch': 0,
                          'train_steps': 1,
                          'train_step': 0,
                          'train_acc': 0,
                          'train_loss': 0,
                          'train_comment': 'Prepare is done!'
                      })
    ## hly celery end

    # 定义模型，得出输出
    x, y_true, y_predict, keep_prob = model()

    # 进行交叉熵损失计算
    # 3.计算交叉熵损失
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失,tf.reduce_mean对列表求平均值
        loss = -tf.reduce_sum(y_true * tf.log(y_predict))

    # 4.梯度下降求出最小损失,注意在深度学习中，或者网络层次比较复杂的情况下，学习率通常不能太高
    with tf.variable_scope("optimizer"):

        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # 5.计算准确率
    with tf.variable_scope("acc"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # equal_list None个样本 类型为列表1为预测正确，0为预测错误[1, 0, 1, 0......]

        accuray = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)
        epochs = 10
        steps = 3000
        epochs_accuracy = 0
        for epoch in range(epochs):
            epochs_accuracy = 0
            mnist_x, mnist_y = mnist.train.next_batch(50)
            for step in range(steps):
                if step % 100 == 0:
                    start = datetime.datetime.now()
                    # 评估模型准确度，此阶段不使用Dropout
                    train_accuracy = accuray.eval(feed_dict={x: mnist_x, y_true: mnist_y, keep_prob: 1.0})
                    epochs_accuracy +=train_accuracy
                    print("step %d, training accuracy %g" % (step, train_accuracy))
                    ## hly celery begin
                    now = datetime.datetime.now()
                    task_took_seconds = int(
                        (now - start).total_seconds())
                    self.update_state(state='PROGRESS',
                                      meta={
                                          'train_ckpt_id': ckpt_id,
                                          'train_dataset_id': dataset_id,
                                          'train_took_seconds': task_took_seconds,
                                          'train_epochs': epochs,
                                          'train_epoch': epoch,
                                          'train_steps': steps,
                                          'train_step': step,
                                          'train_acc': train_accuracy,
                                          'train_loss': train_accuracy,
                                          'train_comment': 'Task is running!'
                                      })
                    values = {
                        'train_id': train_id,
                        'train_status': 'PROGRESS',
                        'update_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'train_dataset_id': dataset_id,
                        'train_ckpt_id': ckpt_id,
                        'train_epochs': epochs,
                        'train_epoch': epoch,
                        'train_steps': steps,
                        'train_step': step,
                        'train_acc': train_accuracy,
                        'train_loss': train_accuracy,
                        'train_modal': train_modal
                    }
                    insert_new_train_record(values)
                    ## hly celery end

                # 训练模型，此阶段使用50%的Dropout
                train_op.run(feed_dict={x: mnist_x, y_true: mnist_y, keep_prob: 0.5})
            epochs_accuracy = epochs_accuracy/(steps/100)
            ## hly celery begin
            ## 下面自定义ckpt文件名，也可以根据情况保存ckpt后获取保存的文件名再存入数据库
            ckpt_name = str((epoch + 1) * (step + 1)) + '.ckpt'
            file_dir = os.path.join(basedir, CKPT_FOLDER, str(train_id))
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_ab_path = os.path.join(file_dir, ckpt_name)
            ckpt_record_values = {
                'train_id': train_id,
                'dataset_id': dataset_id,
                'ckpt_path': file_dir,
                'create_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'ckpt_name': ckpt_name,
                'ckpt_tag': 'train',
                'ckpt_status': 'PROGRESS',
                'epoch': epoch,
                'step': step,
                'acc': epochs_accuracy,
                'loss': epochs_accuracy,
                'modal': train_modal,
                'comment': 'this is ckpt record'
            }
            result = insert_new_ckpt_record(ckpt_record_values)
            if result > 0:
                ckpt_id = result

            ## hly celery end

            # 将模型保存在你自己想保存的位置
            saver.save(sess, file_ab_path)
    return None



# hly celery  参数得加入 self
def conv_fc_pure():
    # 获取数据，MNIST_data是楼主用来存放官方的数据集，如果你要这样表示的话，那MNIST_data这个文件夹应该和这个python文件在同一目录
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 定义模型，得出输出
    x, y_true, y_predict, keep_prob = model()

    # 进行交叉熵损失计算
    # 3.计算交叉熵损失
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失,tf.reduce_mean对列表求平均值
        loss = -tf.reduce_sum(y_true * tf.log(y_predict))

    # 4.梯度下降求出最小损失,注意在深度学习中，或者网络层次比较复杂的情况下，学习率通常不能太高
    with tf.variable_scope("optimizer"):

        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # 5.计算准确率
    with tf.variable_scope("acc"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # equal_list None个样本 类型为列表1为预测正确，0为预测错误[1, 0, 1, 0......]

        accuray = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)
        epochs = 10
        steps = 3000
        epochs_accuracy = 0
        for epoch in range(epochs):
            epochs_accuracy = 0
            mnist_x, mnist_y = mnist.train.next_batch(50)
            for step in range(steps):
                if step % 100 == 0:
                    start = datetime.datetime.now()
                    # 评估模型准确度，此阶段不使用Dropout
                    train_accuracy = accuray.eval(feed_dict={x: mnist_x, y_true: mnist_y, keep_prob: 1.0})
                    epochs_accuracy +=train_accuracy
                    print("step %d, training accuracy %g" % (step, train_accuracy))

                # 训练模型，此阶段使用50%的Dropout
                train_op.run(feed_dict={x: mnist_x, y_true: mnist_y, keep_prob: 0.5})
            epochs_accuracy = epochs_accuracy/(steps/100)
            ## hly celery begin
            ## 下面自定义ckpt文件名，也可以根据情况保存ckpt后获取保存的文件名再存入数据库
            ckpt_name = str((epoch + 1) * (step + 1)) + '.ckpt'
            file_dir = os.path.join(basedir, CKPT_FOLDER, str(train_id))
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_ab_path = os.path.join(file_dir, ckpt_name)


            ## hly celery end

            # 将模型保存在你自己想保存的位置
            saver.save(sess, file_ab_path)
    return None

if __name__ == "__main__":
    conv_fc_pure()
