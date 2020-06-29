import numpy as np
import tensorflow as tf
from celery import Celery
import datetime
import time
from celery import Celery
import pathlib
import os
# from main import insert_new_train_record

# mysql
import pymysql
from DBUtils.PooledDB import PooledDB

# 建立数据库连接池
dbPool = PooledDB(pymysql, 5, host='127.0.0.1', user='root', passwd='123', db='hwr', port=3306)  # 5为连接池里的最少连接数

CKPT_FOLDER = 'static/data/ckpt'
basedir = os.path.abspath(os.path.dirname(__file__))

def long_task(self,train_id,train_modal,dataset_data,ckpt_data):

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

    start = datetime.datetime.now()
    dt =  "["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"]"+ "  Train:"+str(train_id)+" |"
    print(dt+"  Params check done!   dataset_id:" + str(dataset_id) + "  ckpt_id:" + str(ckpt_id)+"  Start prepare for train!")
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
    time.sleep(5)
    dt = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]" + "  Train:" + str(train_id) + " |"
    print(dt+"  Prepare work done!  Start training!")
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
                          'train_loss':0,
                          'train_comment': 'Prepare is done!'
                      })

    epochs = 10
    steps = 10
    for epoch in range(epochs):
        for step in range(steps):
            now = datetime.datetime.now()
            task_took_seconds = int(
                (now - start).total_seconds())
            dt =  "["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"]"+ "  Train:"+str(train_id)+" |"
            print(dt+"  this epoch:" + str(epoch) + " step:" + str(step*100)+"  acc:"+str(round((epoch*steps+step)/(epochs*steps), 5)))
            self.update_state(state='PROGRESS',
                              meta={
                                  'train_ckpt_id': ckpt_id,
                                  'train_dataset_id': dataset_id,
                                  'train_took_seconds': task_took_seconds,
                                  'train_epochs': epochs,
                                  'train_epoch': epoch,
                                  'train_steps': steps*100,
                                  'train_step': step*100,
                                  'train_acc': round((epoch*steps+step)/(epochs*steps), 5),
                                  'train_loss': round((epoch*steps+step)/(epochs*steps), 5),
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
                'train_steps': steps*100,
                'train_step': step*100,
                'train_acc': round((epoch*steps+step)/(epochs*steps), 5),
                'train_loss':round((epoch*steps+step)/(epochs*steps), 5),
                'train_modal':train_modal
            }
            insert_new_train_record(values)
            time.sleep(2)

        ## 模拟保存ckpt，顺便把ckpt的信息保存到数据库
        ckpt_name =  str((epoch+1) * (step+1)) + '.ckpt'
        dt = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]" + "  Train:" + str(train_id) + " |"
        print(dt + " save ckpt name:"+ckpt_name+" this epoch:" + str(epoch) + " step:" + str(step * 100))
        file_dir = os.path.join(basedir, CKPT_FOLDER, str(train_id))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_ab_path = os.path.join(file_dir,ckpt_name)
        pathlib.Path(file_ab_path).touch()
        ckpt_record_values={
            'train_id': train_id,
            'dataset_id': dataset_id,
            'ckpt_path':file_dir,
            'create_time':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ckpt_name':ckpt_name,
            'ckpt_tag':'train',
            'ckpt_status': 'PROGRESS',
            'epoch': epoch,
            'step': step*100,
            'acc': round((epoch*steps+step)/(epochs*steps), 5),
            'loss': round((epoch*steps+step)/(epochs*steps), 5),
            'modal' :train_modal,
            'comment':'this is ckpt record'
        }
        result = insert_new_ckpt_record(ckpt_record_values)
        if result>0:
            ckpt_id = result

    dt = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]" + "  Train:" + str(train_id) + " |"
    print(dt + " Training  done!")

    self.update_state(state='SUCCESS',
                      meta={
                          'train_ckpt_id': ckpt_id,
                          'train_dataset_id': dataset_id,
                          'train_took_seconds': task_took_seconds,
                          'train_epochs': epochs,
                          'train_epoch': epochs,
                          'train_steps': steps*100,
                          'train_step': steps*100,
                          'train_acc': 1,
                          'train_loss': 1,
                          'train_comment': 'Task is completed!'
                      })
    values = {
        'train_id':train_id,
        'train_status':'SUCCESS',
        'update_time':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'train_dataset_id':dataset_id,
        'train_ckpt_id':ckpt_id,
        'train_epochs':epochs,
        'train_epoch':epoch,
        'train_steps':steps*100,
        'train_step':step*100,
        'train_acc':1,
        'train_loss':1,
        'train_modal':train_modal
    }
    insert_new_train_record(values)
    return;


def insert_new_train_record(values):
    print('insert_new_train_record')
    try:
        # 调用连接池
        conn = dbPool.connection()
        # 获取执行查询的对象
        cursor = conn.cursor()
        # 执行那个查询，这里用的是 select 语句
        sql = "insert into train_record({}".format(', '.join('{}'.format(k) for k in values.keys())) + ") values({}".format(
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
        sql = "insert into ckpt({}".format(', '.join('{}'.format(k) for k in ckpt_values.keys())) + ") values({}".format(
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
    print('get_dataset_info_by_id  dataset_id:'+str(dataset_id))
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
        return {"success": 1, "msg": "找到dataset,开始训练!",'dataset_data':ds_data}
    except IOError:
        conn.rollback()  # 出现异常 回滚事件
        print("Error: Function get_dataset_info_by_id happen Error")
        return {"success": 0, "msg": "get_dataset_info_by_id Mysql error!"}
    finally:
        cursor.close()
        conn.close()
    return {"success": 0, "msg": "get_dataset_info_by_id fail!"}


def get_ckpt_info_by_id(ckpt_id):
    print('get_ckpt_info_by_id  ckpt_id:'+str(ckpt_id))
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
