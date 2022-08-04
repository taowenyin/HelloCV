# coding=utf8

import argparse
import numpy as np
import pandas as pd
import datetime
import os

from tqdm import tqdm
from alive_progress import alive_bar

# V0.2 2022.8.4
# 1. 需要安装的依赖包
# conda install openpyxl Pandas
# pip install tqdm alive-progress
# 2. 指令示例
# python parse_acc_data.py --data-folder /Users/taowenyin/Database/jiangnan --axes Timestamp Z MAX_X

def half_time2full_time(data_time):
    # 转化为标准时间
    data_time = data_time.replace(',', ' ')
    pm_am = data_time[-2:]
    if pm_am == '下午':
        data_time = data_time.replace('下午', 'PM')
    else:
        data_time = data_time.replace('上午', 'AM')
    # 转换时间为24小时制
    data_time = datetime.datetime.strptime(
        data_time, '%m月 %d  %Y %I:%M:%S %p').strftime('%Y-%m-%d %H:%M:%S')

    return data_time


def parse_data_file(root_path, data_file, axes, save_path, suffix):
    # 读取CSV文件，以“，”作为间隔
    file_data = pd.read_csv(os.path.join(root_path, data_file), sep=',', encoding='utf-8', header=0)
    file_name, _ = os.path.splitext(data_file)
    # 读取原始数据
    axes = [ax.replace('_', ' ') for ax in axes]
    org_data = file_data[axes]
    new_data = None

    for i in tqdm(range(len(org_data)), desc='处理{}中的数据'.format(data_file), leave=False):
        data_list = []
        fields_list = []
        fields_top_list = []

        # 逐行读取数据
        item_data = org_data.iloc[i]
        # 读取每个字段的数据
        for _, field in enumerate(axes):
            if field in ['Timestamp', 'MAX X', 'MAX Y', 'MAX Z', 'RMS X', 'RMS Y', 'RMS Z']:
                field_data = item_data[field]
                if field == 'Timestamp':
                    # 把12小时制转化为24小时制
                    field_data = [half_time2full_time(field_data)]
                else:
                    field_data = [field_data]
                # 获取顶部字段
                if i == 0:
                    field_top_data_labels_list = [field]

                field_data_labels_list = [field]
            else:
                # field_data = item_data[field].split(' ')
                field_data = [float(data) for data in item_data[field].split(' ')]
                # 获取顶部字段
                if i == 0:
                    field_top_data_labels_list = [(field + str(j + 1)) for j in range(2048)]

                field_data_labels_list = [(field + str(j + 1)) for j in range(len(field_data))]

            if i == 0:
                fields_top_list.extend(field_top_data_labels_list)

            # 创建字段和所有数据
            fields_list.extend(field_data_labels_list)
            data_list.extend(field_data)

        # 在新表中创建完整字段
        if new_data is None:
            new_data = pd.DataFrame(columns=fields_top_list)

        # 插入分割好的数据
        insert_data = pd.DataFrame([data_list], columns=fields_list)
        new_data = pd.concat([new_data, insert_data], ignore_index=True)

    # 只有读取时间数据才计算差值
    if 'Timestamp' in axes:
        # 获取所有时间
        data_time_list = [datetime.datetime.strptime(data_time, '%Y-%m-%d %H:%M:%S')
                          for data_time in new_data['Timestamp']]
        # 保存时间差的数据
        diff_list = [0]
        for i in range(len(data_time_list)):
            if i == 0:
                continue
            # 计算时间差并添加到列表
            diff = (data_time_list[i] - data_time_list[i - 1]).seconds
            diff_list.append(diff)
        # 插入时间差的列
        new_data.insert(1, "diff", np.array(diff_list))

    with alive_bar(title='向{}写入新数据...'.format(str(file_name + suffix))):
        new_data.to_excel(os.path.join(save_path, file_name + suffix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运输数据分析')
    parser.add_argument('--data-folder', required=True, help='数据文件夹的路径')
    parser.add_argument('--save-path', help='解析后的数据保存路径，默认情况下和读取路径一致')
    parser.add_argument('--suffix', default='_new.xlsx', help='处理后的数据后缀名')
    parser.add_argument('--axes', nargs='+', default='all', help='要提取的数据索引，默认情况下全部读取')
    args = parser.parse_args()

    # 获取所有需要的参数
    root_path = args.data_folder
    suffix = args.suffix
    axes = args.axes
    if args.save_path is None:
        save_path = root_path
    else:
        save_path = args.save_path

    # 读取所有数据文件，并逐一处理
    data_files_list = os.listdir(root_path)
    for data_file in tqdm(data_files_list, desc='读取的文件数量'):
        _, file_ext = os.path.splitext(data_file)
        if file_ext == ".csv" and not data_file.endswith(suffix):
            parse_data_file(root_path, data_file, axes, save_path, suffix)

    print('===>数据全部处理完成...')
