import numpy as np
import torch
from torch.utils.data import Dataset
import time
from datetime import datetime
import pandas as pd


GRAVITATIONAL_ACCELERATION = 9.80665


class base_loader(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)


def convert_unit_system(args, df):
    """
    The unit systems of output CSVs (E4-preprocessing/zip_to_csv.csv) are follows.
    # if logi dataset, use another cols...
    * acc: [m/s^2] (raw value[G] are mutiplied with 9.80665)
    * gyro: [dps]
    * quat: [none]
    """
    # ACC: [m / s ^ 2] = > [G]

    if args.dataset== 'logi':
        cols = ['LW_x', 'LW_y', 'LW_z']
        # cols = ['LW_x', 'LW_y', 'LW_z', 'RW_x', 'RW_y', 'RW_z']
    elif args.dataset=='openpack':
        cols = ["acc_x", "acc_y", "acc_z"]
    elif args.dataset=='ome':
        cols = ["x", "y", "z"]
    else:
        NotImplementedError
    # allcols = list(df.columns.values)
    # cols = list(set(allcols) - set(['time']))
    df[cols] = df[cols] / GRAVITATIONAL_ACCELERATION

    return df


def generate_labels_for_each_point(tmpdata, label):
    '''
    :param data: ['time',...]. The attribute box corresponds to period id
    :param label: label[['start','end','id']]
    两者都是timestamp，label是1Hz，data为30Hz。通过拼接两个表得到每个点的label
    :return:
    '''
    # TODO 对齐好像有bug
    data = tmpdata.copy()
    data['labelT'] = data['time'].apply(lambda x:int(x/1000))  # u0104,0105,0108,0109
    label['time'] = label['time'].apply(lambda x:int(x))  # u0104,0105,0108,0109
    # 填充空时间戳
    new_range = pd.DataFrame({'time': range(label['time'].values[0], label['time'].values[-1])})
    merged = new_range.merge(label, on='time', how='left')
    merged[['operation', 'box']] = merged[['operation', 'box']].fillna(method='ffill')
    newdata = data.join(merged.set_index('time'), on='labelT')  # u0104,0105,0108,0109
    ope_dict = {'Picking':100, 'Relocate Item Label':200,
                'Assemble Box':300, 'Insert Items':400,
                'Close Box':500, 'Attach Box Label':600,
                'Scan Label':700, 'Attach Shipping Label':800,
                'Fill out Order':1000, 'Put on Back Table':900,0:0}
    newdata_cleaned = newdata.dropna()
    # newdata_cleaned = newdata.dropna(subset=['operation', 'box'])
    # newdata['operation'] = newdata['operation'].fillna(0)
    # newdata['box'] = newdata['box'].fillna(0)
    newdata_cleaned = newdata_cleaned[newdata_cleaned.operation != "Null"]
    newdata_cleaned.operation = newdata_cleaned.operation.map(lambda x:ope_dict[x])

    # other datasets
    # data['labelT'] = data['time'].apply(lambda x:int(x/1000)*1000)
    # https: // www.modb.pro / db / 177052
    # newdata = data.join(label.set_index('time'), on='labelT')
    # newdata['operation'] = newdata['operation'].fillna(0)
    # newdata['box'] = newdata['box'].fillna(0)
    newlabel = newdata_cleaned[['time', 'operation', 'box']]
    return newlabel, newdata_cleaned[['time','acc_xl','acc_yl','acc_zl','acc_xr','acc_yr','acc_zr']]



def transfer_str2time(x):
    try:
        result = time.mktime(datetime.strptime(x, '%Y%m%d_%H:%M:%S.%f').timetuple())
    except:
        result = time.mktime(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f').timetuple())
    return result


def event2ope(eventarray):
    '''
    eventarray: job0-jumbi
    return: 0 as period, jumbi as opertion
    '''
    operation_dict = {'jumbi':0, 'neji1':1, 'neji2':2, 'neji3':3,
                      'neji4':4, 'neji5':5, 'neji6':6, 'neji7':7,
                      'neji8':8, 'neji9&button':9, 'haraidashi':10,
                      'haraidashi1':10, 'waiting':11}  # 改了ope

    periods = np.zeros(eventarray.shape)
    operations = np.zeros(eventarray.shape)
    for i in range(eventarray.shape[0]):
        [job, ope] = eventarray[i].split("-", 1)
        periods[i] = int(job[3:])
        operations[i] = operation_dict[ope]
    return periods, operations


def event2ope_ome(eventarray):
    '''
    eventarray: job0-jumbi
    return: 0 as period, jumbi as opertion
    '''
    operation_dict = {'kibantoritsuke':0, 'ABB':1, 'RBOffset':2, 'ukedai':3,
                      'AGC&FlexFilter':4, 'Blooming':5, 'Head':6, 'kibanharaidashi':7}  # 改了ope

    periods = np.zeros(eventarray.shape)
    operations = np.zeros(eventarray.shape)
    for i in range(eventarray.shape[0]):
        [job, ope] = eventarray[i].split("-", 1)
        periods[i] = int(job[3:])
        operations[i] = operation_dict[ope]
    return periods, operations


def match_data_from_label(data, label):
    # label[['time', 'period', 'operation']].values
    ope = -1 * np.ones(data.shape)
    period = -1 * np.ones(data.shape)
    # todo 需要自己处理标签，让第一个周期对上data
    for idx in range(len(data)):
        for l_idx in range(len(label) - 1):
            # 数据的时间在label时间中间
            if (data[idx] >= label[l_idx][0]) and (data[idx] < label[l_idx + 1][0]):
                ope[idx] = label[l_idx][2]
                period[idx] = label[l_idx][1]
    return ope, period

def generate_labels_for_each_point_neji(tmpdata, label):
    '''
    toshiba data, timestamp different from openpack
    l_df.columns=['event_type', 'starttime', '_'] job0-jumbi, 2016-03-09T22:04:06.369
    label needs to generate operation information from event_type
    data_df.columns=["timeP","acc_xr", "acc_yr", "acc_zr"] 20160309_22:05:00.964000
    :return: fullabel
    '''
    # -----------transfer time---------------
    label['time'] = label['starttime'].map(transfer_str2time)
    tmpdata['time'] = tmpdata['timeP'].map(transfer_str2time)

    #------------tansfer event_type----------
    tmplabels = label.event_type.values
    periods, operations = event2ope(tmplabels)
    label['period'] = periods
    label['operation'] = operations

    # -----------concatenate-----------------
    tmpl = label[['time', 'period', 'operation']].values
    tmpd = tmpdata['time'].values
    ope, peri = match_data_from_label(tmpd, tmpl)

    tmpdata['operation'] = ope
    tmpdata['period'] = peri
    # 去掉所有ope=-1的行
    newdata = tmpdata.drop(tmpdata[tmpdata.operation==-1].index)
    newlabel = newdata[['time', 'operation', 'period']]
    newdata = newdata[['time', 'acc_xr', 'acc_yr', 'acc_zr']]
    return newlabel, newdata



def generate_labels_for_each_point_ome(tmpdata, label):
    '''
    toshiba data, timestamp different from openpack
    l_df.columns=['event_type', 'starttime', '_'] job0-jumbi, 2016-03-09T22:04:06.369
    label needs to generate operation information from event_type
    data_df.columns=["timeP","acc_xr", "acc_yr", "acc_zr"] 20160309_22:05:00.964000
    :return: fullabel
    '''
    # -----------transfer time---------------
    label['time'] = label['starttime'].map(transfer_str2time)
    tmpdata['time'] = tmpdata['timeP'].map(transfer_str2time)

    #------------tansfer event_type----------
    tmplabels = label.event_type.values
    periods, operations = event2ope_ome(tmplabels)
    label['period'] = periods
    label['operation'] = operations

    # -----------concatenate-----------------
    tmpl = label[['time', 'period', 'operation']].values
    tmpd = tmpdata['time'].values
    ope, peri = match_data_from_label(tmpd, tmpl)

    tmpdata['operation'] = ope
    tmpdata['period'] = peri
    # 去掉所有ope=-1的行
    newdata = tmpdata.drop(tmpdata[tmpdata.operation==-1].index)
    newlabel = newdata[['time', 'operation', 'period']]
    newdata = newdata[['time', 'acc_xr', 'acc_yr', 'acc_zr']]
    return newlabel, newdata