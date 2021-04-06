import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import pathlib as pl
from utils import read_data_cfg

def count_people(json_path):

    #output_path = 'K:/dataset/flir_output dataset/'
    #category_dict_path = 'K:/dataset/flir dataset/train/thermal_annotations.json'
    with open(json_path) as json_file:
        data = json.load(json_file)
        categories = ['person','car','bicycle','dog']
        cat = pd.DataFrame(data['categories']).rename(columns={'id':'category_id','name':'category'})
        #for c in categories:
        annotations = pd.DataFrame(data['annotations'])
        images = pd.DataFrame(data['images']).rename(columns={'id':'image_id'})
        df = annotations.merge(cat,how='left',on=['category_id'])
        #df['category'] = df['category'].fillna('empty')
        #g = df.groupby(['image_id','category']).size().reset_index(name='count').groupby(['count','category']).size().reset_index(name='count_c')
        #g['count_c'] = g[['count_c']].apply(lambda x: x/x.sum()*100)
        return(df)

def count_kaist_people(data_path):
    options = read_data_cfg(data_path)
    train_file = options['train']
    #output_path = 'K:/dataset/flir_output dataset/'
    #category_dict_path = 'K:/dataset/flir dataset/train/thermal_annotations.json'
    data = []
    with open(train_file) as tf:
        images = tf.readlines()
    for i in images:
        labpath = i.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png','.txt').replace('.tif', '.txt')
        txt = pl.Path(labpath.rstrip())
        if txt.exists():
            with txt.open('r') as t:
                size = len(t.readlines())
            data.append({'file_name':i.rstrip(),'size':size})
    df = pd.DataFrame(data).groupby('size').count()
    df.plot(grid=True,marker='o',markevery=2,ylabel='number of annotations',xlabel='annotations per image')
    plt.xlabel('annotations per image')
    plt.ylabel('number of annotations')
    plt.legend(['person'])
    plt.show()
    # # Close file
    # rd.close()
    # data = json.load(json_file)
    # categories = ['person','car','bicycle','dog']
    # cat = pd.DataFrame(data['categories']).rename(columns={'id':'category_id','name':'category'})
    # #for c in categories:
    # annotations = pd.DataFrame(data['annotations'])
    # images = pd.DataFrame(data['images']).rename(columns={'id':'image_id'})
    # df = annotations.merge(cat,how='left',on=['category_id'])
    # #df['category'] = df['category'].fillna('empty')
    # #g = df.groupby(['image_id','category']).size().reset_index(name='count').groupby(['count','category']).size().reset_index(name='count_c')
    # #g['count_c'] = g[['count_c']].apply(lambda x: x/x.sum()*100)
    # return(df)

def square_mean_loss(annotation_json,detection_json):
    with open(annotation_json) as ann_file:
        ann_data = json.load(ann_file)
    with open(detection_json) as det_file:
        det_data = json.load(det_file)
    category_dict = {
        1:"person",
        2:"bicycle",
        3:"car",
        17:"dog"
    }
    # cat = pd.DataFrame(ann_data['categories']).rename(columns={'id':'category_id','name':'category'})
    images = pd.DataFrame(ann_data['images']).rename(columns={'id':'image_id'})
    ann_df = pd.DataFrame(ann_data['annotations'])

    #ann_df = pd.merge(pd.DataFrame(ann_data['annotations']),cat,on=['category_id'],how='inner')
    ann_df = pd.merge(images,ann_df,on=['image_id'],how='left')
    ann_df['file_name'] = ann_df['file_name'].apply(lambda x: pl.Path(x).stem)
    #ann_df['image_id'] = ann_df['image_id'].astype('category')
    ann_df = ann_df.groupby(['file_name','category_id']).size().reset_index(name='labels').rename(columns={'file_name':'image_id'})
    ann_df = ann_df[ann_df['category_id'] == 1]
    det_df = pd.DataFrame(det_data)
    det_df['score'] = det_df['score'].round(2)
    mses = {}
    mse_by_num = {}
    # positive = {}
    # negative = {}
    for cat_id in sorted(ann_df['category_id'].unique()):
        mses[cat_id] = []
        mse_by_num[cat_id] = []
        confs = sorted(det_df[det_df.category_id == cat_id].score.unique())
        for score_thresh in confs:
            # positive[cat_id] = []
            # negative[cat_id] = []
            det_df_c = det_df[(det_df['score'] >= score_thresh) & (det_df['category_id']==cat_id)].groupby(['image_id','category_id']).size().reset_index(name='predictions')
            df = ann_df[ann_df['category_id'] == cat_id].merge(det_df_c, how='left',
                                                               on=['image_id', 'category_id']).fillna(0)
            mse = mean_squared_error(df['predictions'], df['labels'], squared=True)

            #
            mses[cat_id].append((score_thresh,mse))
            # negative[cat_id].append((df[df['count_x'] - df['count_y']>0].count_x - df[df['count_x'] - df['count_y']>0].count_y).sum())
            # positive[cat_id].append((df[df['count_x'] - df['count_y']<0].count_y - df[df['count_x'] - df['count_y']<0].count_x).sum())
        if len(mses[cat_id]) > 0:
            best,best_mse = min(mses[cat_id],key= lambda t:t[1])
            print(category_dict[cat_id], ' MSE for conf={:.2f}: {:.4f}'.format(best,best_mse))
            det_df_c = det_df[(det_df.score >= best) & (det_df.category_id == cat_id)].groupby(
                ['image_id', 'category_id']).size().reset_index(name='predictions')
            df = ann_df[ann_df['category_id'] == cat_id].merge(det_df_c, how='left',
                                                               on=['image_id', 'category_id']).fillna(0)
            nums = sorted(df.labels.unique())
            for i in nums:
                mse_by_num[cat_id].append((i,mean_squared_error(df[(df.labels == i) & (df.category_id == cat_id)].labels,
                                                    df[(df.labels == i) & (df.category_id == cat_id)].predictions,
                                                    squared=True)))
        #plt.plot(conf,positive[cat_id],label=category_dict[cat_id])
        # plt.plot(conf,negative[cat_id],label=category_dict[cat_id])
    fig = plt.figure()
    for cat_id in mses.keys():
        plt.plot([t[0] for t in mses[cat_id]], [t[1] for t in mses[cat_id]], label=category_dict[cat_id])
        if cat_id ==1:
            mse_arr = np.array(mses[cat_id])
            np.save(pl.Path.joinpath(pl.Path(detection_json).parent,'mse_'+category_dict[cat_id]),mse_arr)
    plt.xlim(0.005, 1)
    plt.grid()
    plt.legend()
    plt.xlabel('confidence')
    plt.ylabel('mse')
    plt.savefig(pl.Path.joinpath(pl.Path(detection_json).parent,'mse.png'))

    fig = plt.figure()
    for cat_id in mse_by_num.keys():
        plt.plot([t[0] for t in mse_by_num[cat_id]], [t[1] for t in mse_by_num[cat_id]], label=category_dict[cat_id])
        if cat_id ==1:
            mse_by_num_arr = np.array(mse_by_num[cat_id])
            np.save(pl.Path.joinpath(pl.Path(detection_json).parent,'mse_by_num_'+category_dict[cat_id]),mse_by_num_arr)
    plt.grid()
    plt.legend()
    plt.xlabel('number of objects')
    plt.ylabel('mse')
    plt.savefig(pl.Path.joinpath(pl.Path(detection_json).parent,'mse_by_num.png'))
    return()

if __name__ == '__main__':
    #count_kaist_people('data/kaist.data')
    # json_path = 'D:/dataset/flir_dataset/train/thermal_annotations.json'
    # df = count_people(json_path)
    # df.groupby('category').size().plot(kind='bar',rot=45,ylabel='number of annotations')
    #     #plt.plot(g[g.category==cat]['count'],g[g.category==cat].count_c,label=cat,marker='o',markevery=5)
    # axes = plt.gca()
    # axes.yaxis.grid()
    # plt.show()
    # #Group detections by
    # plottable = df.groupby(['image_id','category']).size().reset_index(name='count').groupby(['category','count']).size().reset_index(name='count_c').pivot(index='count', columns='category', values='count_c')
    # plottable.plot(grid=True,marker='o',markevery=5,xticks=np.arange(0,20,step=3),ylabel='number of annotations',xlabel='annotations per image')
    # plt.show()
    # square_mean_loss('D:/dataset/flir_dataset/val/thermal_annotations.json','D:/results/evaluation/kaist/valid/detection_results.json')

    kaist_mse = np.load('D:/results/evaluation/kaist/train/mse_by_num_person.npy')
    flir_mse = np.load('D:/results/evaluation/kaist/valid/mse_by_num_person.npy')
    plt.plot(kaist_mse[:,0],kaist_mse[:,1], label='train')
    plt.plot(flir_mse[:,0],flir_mse[:,1], label='test')
    # plt.xlim(0.005, 1)
    plt.grid()
    plt.xlabel('number of objects/image')
    plt.ylabel('mse')
    plt.legend()
    # plt.show()
    plt.savefig('D:/results/evaluation/mse_by_num_kaist_performance.png')

    #mse_by_det_num('K:/dataset/flir_dataset/val/thermal_annotations.json','K:/results/test_kaist_evaluation_0005/detection_results.json',0.3590)