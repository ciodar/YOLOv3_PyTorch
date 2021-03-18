import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import pathlib as pl

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
    cat = pd.DataFrame(ann_data['categories']).rename(columns={'id':'category_id','name':'category'})
    images = pd.DataFrame(ann_data['images']).rename(columns={'id':'image_id'})
    ann_df = pd.DataFrame(ann_data['annotations'])
    #ann_df = pd.merge(pd.DataFrame(ann_data['annotations']),cat,on=['category_id'],how='inner')
    ann_df = pd.merge(images,ann_df,on=['image_id'],how='left')
    ann_df['file_name'] = ann_df['file_name'].apply(lambda x: pl.Path(x).stem)
    #ann_df['image_id'] = ann_df['image_id'].astype('category')
    ann_df = ann_df.groupby(['file_name','category_id']).size().reset_index(name='count').rename(columns={'file_name':'image_id'})
    det_df = pd.DataFrame(det_data)
    det_df['score'] = det_df['score'].round(2)
    det_df = det_df.groupby(['image_id','category_id','score']).size().reset_index(name='count')
    plottable = det_df.pivot(index=['score','category_id'],values='count')
    plottable.plot()
    plt.show()
    conf = np.linspace(0,1,40)
    mse = {}
    positive = {}
    negative = {}
    for cat_id in sorted(ann_df['category_id'].unique()):
        mse[cat_id] = []
        positive[cat_id] = []
        negative[cat_id] = []
        for i in conf:
            det_df_c = det_df[(det_df['score']>=i) & (det_df['category_id']==cat_id)].groupby(['image_id','category_id']).size().reset_index(name='count')
            df = ann_df[ann_df['category_id']==cat_id].merge(det_df_c,how='left',on=['image_id','category_id']).fillna(0)
            i_mse = mean_squared_error(df['count_x'], df['count_y'], squared=False)
            print(category_dict[cat_id],' MSE for i= {:.4f}: {:.4f}'.format(i,i_mse))
            mse[cat_id].append(i_mse)
            negative[cat_id].append((df[df['count_x'] - df['count_y']>0].count_x - df[df['count_x'] - df['count_y']>0].count_y).sum())
            positive[cat_id].append((df[df['count_x'] - df['count_y']<0].count_y - df[df['count_x'] - df['count_y']<0].count_x).sum())
        #plt.plot(conf,mse[cat_id],label=category_dict[cat_id])
        #plt.plot(conf,positive[cat_id],label=category_dict[cat_id])
        plt.plot(conf,negative[cat_id],label=category_dict[cat_id])
    plt.yticks(np.arange(0, 6500, 500))
    plt.xticks(np.arange(0, 1, 0.05),rotation=90)
    plt.grid()
    plt.legend()
    plt.xlabel('confidence')
    plt.ylabel('errors')
    plt.show()
    #plt.yscale('log')
    #plt.xticks(np.arange(0, 1, 0.05),rotation=90)
    #plt.yticks(np.arange(0, 12, 2))
    #plt.grid()
    #plt.xlabel('confidence')
    #plt.ylabel('RMSE')
    #plt.legend()
    #plt.show()
    return()

def mse_by_det_num(annotation_json,detection_json,conf):
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
    #cat = pd.DataFrame(ann_data['categories']).rename(columns={'id':'category_id','name':'category'})
    images = pd.DataFrame(ann_data['images']).rename(columns={'id':'image_id'})
    #ann_df = pd.merge(pd.DataFrame(ann_data['annotations']),cat,on=['category_id'],how='inner')
    #ann_df = ann_df[ann_df['category']=='person']
    ann_df = pd.DataFrame(ann_data['annotations'])
    ann_df = pd.merge(images,ann_df,on=['image_id'],how='left')
    ann_df['file_name'] = ann_df['file_name'].apply(lambda x: pl.Path(x).stem)
    ann_df['image_id'] = ann_df['image_id'].astype('category')
    ann_df = ann_df.groupby(['file_name','category_id']).size().reset_index(name='count').rename(columns={'file_name':'image_id'})
    det_df = pd.DataFrame(det_data)
    det_df_c = det_df[det_df['score']>=conf].groupby(['image_id','category_id']).size().reset_index(name='count')
    df = ann_df.merge(det_df_c,how='left',on=['image_id','category_id']).fillna(0)
    mse = {}
    #for cat_id in sorted(df['category_id'].unique()):
    cat_id = 1
    mse[cat_id]={}
    for i in sorted(df[df.category_id==cat_id].count_x.unique()):
        mse[cat_id][i] = mean_squared_error(df[(df.count_x==i) & (df.category_id==cat_id)].count_x, df[(df.count_x==i) & (df.category_id==cat_id)].count_y, squared=False)
    plt.plot(*zip(*(mse[cat_id]).items()),label=category_dict[cat_id])
    #plt.ylim(0,30)
    #plt.xticks(np.arange(0, 1, 0.05),rotation=90)
    #plt.yticks(np.arange(0, 30, 2))
    plt.grid()
    #plt.legend()
    plt.xlabel('# of detection per image')
    plt.ylabel('RMSE')
    plt.show()
    return()




def create_train_file():
    json_path = 'K:/dataset/flir dataset/train/thermal_annotations.json'
    with open(annotation_json) as ann_file:
        ann_data = json.load(ann_file)


if __name__ == '__main__':
    json_path = 'D:/dataset/flir_dataset/train/thermal_annotations.json'
    df = count_people(json_path)
    df.groupby('category').size().plot(kind='bar',rot=45,ylabel='number of annotations')
        #plt.plot(g[g.category==cat]['count'],g[g.category==cat].count_c,label=cat,marker='o',markevery=5)
    axes = plt.gca()
    axes.yaxis.grid()
    plt.show()
    #Group detections by
    plottable = df.groupby(['image_id','category']).size().reset_index(name='count').groupby(['category','count']).size().reset_index(name='count_c').pivot(index='count', columns='category', values='count_c')
    plottable.plot(grid=True,marker='o',markevery=5,xticks=np.arange(0,20,step=3),ylabel='number of annotations',xlabel='annotations per image')
    plt.show()
    #square_mean_loss('K:/dataset/flir_dataset/val/thermal_annotations.json','K:/results/test_flir_evaluation_0005/detection_results.json')

    #mse_by_det_num('K:/dataset/flir_dataset/val/thermal_annotations.json','K:/results/test_kaist_evaluation_0005/detection_results.json',0.3590)