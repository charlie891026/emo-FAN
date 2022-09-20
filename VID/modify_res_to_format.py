import os
import numpy as np
from scipy.special import softmax
import csv

column = ['filename','happy','sad','neutral','angry']
res = []
# read pred_res
pred_res = np.loadtxt(open("res.csv","rb"), delimiter=",", skiprows=0)
# read file name
fn = []
def check_nan_ele_in_np(arr):
    mask = np.isnan(arr)
    if True in mask:
        return True
    else:
        return False
with open('vid_name.txt') as f:
    for i in f.readlines():
        fn.append(i.strip())
# mapping filename and pred_res
for i in range(pred_res.shape[0]):
    print('mapping',i,'th file and res')
    tmp_res=[fn[i]+'.mp4']
    if check_nan_ele_in_np(pred_res[i]):
        print(fn[i],'is nan')
        continue
    print('ori score',pred_res[i])
    pred_score = softmax(pred_res[i])
    print('softmax score',pred_score)
    for s in pred_score:
        tmp_res.append(s)
    print('tmp res:',tmp_res)
    res.append(tmp_res)
print('res',res)
#save result
with open("vid_pred_res_matched_4emo.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(res)