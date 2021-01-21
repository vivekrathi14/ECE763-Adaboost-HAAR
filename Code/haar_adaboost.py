# -*- coding: utf-8 -*-
"""
@author: Vivek Rathi
"""
#import packages
import numpy as np
import cv2
import glob
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from matplotlib import pyplot as plt
from skimage.feature import haar_like_feature_coord
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import random

# Data path
face_path = '../face/Face16'
n_face_path = '../non_face/Nonface16'

# create list of images (unflattened)
def create_list_images(path):
    list_images = []
    for imagepath in tqdm(glob.glob(path + '\*')):
        image = cv2.imread(imagepath,0)
        image = image/255
        list_images.append(image.astype(np.float32))
        if len(list_images) == 1200:
            break
        random.shuffle(list_images)
    return list_images

# get haar features
def haar_feature(img,feature_type_):
    int_img = integral_image(img)
    return haar_like_feature(int_img, 0, 0, int_img.shape[0],int_img.shape[1],feature_type_)

# get haar feature list
def h_feature_list(dataset,f_type):
    h_f_list = [haar_feature(data,f_type) for data in tqdm(dataset)]
    return np.asarray(h_f_list)

# get an idea of kernels & haar features used by default code
def visualize_haar_f_kernel(img,f_type):
    coord,_ = haar_like_feature_coord(img.shape[0], img.shape[1],f_type)
    coord_array = np.asarray(coord)
    # pt(x,y) image(y,x)
    point_coord = [[(c2[1],c2[0]) for c1 in c for c2 in c1] for c in coord_array]
    for p in point_coord:
        img_v = np.ones((img.shape[0],img.shape[1])).astype('uint8')
        img_v = img_v* 127
        img_v = cv2.rectangle(img_v,p[0],p[1],255,-1)
        img_v = cv2.rectangle(img_v,p[2],p[3],0,-1)
        cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
        cv2.imshow('Image',img_v)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        

# visualize haar features on an image 
def visualize_haar_f(img,c):
    p = [(c2[1],c2[0]) for c1 in c for c2 in c1]
    img_c = np.copy(img)
    j = 0 # index
    for i in range(len(p)//2):
        if i%2 == 0:
            img_c = cv2.rectangle(img_c,p[j],p[j+1],1,-1)
        else:
            img_c = cv2.rectangle(img_c,p[j],p[j+1],0,-1)
        j+=2
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image',img_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# get all haar parameters    
def h_f_param(data_f,data_nf,f_type):
    hf_data_f = h_feature_list(data_f,f_type) # haar feature for face images
    hf_data_nf = h_feature_list(data_nf,f_type) # haar feature for non face imag|----Haar Features-----|
#         input vector visualization
#           ---Haar Features---
#          I  [ h1 h2 h3 h4 ] --> x1
#          M  [ h1 h2 h3 h4 ] --> x2
#          G  [ h1 h2 h3 h4 ] --> x3

    input_ada_xi = np.vstack((hf_data_f,hf_data_nf))
    
    # get threshold using mean logic
    mean_hf_f = hf_data_f.mean(axis=0) # sum along vertcially --> [h1 h2 h3 h4]
    mean_hf_nf = hf_data_nf.mean(axis=0)
    thres = (mean_hf_f + mean_hf_nf)/2 # get mean threshold vector --> [h1m h2m h3m h4m]
    
#    thres = get_threshold(hf_data_f[100:200],hf_data_nf[100:200],N)
    return thres, input_ada_xi

# get threshold for comparision using linear search
def get_threshold(hf_data,hnf_data,N):
    input_xi = np.vstack((hf_data,hnf_data))
    e_list = []
    for x in tqdm(input_xi):
        bool_t = input_xi >= x
        a_t = np.vstack((np.where(bool_t[:N,:]==False,1,0),\
                               np.where(~bool_t[N:,:]==False,1,0)))
        e_list.append(a_t.sum(axis=0))
    e_a = np.asarray(e_list)
    
    max_acc_rows = np.argsort(e_a,axis=0)[-1]  # argsort gives index of sorted array
    thres = [input_xi[row][i] for i,row in tqdm(enumerate(max_acc_rows))]
    
    return np.asarray(thres)

# Adaboost algorithm
def adaboost(input_xi,threshold,N):
    # create output for N images
    y_face_data = np.ones(N) # row vetcor --> [1 1 1 1]
    y_n_face_data = np.ones(N)
    # create output array                                 #                 face       non-face
    output_yi = np.hstack((y_face_data,y_n_face_data*-1)) # row vercor --> [1 1 1 ... -1 -1 -1 ...]
    
    #adaboost
    maxT = 100 # maximum iterations (change it to 10 for top 10 haar features)
    num_images = 2*N
    #these depends on no. of your haar feature
    weights = np.ones(num_images) / num_images # row vector
    
    # epsilon_t, hxinoty, hxi
    error_t = np.zeros(input_xi.shape[1]) # row vector --> [eh1 eh2 eh3 eh4] (size of haar feature vector for an image)
    bool_t = input_xi >= threshold # matrix --> no. of images x no. of haar features
    
    # get incorrect samples
    h_x_not_y = np.vstack((np.where(bool_t[:N,:]==False,1,0*bool_t[:N,:]),\
                           np.where(~bool_t[N:,:]==False,1,0*~bool_t[N:,:])))
    
    # get correct samples
    h_x = np.vstack((np.where(bool_t[:N,:]==False,-1,1*bool_t[:N,:]),\
                           np.where(~bool_t[N:,:]==False,1,-1*~bool_t[N:,:])))
    hf_i = [] # haar feature array
    hf_a = []
    alpha = []
    z_t = 0
    #iterate
    for t in range(maxT):
        error_t = np.dot(weights,h_x_not_y) # calculate error
        ht = np.argsort(error_t)[0] # get haarfeature index with least error (trick to get top 10b haar features)
        print("Iteration {} - {}".format(t+1,ht))
        hf_i.append(ht)
        et_ht = error_t[ht]
        alpha_t = 0.5 * np.log((1-et_ht)/(et_ht))
        alpha.append(alpha_t)
        ht_x = h_x[:,ht] #weak classifier
        power = np.multiply(output_yi,ht_x) * -1 * alpha_t
        z_t = np.multiply(weights,np.exp(power)).sum()
        weights = np.multiply(weights,np.exp(power)) / z_t
        hf_a.append(input_xi[:,ht])
        Fx = np.dot(alpha,hf_a)
        Hx = np.sign(Fx) # signum function (positive and negative)
        neg = (np.sum(Hx != output_yi))
        print("error -{}, alpha-{}, zt-{}".format(neg,alpha_t,z_t))
        
        # negatives less than 100 i.e. 10% stop training
        if neg <= 100:
            break
    return hf_i,alpha

# test the classifier obtained using adaboost
def test_adaboost(data,index,alpha,feature_type):
    N = len(data)//2
    y_face_data = np.ones(N)
    y_n_face_data = np.ones(N)
    yi = np.hstack((y_face_data,y_n_face_data*-1))
    a_hf_ = get_adaboost_haar_f(index,data,feature_type)
    Fx = np.dot(alpha,a_hf_)
    Hx = np.sign(Fx)
    error = np.sum(Hx != yi)
    TP = np.sum(Hx[:N] == yi[:N] )
    FP = np.sum(Hx[:N] != yi[:N])
    TN = np.sum(Hx[N:] == yi[N:] )
    FN = np.sum(Hx[N:] != yi[N:] )
    
    return error,TP,FP,TN,FN,Hx

# visualize adaboost haar features
def adaboost_haar_vis(face_data,hfs,f_type):
    coord,_ = haar_like_feature_coord(face_data[0].shape[0], face_data[0].shape[1],f_type)
    for i in range(10):
        visualize_haar_f(face_data[i],coord[hfs[i]])

# get adaboost given best haar features
def get_adaboost_haar(img,index_,f_type,f_coord):
    int_img = integral_image(np.asarray(img))
    return haar_like_feature(int_img, 0, 0, int_img.shape[0],int_img.shape[1],\
                      f_type[index_:index_+1],f_coord[index_:index_+1])
    
# get adaboost given best haar features list    
def get_adaboost_haar_f(index,dataset,feature_type):
    f_coord,f_type = haar_like_feature_coord(dataset[0].shape[0],dataset[0].shape[1],feature_type)
    H_list = [np.asarray([get_adaboost_haar(data,ind,f_type,f_coord)[0] for data in dataset])\
              for ind in tqdm(index)]
    return np.asarray(H_list)

# ROC plot
def ROC_plot(predictions,length):    
    actual = np.append([1]*length,[0]*length)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.plot(false_positive_rate, true_positive_rate,label= "AUC " + str(roc_auc))
    plt.legend(loc='upper right')
    plt.show()
    


# get data list
face_list = create_list_images(face_path)
n_face_list = create_list_images(n_face_path)

#create dataset - training & test
face_data_train = face_list[:1000]
n_face_data_train = n_face_list[:1000]
test_data = face_list[1000:] + n_face_list[1000:]

#feature type for haar features
#    feature_type = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
feature_type = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y']

# get threshold through linear search
thres, input_ada_xi = h_f_param(face_data_train,n_face_data_train,feature_type)

# get len of data
N = len(face_data_train)

# gets haar features & alphas from adaboost
hfs,alphas = adaboost(input_ada_xi,thres,N)

# visualise haar features from adaboost
adaboost_haar_vis(face_data_train,hfs,feature_type)

# get confusion matrix & predictions for top 10 adaboost only
error,TP,FP,TN,FN,pred = test_adaboost(test_data,hfs[:10],alphas[:10],feature_type)

# plot ROC
ROC_plot(pred,len(test_data)//2)

print("True Positives: {}/200,".format(TP))
print("False Positives: {}/200,".format(FP))
print("True Negatives: {}/200,".format(TN))
print("False Negatives: {}/200,".format(FN))
print("Misclassification Error Rate: {}".format(error*100/len(test_data)))
    
