import os
import random
# import scipy.io as sio
import h5py
# import cv2
import numpy as np
# import tensorflow as tf

"""
 file path for save lists and feature
"""
txt_file = '/home/chang/tourismData/featureNew/tourismImageList.txt'
# txt_file = '/deep/tourismDatasetClassification/tourismChosedTextfile/testList2.txt'
feature_path = '/home/chang/tourismData/featureNew/gist/'
save_path = '/home/chang/tourismData/'
save_txt_path = '/home/chang/tourismData/'

"""
 parameters
"""
num_classes = 156
num_train = 50
num_valid = 5


def get_list_and_pack_feat(txt_path):
    files = open(txt_path, 'r')
    feat = []
    coarse_label = []
    fine_label = []

    """
     deal with the txt list
    """
    for file_name in files:
        file_name = file_name.split()

        # get the filename and label of each image
        feat_path = feature_path + file_name[0] + '.mat'
        # print(('The Name: %s, Coarse Label: %d, Fine label: %d')
        #         % (file_name[0], int(file_name[1]), int(file_name[2])))

        # print feat_path
        if os.path.isfile(feat_path):
            # print feat_path
            mat = h5py.File(feat_path, 'r')
            # keys = mat.keys()
            # print keys
            data = mat.get('descrs')
            # data = mat.get('feat')
            data = np.array(data)
            # d = data[0]
            # print d
            # if int(file_name[1]) == 21:
            #     print('The data shape: (%d, %d)' % (data.shape[0], data.shape[1]))
            #     print data
            feat.append(data)
            mat.close()
            coarse_label.append(int(file_name[1]))
            fine_label.append(int(file_name[2]))
            # print len(feat)
            # print len(fine_label)
            # print len(coarse_label)

    # all the feature
    feat = np.array(feat)
    # all the label (scene area)
    coarse_label = np.array(coarse_label)
    # all the label (scene spot)
    fine_label = np.array(fine_label)
    return feat, coarse_label, fine_label


def split_data(feature, num_train, num_valid, class_num):
    # shuffle data
    rand_num = [[i] for i in range(len(feature))]
    random.shuffle(rand_num)
    feature = feature[rand_num, :]
    
    # split train, valid
    train_feat = feature[0:num_train, :, :]
    valid_feat = feature[num_train:num_train + num_valid, :, :]
    test_feat = feature[num_train + num_valid:len(feature), :, :]
    
    train_feat = train_feat.reshape(train_feat.shape[0] * train_feat.shape[1],
                                    train_feat.shape[2], train_feat.shape[3])
    valid_feat = valid_feat.reshape(valid_feat.shape[0] * valid_feat.shape[1],
                                    valid_feat.shape[2], valid_feat.shape[3])
    test_feat = test_feat.reshape(test_feat.shape[0] * test_feat.shape[1],
                                  test_feat.shape[2], test_feat.shape[3])

    return train_feat, valid_feat, test_feat


def save_feat_into_disk(file_handle, feat, class_index, save_path, split_name):
    num_example = feat.shape[0]
    for i in range(num_example):
        file_path = os.path.join(save_path, split_name, str(class_index))
        if os.path.exists(file_path):
            print("Exists!")
        else:
            os.mkdir(file_path)
        # save
        # feat_name = file_path + '/feat_' + str(i) + '.jpg'
        feat_name = file_path + '/feat_' + str(i) + '.npy'
        file_handle.write(feat_name + ' ' + str(class_index) + '\n')
        # cv2.imwrite(feat_name, feat[i,])
        np.save(feat_name, feat[i, :])


def split_train_valid_test(data, cl, fl):
    # build text file saving training, valid and testing list
    train_file = open(save_txt_path + 'trainList2.txt', 'w')
    valid_file = open(save_txt_path + 'validList2.txt', 'w')
    test_file = open(save_txt_path + 'testList2.txt', 'w')
    # reshape data into num_train x input_dimension
    data = np.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])

    # class number
    c_num_classes = np.unique(cl)
    for i in c_num_classes:
        # print i
        class_index = []
        for j in range(len(cl)):
            if cl[j] == i:
                class_index.append(j)
                # print j
        class_data = data[class_index[0]:class_index[-1] + 1] 
        class_cl = cl[class_index[0]:class_index[-1] + 1]
        class_fl = fl[class_index[0]:class_index[-1] + 1]

        # concat each feat into one feature within same coarse class
        f_num_classes = np.unique(class_fl)
        each_class_num = []
        for j in f_num_classes:
            num_example_each_class = list(class_fl).count(j)
            # print(('Class %d, Num %d') % (int(j), int(num_example_each_class)))
            each_class_num.append(num_example_each_class)
        
        # pack feature
        start_index = 0
        end_index = 0
        pack_feat = []
        if min(each_class_num) != max(each_class_num):
            for j in f_num_classes:
                end_index += int(each_class_num[j-1])
                # print start_index, end_index
                temp_data = class_data[start_index:end_index, :]
                temp_data = temp_data[0:min(each_class_num)]
                # print np.array(temp_data).shape
                pack_feat.append(temp_data)
                # np.append(pack_feat, temp_data)
                # print pack_feat
                start_index = end_index
        else:
            for j in f_num_classes:
                end_index += int(each_class_num[j-1])
                # print start_index, end_index
                temp_data = class_data[start_index:end_index, :]
                # print np.array(temp_data).shape
                pack_feat.append(temp_data)
                # np.append(pack_feat, temp_data)
                # print pack_feat
                start_index = end_index
        # pack_feat = np.array(pack_feat)
        print("Num Class: %d" % i)
        print pack_feat[0].shape
        pack_feat = np.transpose(pack_feat, [1, 0, 2])
        train_feat, valid_feat, test_feat = split_data(pack_feat, num_train, num_valid, i)
        
        """
        save feature : num_example x num_fine_classes x feat_dimension
        """
        # save training examples
        save_feat_into_disk(train_file, train_feat, i, save_path, 'train')
        
        # save validation examples
        save_feat_into_disk(valid_file, valid_feat, i, save_path, 'valid')
        
        # save test examples
        save_feat_into_disk(test_file, test_feat, i, save_path, 'test')
        
        # check
        print train_feat.shape, valid_feat.shape, test_feat.shape
        
    # close the text file
    train_file.close()
    valid_file.close()
    test_file.close()
    # f_num_classes = np.unique(fl)
if __name__ == '__main__':

    """
    get data from the file
    """
    feature, c_label, f_label = get_list_and_pack_feat(txt_file)
    
    #    split train, validation, test    #
    split_train_valid_test(feature, c_label, f_label)

    # for file_name in file_names:
    #    return
