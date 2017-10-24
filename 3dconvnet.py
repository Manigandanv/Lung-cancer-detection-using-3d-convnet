import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.estimator import regression
import numpy as np
from tqdm import tqdm
from tflearn.optimizers import Adam
train_data=np.load('D:/traindata-50-50-20.npy')
train=train_data[-1397:]
vald=train_data[-1397:]
test_data=np.load('D:/testdata-50-50-20.npy')
X = np.array([i[0] for i in train]).reshape(-1,20,50,50,1)
Y = [i[1] for i in train]
y1=tflearn.data_utils.to_categorical(Y, nb_classes=2)
v1 = np.array([i[0] for i in vald]).reshape(-1,20,50,50,1)
V = [i[1] for i in vald]
v2=tflearn.data_utils.to_categorical(V, nb_classes=2)
num_classes = 2
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_blur(sigma_max=3.0)
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=25.)
network = input_data(shape=[None, 20,50,50,1],data_preprocessing=img_prep,data_augmentation=img_aug,name='input')
network = conv_3d(network, 32, 3,3, activation='relu')
network = max_pool_3d(network, 2,2)
#network = dropout(network, 0.25)
#network = conv_3d(network, 64, 2,2, activation='relu')
network = conv_3d(network, 64, 3,3, activation='relu')
network = max_pool_3d(network, 2,2)
#network = dropout(network, 0.25)
#network = conv_3d(network, 128, 3,3, activation='relu')
#network = conv_3d(network, 128, 2,2, activation='relu')
#network = max_pool_3d(network, 2,2)
#network = dropout(network, 0.25)
#network = conv_3d(network, 256, 3,3, activation='relu')
#network = conv_3d(network, 256, 1,1, activation='relu')
#network = max_pool_3d(network, 2,2)
#network = dropout(network, 0.25)
#network = conv_3d(network, 512, 1,1, activation='relu')
#network = conv_3d(network, 128, 2,2, activation='relu')
#network = max_pool_3d(network, 1,1)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.8)
softmax = fully_connected(network, num_classes, activation='softmax')
adam = Adam(learning_rate=0.0001, beta1=0.98, beta2=0.9999)
regression = regression(softmax, optimizer=adam,loss='categorical_crossentropy',learning_rate=0.001)
model = tflearn.DNN(regression,tensorboard_verbose=3,tensorboard_dir='log')
model.fit(X, y1, n_epoch=1,shuffle=True,validation_set=(v1,v2),show_metric=True, batch_size=1, snapshot_step=500)
#score=model.evaluate(np.array(v1),np.array(v2))
#print('Test score:',score[0],'\nTest loss:',score[1])
#print('Test loss:',score[1])
with open('D:/prob.csv','w') as f:
    f.write('probability\n')         
with open('D:/prob.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        data = img_data.reshape(20,50,50,1)
        model_out = model.predict([data])[0]
        #model_out=model.predict_label([data])[0]
        f.write('{}\n'.format(model_out[1]))
        #print(model.predict([data])[0])