# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating,
as the compilation time can be a blocker using Theano.
Timings:
Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min
Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
#from PIL import Image

from six.moves import range
import matplotlib.pyplot as plt
#from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
#from input_data import *
import numpy as np

np.random.seed(1337)
num_classes = 10
#import numpy as np
import os  
import cv2
image_row=100
image_col=100
valid_percent=0.2

def load_file_to_list(dir_path,valid_percent):
    train_ls=[]
    label_ls=[]
    
    files= os.listdir(dir_path) 
    for file_label,file in enumerate(files):
        if  os.path.isdir(dir_path+file):
            loc_path = dir_path+file+'/'
            imas = os.listdir(dir_path+file)
            for img in imas:
                if (not os.path.isdir(img)) and (img!='Thumbs.db'):
                    train_ls.append(loc_path+img)
                    label_ls.append(file_label)
    temp = np.array([train_ls, label_ls])
    temp = temp.transpose()
    np.random.shuffle(temp)
    #label_ls=np.zeros((len(label_ls),len(files)),dtype="float32")
    label_ls=np.zeros((len(label_ls),),dtype="float32")
    #从打乱的temp中再取出list（img和lab）
    #给label手动one-hot
    #image_list = list(temp[:, 0])
    #label_list = list(temp[:, 1])
    #for index,i in enumerate(label_list):        
        #label_ls[index,int(i)]=1

    #valid_list = image_list[:int(len(image_list)*valid_percent)] 
    #valid_label = label_ls[:int(len(image_list)*valid_percent),:]
    #train_list = image_list[int(len(image_list)*valid_percent):]
    #train_label = label_ls[int(len(image_list)*valid_percent):,:]
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    for index,i in enumerate(label_list):        
        label_ls[index]=int(i)


    valid_list = image_list[:int(len(image_list)*valid_percent)] 
    valid_label = label_ls[:int(len(image_list)*valid_percent),]
    train_list = image_list[int(len(image_list)*valid_percent):]
    train_label = label_ls[int(len(image_list)*valid_percent):,]
    return train_list,train_label,valid_list,valid_label

def process_line(data_list,data_label,i,scale):
    img_x = cv2.imread(data_list[i])
    img_y = data_label[i,]
    #temp=tf.image.resize_images(img_x, size=[int(image_row/scale),int(image_col/scale)], method=2 )
    #img_z = tf.image.resize_images(tf.image.resize_images(img_x, size=[int(image_row/scale),int(image_col/scale)], method=2 )
                              # ,size=[image_row,image_col], method=2)
    temp = cv2.resize(img_x,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_CUBIC)
    img_z= cv2.resize(temp,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    x = np.array(img_x,dtype="float32")
    y = np.array(img_y,dtype="float32")
    z = np.array(img_z,dtype="float32")

    
    return (x[:,:,0]- 127.5) / 127.5,y,(z[:,:,0]- 127.5) / 127.5

def batch_train_re(train_list,train_label,batch_size,scale):
    X = np.empty((batch_size,image_row,image_col,1),dtype="float32")
    Y = np.empty((batch_size,),dtype="float32")
    Z = np.empty((batch_size,image_row,image_col,1),dtype="float32")
    cnt=0
    i=0
    #for i in range(len(train_list)):
    while True:
        X[cnt,:,:,0],Y[cnt,],Z[cnt,:,:,0]=process_line(train_list,train_label,i,scale)
        cnt+=1
        if cnt%batch_size==0:
            yield X,Y,Z
            cnt=0
            X = np.empty((batch_size,image_row,image_col,1),dtype="float32")
            Y = np.empty((batch_size,),dtype="float32")
            Z = np.empty((batch_size,image_row,image_col,1),dtype="float32")
        i += 1
        if i>=(len(train_list)):
            i =0
def batch_valid_re(valid_list,valid_label,batch_size,scale):
    X = np.empty((batch_size,image_row,image_col,1),dtype="float32")
    Y = np.empty((batch_size,),dtype="float32")
    Z = np.empty((batch_size,image_row,image_col,1),dtype="float32")
    cnt=0
    i=0
    #for i in range(len(train_list)):
    while True:
        X[cnt,:,:,0],Y[cnt,],Z[cnt,:,:,0]=process_line(valid_list,valid_label,i,scale)
        cnt+=1
        if cnt%batch_size==0:
            yield X,Y,Z
            cnt=0
            X = np.empty((batch_size,image_row,image_col,1),dtype="float32")
            Y = np.empty((batch_size,),dtype="float32") 
            Z = np.empty((batch_size,image_row,image_col,1),dtype="float32")
        i += 1
        if i>=(len(valid_list)):
            i =0
def disp_batch_re(train_list,train_label,num_classes,scale):
    X = np.empty((num_classes ,image_row,image_col,1),dtype="float32")
    Y = np.empty((num_classes ,),dtype="float32")
    Z = np.empty((num_classes ,image_row,image_col,1),dtype="float32")   
    for i in range(num_classes):
        indices = np.argsort(train_label, axis=0)
        ind = np.random.randint(0, 100)
        k=int(indices[int(i*(len(train_label)/num_classes-1)+ind)])
        X[i,:,:,0],Y[i,],Z[i,:,:,0]=process_line(train_list,train_label,k,scale)
    return X,Y,Z


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)

    cnn = Sequential()

    cnn.add(Conv2D(16, kernel_size=3, padding="same",input_shape=(latent_size,latent_size,1)))
    cnn.add(Activation("relu"))
    cnn.add(Conv2D(32, kernel_size=3, padding="same"))
    cnn.add(Activation("relu"))
    cnn.add(BatchNormalization(momentum=0.8))
    cnn.add(Conv2D(16, kernel_size=3, padding="same"))
    cnn.add(Activation("relu"))
    cnn.add(Conv2D(1, kernel_size=3, padding='same'))
    cnn.add(Activation("tanh"))

    cnn.summary()
    #卧槽，怎么把分类标签信息导入进去
    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,latent_size,1 ))#输入的低分辨率图像

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Reshape((latent_size,latent_size,1))(Embedding(num_classes, latent_size*latent_size,
                              embeddings_initializer='glorot_normal')(image_class))
    #生成100维的数据，嵌入层
    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)



def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(100,100, 1)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(32, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(16, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(100, 100, 1))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    epochs =50
    #batch_size = 100
    latent_size = 100
    scale =2
    batch_size=32
    dir_path='C:/Users/topchoice/Documents/python_pRa/sar_newBegin/land_more_less/'
    train_list,train_label,valid_list,valid_label=load_file_to_list(dir_path,valid_percent)
    train_batch=batch_train_re(train_list,train_label,batch_size,scale)
    valid_batch=batch_valid_re(valid_list,valid_label,batch_size*100,scale)
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, latent_size,1))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)#输出的fake是D输出的是否为真实图片的概率，aux是D输出的分类向量
    combined = Model([latent, image_class], [fake, aux])
                      #gan_in                  gan_out 
    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]


    num_train, num_test = len(train_list), len(valid_list)

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(len(train_list) / batch_size)
        progress_bar = Progbar(target=num_batches)

        # we don't want the discriminator to also maximize the classification
        # accuracy of the auxiliary classifier on generated images, so we
        # don't train discriminator to produce class labels for generated
        # images (see https://openreview.net/forum?id=rJXTf9Bxg).
        # To preserve sum of sample weights for the auxiliary classifier,
        # we assign sample weight of 2 to the real images.
        disc_sample_weight = [np.ones(2 * batch_size),
                              np.concatenate((np.ones(batch_size) * 2,
                                              np.zeros(batch_size)))]

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # generate a new batch of noise
            #noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch_HR,label_batch ,image_batch_LR = next(train_batch)#修改输入数据，
#            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
#            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
#
#            # sample some labels from p_c
#            sampled_labels = np.random.randint(0, num_classes, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [image_batch_LR,label_batch.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch_HR, generated_images))
            # 可以合并后随机一下
            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)
            aux_y = np.concatenate((label_batch, label_batch), axis=0)

            # see if the discriminator can figure itself out...
            #数据输入的入口，不用怎么改
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
#            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
#            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)
            image_batch_HR_2,label_batch_2 ,image_batch_LR_2 = next(train_batch)
            x_D=np.concatenate((image_batch_LR,image_batch_LR_2))
            sampled_labels_D=np.concatenate((label_batch,label_batch_2))
            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size) * soft_one

            epoch_gen_loss.append(combined.train_on_batch(
                [x_D, sampled_labels_D.reshape((-1, 1))],
                [trick, sampled_labels_D]))#主要修改的方向
            #combined = Model([latent, image_class], [fake, aux])
            progress_bar.update(index + 1)

        print('Testing for epoch {}:'.format(epoch))

        # evaluate the testing loss here

        # generate a new batch of noise
        #noise = np.random.uniform(-1, 1, (num_test, latent_size))
        image_batch_HR,label_batch ,image_batch_LR = next(valid_batch)
        # sample some labels from p_c and generate images from them to test
        #sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict(
            [image_batch_LR, label_batch.reshape((-1, 1))], verbose=False)

        x = np.concatenate((image_batch_HR, generated_images))
        y = np.array([1] * batch_size*100  + [0] * batch_size*100 )
        aux_y = np.concatenate((label_batch, label_batch), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
#        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
#        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)
        image_batch_HR_2,label_batch_2 ,image_batch_LR_2 = next(valid_batch)
        x_G=np.concatenate((image_batch_LR,image_batch_LR_2))
        sampled_labels_G=np.concatenate((label_batch,label_batch_2))
        
        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [x_G, sampled_labels_G.reshape((-1, 1))],
            [trick, sampled_labels_G], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'ac_gan_diff_chg_weights/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'ac_gan_diff_chg_weights/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)
        r, c = 2, 10
        #noise = np.random.normal(0, 1, (r * c, 100))
        #sampled_labels = np.arange(0, 10).reshape(-1, 1)
        image_batch_HR_disp,label_batch_disp ,image_batch_LR_disp=disp_batch_re(train_list,train_label,num_classes,scale)
        gen_imgs = generator.predict([image_batch_LR_disp, label_batch_disp])

    # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        image_batch_HR_disp=0.5*image_batch_HR_disp+0.5
        disp_imgs=np.concatenate((gen_imgs ,image_batch_HR_disp))
        disp_labels=np.concatenate((label_batch_disp ,label_batch_disp))
        fig, axs = plt.subplots(r, c)
        fig.suptitle("SRACGAN: Generated image", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(disp_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Cls: %d" % label_batch_disp[cnt])
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig("./images/SR_%d.png" % epoch)
        plt.close()

    with open('acgan-history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)