num_classes = 10
import numpy as np
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
dir_path='C:/Users/topchoice/Documents/python_pRa/sar_newBegin/land_more_less/'
train_list,train_label,valid_list,valid_label=load_file_to_list(dir_path,valid_percent)
#    train_batch=batch_train_re(train_list,train_label,batch_size,scale)
#    valid_batch=batch_valid_re(valid_list,valid_label,batch_size*100,scale)
scale=2
image_batch_HR_disp,label_batch_disp ,image_batch_LR_disp=disp_batch_re(train_list,train_label,num_classes,scale)