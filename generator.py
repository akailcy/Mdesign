
def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()

    # cnn.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
    # cnn.add(Reshape((7, 7, 128)))
    # cnn.add(BatchNormalization(momentum=0.8))
    # cnn.add(UpSampling2D())
    cnn.add(Conv2D(64, kernel_size=3, padding="same"))
    cnn.add(Activation("relu"))
    cnn.add(Conv2D(128, kernel_size=3, padding="same"))
    cnn.add(Activation("relu"))
    cnn.add(BatchNormalization(momentum=0.8))
    #cnn.add(UpSampling2D())
    cnn.add(Conv2D(64, kernel_size=3, padding="same"))
    cnn.add(Activation("relu"))
    #cnn.add(BatchNormalization(momentum=0.8))
    cnn.add(Conv2D(self.channels, kernel_size=3, padding='same'))
    cnn.add(Activation("tanh"))

    cnn.summary()
    #卧槽，怎么把分类标签信息导入进去
    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,latent_size,1 ))#输入的低分辨率图像

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Reshape((latent_size,latent_size))(Embedding(num_classes, latent_size*latent_size,
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

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
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
