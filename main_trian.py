train_list,train_label,valid_list,valid_label=load_file_to_list(dir_path,valid_percent)
train_batch=batch_train_re(train_list,train_label,batch_size)
valid_batch=batch_valid_re(valid_list,valid_label,batch_size*10)
if __name__ == '__main__':

    # batch and latent size taken from the paper
    epochs = 100
    batch_size = 64
    latent_size = 100

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

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    num_train, num_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
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
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            # image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            # label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            image_batch_HR,label_batch ,image_batch_LR = next(train_batch)#修改输入数据，
            # sample some labels from p_c
            #sampled_labels = np.random.randint(0, num_classes, batch_size)
			#看看标签维度对上没
            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [image_batch_LR, label_batch.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch, generated_images))
			#把输入数据和生成器生成的图片合并
            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)
            aux_y = np.concatenate((label_batch, label_batch), axis=0)
			#y表示图片是否为真实图像，aux_y为图像分类标签
            # see if the discriminator can figure itself out...
            #数据输入的入口，不用怎么改
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
			image_batch_HR_2,label_batch_2 ,image_batch_LR_2 = next(train_batch)
            ##noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
			x_D=np.concatenate(image_batch_LR,image_batch_LR_2)
			sampled_labels_D=np.concatenate(label_batch,label_batch_2)
			#妈的，这个noise怎么搞 继续next（train）？
            ##sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)
			#图像标签
            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size) * soft_one
			#是否是高分辨率原始图像
            epoch_gen_loss.append(combined.train_on_batch(
                [x_D, sampled_labels_D.reshape((-1, 1))],
                [trick, sampled_labels_D]))

            progress_bar.update(index + 1)

        print('Testing for epoch {}:'.format(epoch))

        # evaluate the testing loss here

        # generate a new batch of noise
        ##noise = np.random.uniform(-1, 1, (num_test, latent_size))
		image_batch_HR,label_batch ,image_batch_LR = next(valid_batch)
        # sample some labels from p_c and generate images from them
        ##sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict(
            [image_batch_LR, label_batch.reshape((-1, 1))], verbose=False)

        x = np.concatenate((image_batch_HR, generated_images))
        y = np.array([1] * batch_size*10 + [0] * batch_size*10)
        aux_y = np.concatenate((label_batch, label_batch), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)