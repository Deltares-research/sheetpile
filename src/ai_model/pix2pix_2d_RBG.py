import tensorflow as tf

import time
import datetime
import numpy as np
import plotly.express as px
from PIL import Image

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from IPython import display

OUTPUT_CHANNELS = 4
AMAX = 4.5

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result



def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 4])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 4], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512,
        4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(
        zero_pad1
    )  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1,
                                  4,
                                  strides=1,
                                  kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.reduce_mean(loss_object(tf.ones_like(disc_real_output), disc_real_output))
    generated_loss = tf.reduce_mean(loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    ))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
        tf.summary.scalar("disc_loss", disc_loss, step=step)


def generate_images_slider_pl(model, test_inputs, tars, savefig=True, step=None): 
    prediction = model(test_input, training=True)
    test_input = np.array([slice_t.T for slice_t in test_input.numpy()[0, ..., 0]])
    tar = np.array([slice_t.T for slice_t in tar.numpy()[0, ..., 0]])
    prediction = np.array([slice_t.T for slice_t in prediction.numpy()[0, ..., 0]])
    display_list = np.array([test_input, tar, prediction])
    fig = px.imshow(display_list , animation_frame=1, facet_col=0)
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for counter, annotation in enumerate(fig.layout.annotations):
        annotation.text = title[counter]
    fig.update_layout(sliders=[{"currentvalue": {"prefix": "No of slice in the y direction="}}])
    if savefig:
        fig.write_html(f"gauto\\gans\\pix2pix\\output\\train_pix2pix{step}.html")
    else:
        fig.show()


def generate_images(model, test_input, tar, savefig=True, step=None):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0, :, :, 0].numpy(), tar[0, :, :].numpy(), prediction[0, :, :, 0].numpy()]
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    if savefig:
        plt.savefig(f"output\\train_pix2pix{step}.png")
    else:
        plt.show()
    plt.clf()
    plt.hist(np.array(tar[0, :, :]).flatten(), bins=100, alpha=0.5, label="Ground Truth")
    plt.hist(np.array(prediction[0, :, :, 0]).flatten(), bins=100, alpha=0.5, label="Prediction")
    plt.legend(loc="upper right")
    plt.title(f"Results at step {step}")
    if savefig:
        plt.savefig(f"output\\train_pix2pix_dist{step}.png")
    else:
        plt.show()


def all_images(model, input_dataset):
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
    rc('text', usetex=True)
    dataset = [value for counter, value in enumerate(input_dataset)]
    predictions = [model(data[0], training=True) for data in dataset]
    data_diff = [data[1].numpy()[0,:,:] - predictions[counter].numpy()[0,:,:,0] for counter, data in enumerate(dataset)]
    for counter, image_output in enumerate(predictions):
        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(18.5, 10.5)
        error = np.mean(np.array(np.abs(data_diff[counter])))
        fig.suptitle(f"Absolute error: {error}", fontsize=15)
        display_list = [dataset[counter][0][0].numpy().T * AMAX, 
                        dataset[counter][1][0].numpy().T * AMAX, 
                        predictions[counter].numpy().T[0,:,:] * AMAX,
                        data_diff[counter].T * AMAX]
        title = ["Input Image", "Ground Truth", "Predicted Image", "Absolute difference"]
        for i in range(4):
            ax[i].set_title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            im = ax[i].imshow(display_list[i], vmin=0, vmax=AMAX)
            ax[i].axis("off")
        fig.colorbar(im, ax=ax.ravel().tolist(), orientation="horizontal")
        fig.savefig(f"slice_{counter}_prediction.png")
        plt.close(fig)

def generate_images_slider(model, training_dataset):
    dataset = [value for counter, value in enumerate(training_dataset)]
    prediction = model(dataset[0][0], training=True)
    fig, ax = plt.subplots(1, 3)
    fig.subplots_adjust(bottom=0.35)
    display_list = [dataset[0][0][0].numpy().T, dataset[0][1][0].numpy().T, prediction[0].numpy().T[0,:,:]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for i in range(3):
        ax[i].set_title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        im = ax[i].imshow(display_list[i], vmin=0, vmax=1)
        ax[i].axis("off")

    axslice = plt.axes([0.25, 0.15, 0.65, 0.03])
    freq = Slider(axslice, "Slice number", 0, len(dataset), 0, valstep=1)



    def update(val):
        prediction = model(dataset[freq.val][0], training=True)
        display_list = [dataset[freq.val][0][0].numpy().T, dataset[freq.val][1][0].numpy().T, prediction[0].numpy().T[0,:,:]]
        title = ["Input Image", "Ground Truth", "Predicted Image"]
        for i in range(3):
            ax[i].set_title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            im = ax[i].imshow(display_list[i], vmin=0, vmax=1)
            ax[i].axis("off")

    cb = plt.colorbar(im ,ax = [ax[0]], location = 'left')

    freq.on_changed(update)
    plt.show()


def fit(train_ds, test_ds, validation_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    for step, value_to_unpack in train_ds.repeat().take(steps).enumerate():
        (input_image, target) = value_to_unpack
        if (step) % 1000 == 0:
            display.clear_output(wait=True)
            if step != 0:
                print(f"Time taken for 1000 steps: {time.time() - start:.2f} sec\n")
            start = time.time()
            generate_images(generator, example_input, example_target, step=step)
            if validation_ds is not None:
                one_one_plot_validation(validation_ds, generator)
            print(f"Step: {step // 1000}k")
        train_step(input_image, target, step)
        # Training step
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)
            # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def one_one_plot_validation(validation_dataset, model):
    # validation
    dataset = [value for counter, value in enumerate(validation_dataset)]
    data_diff = [data[-1].numpy() - data[0].numpy() for data in dataset]
    indexes = [np.nonzero(np.any(data[0] != 0, axis=1))[0][0] for data in data_diff]
    predictions = [model(data[0], training=False) for data in dataset]
    plt.clf()
    for counter, ind in enumerate(indexes):
        plt.plot(dataset[counter][-1].numpy()[0, ind, :], predictions[counter][-1].numpy()[ind, :, 0], "o", label=f"Validation set {counter}")
    plt.plot([0,1], [0,1], label="1-1 line")
    plt.xlabel("Expected normalized value")
    plt.ylabel("Predicted normalized value")
    plt.legend()
    plt.savefig("Validation_output.png")



def load_all_data(directory):
    # get all the directories and load the data
    all_directories = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]
    all_data = []
    for directory in all_directories[:30]:
        # get all the png files
        all_files = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isfile(os.path.join(directory,o)) and o.endswith(".png")]
        feature_names = [os.path.basename(file).split(".png")[0] for file in all_files]
        if len(all_files) == 7:
            # load the data
            data = []
            for file in all_files:
                image = Image.open(file)
                # resize the image and maintain aspect ratio
                image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                image_array = np.array(image)
                # extract only the red channel since the image is grayscale
                image_array = image_array[:, :, 0] / IMAGE_IDX
                data.append(image_array)
            all_data.append(np.array(data))
    return feature_names, np.array(all_data).astype(np.float32)



def set_up_and_train_2d_model():

    feature_names, dataset_all = load_all_data("D:/SheetPileGenerator/test/results")
    # find indexes for input
    inputs_lookup = ['FRICTION_ANGLE_material', 'geometry', 'water_pressure_stage_3', 'YOUNG_MODULUS_material']
    inputs_indexes = [feature_names.index(input) for input in inputs_lookup]
    # find indexes for output
    outputs_lookup = ['total_displacement_stage_3']
    outputs_indexes = [feature_names.index(output) for output in outputs_lookup]
    inputs_dataset = dataset_all[:, inputs_indexes, :, :]
    outputs_dataset = dataset_all[:, outputs_indexes, :, :]
    # reshape the data
    inputs_dataset = np.reshape(inputs_dataset, (inputs_dataset.shape[0], inputs_dataset.shape[2], inputs_dataset.shape[3], inputs_dataset.shape[1] ))
    outputs_dataset = np.reshape(outputs_dataset, (outputs_dataset.shape[0], outputs_dataset.shape[2], outputs_dataset.shape[3]))
    # split the data into train and test
    percentage_train = 0.8
    train_input_dataset = inputs_dataset[:int(percentage_train * len(inputs_dataset))]
    train_output_dataset = outputs_dataset[:int(percentage_train * len(outputs_dataset))]
    test_input_dataset = inputs_dataset[int(percentage_train * len(inputs_dataset)):]
    test_output_dataset = outputs_dataset[int(percentage_train * len(outputs_dataset)):]

    input_dataset = tf.convert_to_tensor(tf.constant(train_input_dataset))
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset)
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)
    target_dataset = tf.convert_to_tensor(tf.constant(train_output_dataset))
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))

    input_dataset_test = tf.convert_to_tensor(tf.constant(test_input_dataset))
    test_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset_test)
    test_input_dataset = test_input_dataset.batch(BATCH_SIZE)
    target_dataset_test = tf.convert_to_tensor(tf.constant(test_output_dataset))
    test_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset_test)
    test_target_dataset = test_target_dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))

    #validation_dataset = load_validation_data("data\cond_rf\\validation_final")
    #input_dataset_validation = tf.convert_to_tensor(tf.constant(validation_dataset[0]))
    #validation_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset_validation)
    #validation_input_dataset = validation_input_dataset.batch(BATCH_SIZE)
    #target_dataset_validation = tf.convert_to_tensor(tf.constant(validation_dataset[1]))
    #validation_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset_validation)
    #validation_target_dataset = validation_target_dataset.batch(BATCH_SIZE)
    #validation_dataset = tf.data.Dataset.zip((validation_input_dataset, validation_target_dataset))

    fit(train_dataset, test_dataset, None,  steps=50000)

    # TODO plot against all the training inputs
    all_images(generator, test_dataset) 
    # TODO plot diff 
    dataset = [value for counter, value in enumerate(test_dataset)]
    predictions = [generator(data[0], training=True) for data in dataset]
    data_diff = [data[1].numpy() - predictions[counter].numpy() for counter, data in enumerate(dataset)]

    mean_axis_error = np.mean(np.array(np.abs(data_diff)), axis=0)
    plt.clf()
    plt.imshow(mean_axis_error)
    plt.colorbar
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(mean_axis_error.T)
    ax.set_xlabel("Mean error per pixel at the end of training")
    fig.colorbar(im, orientation="horizontal")
    plt.show()

    # TODO calculate all the diffs
    mean_error = np.mean(np.abs(np.array(data_diff)).flatten())
    std_error = np.std(np.array(data_diff).flatten())
    print(f"stats mean abserror {mean_error} std : {std_error}")


if __name__ == "__main__":
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    import os
    AMAX = 4.5
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    OUTPUT_CHANNELS = 1
    LAMBDA = 100
    IMAGE_SIZE = 256
    IMAGE_IDX = 255

    # define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    log_dir = "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_train/keep" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    loss_object = tf.keras.losses.MeanSquaredError()
    generator = Generator()
    discriminator = Discriminator()

    checkpoint_dir = "./training_checkpoints_2d"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    set_up_and_train_2d_model()
