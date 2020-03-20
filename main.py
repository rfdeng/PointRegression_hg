import tensorflow as tf
from tensorflow import keras
import os
import imageio
import glob
import numpy as np
import tensorflow.keras.backend as K

for module in tf, keras:
    print(module.__name__, module.__version__)

class data_generator:
    def __init__(self, datadir, batch_size):
        self._imglist = glob.glob(os.path.join(datadir, '*.jpg'))
        self._batch_size = batch_size
        self._len = len(self._imglist)
        self._batch_id = 0
        self._batch_num = int(self._len / self._batch_size)

    def generator(self):
        while True:
            if self._batch_id > self._batch_num - 1:
                self._batch_id = 0
                np.random.shuffle(self._imglist)

            img_batch = np.array([imageio.imread(self._imglist[i + self._batch_id * self._batch_size])
                                  for i in range(self._batch_size)])
            gt_batch = np.array([np.load(self._imglist[i + self._batch_id * self._batch_size].replace('.jpg', '.npy'))
                                 for i in range(self._batch_size)])
            self._batch_id += 1
            yield (img_batch, (gt_batch, gt_batch))
    
    @property
    def get_size(self):
        return self._len
    
    @property
    def get_batchNum(self):
        return self._batch_num

class Euldis_callback(keras.callbacks.Callback):
    def __init__(self, imgs, gts):
        self.imgs = imgs
        self.gts = gts

    def on_epoch_end(self, epoch, logs={}):
        _, preds = self.model.predict(self.imgs.astype(np.float64))
        se = np.square(preds - self.gts)
        se[self.gts == 0] = se[self.gts == 0] * 0.2
        mse = np.mean(se)
        print('\n\033[1;31;40mthe mse of validation dataset in the {} th epoch is {}.\033[0m'.format(epoch, mse))

        Euldis = []
        for pred, gt in zip(preds, self.gts):
            for i in range(pred.shape[-1]):
                p = pred[:, :, i]
                g = gt[:, :, i]
                max_cor_p = np.where(p == np.max(p))
                max_cor_g = np.where(g == np.max(g))
                tmpp = tuple(map(np.mean, max_cor_p)) # in case the maximum location is not unique
                tmpg = tuple(map(np.mean, max_cor_g))
                Euldis.append(np.sqrt(np.sum(np.square(np.array(tmpp) - np.array(tmpg)))))
        cornerError = np.mean(Euldis)
        print('the corner error of the {} th epoch is {}.'.format(epoch, cornerError))
        return

class PointRegression:
    def __init__(self, nstacks, nclasses):
        self._nstacks = nstacks  # not used yet.
        self._nclasses = nclasses
        
    def resblock(self, bottom, num_out_channels, block_name):
        # skip layer
        if K.int_shape(bottom)[-1] == num_out_channels:
            _skip = bottom
        else:
            _skip = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                           name=block_name + 'skip')(bottom)
            
        # residual block
        x = keras.layers.BatchNormalization()(bottom)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(num_out_channels//2, kernel_size=(1, 1), padding='same', 
                                name=block_name + 'conv_1x1_x1')(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(num_out_channels//2, kernel_size=(3, 3), padding='same',
                                name=block_name + 'conv_3x3_x2')(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), padding='same',
                                name=block_name + 'conv_3x3_x3')(x)

        output = keras.layers.Add()([_skip, x])
        
        return output
    
    
    def hgblock(self, bottom, num_out_channels, block_name):
        side1 = self.resblock(bottom, num_out_channels, block_name + 'side1_')
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(bottom)
        x = self.resblock(x, num_out_channels, block_name + 'lres1_')
        
        side2 = self.resblock(x, num_out_channels, block_name + 'side2_')
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = self.resblock(x, num_out_channels, block_name + 'lres2_')
        
        side3 = self.resblock(x, num_out_channels, block_name + 'side3_')
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = self.resblock(x, num_out_channels, block_name + 'lres3_')
        
        side4 = self.resblock(x, num_out_channels, block_name + 'side4_')
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        for i in range(3):
            x = self.resblock(x, num_out_channels, block_name + 'mres' + str(i+1) + '_')
        x = keras.layers.UpSampling2D()(x)
        
        x = keras.layers.Add()([side4, x])
        x = self.resblock(x, num_out_channels, block_name + 'rres3_')
        x = keras.layers.UpSampling2D()(x)
        
        x = keras.layers.Add()([side3, x])
        x = self.resblock(x, num_out_channels, block_name + 'rres2_')
        x = keras.layers.UpSampling2D()(x)
        
        x = keras.layers.Add()([side2, x])
        x = self.resblock(x, num_out_channels, block_name + 'rres1_')
        x = keras.layers.UpSampling2D()(x)
        
        output = keras.layers.Add()([side1, x])
        
        return output
        
        
    def hghead(self, skipout, hgout, num_out_channels, block_name):
        x = self.resblock(hgout, num_out_channels, block_name + 'res_')
        x = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), padding='same',
                                name=block_name + 'conv1_')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        
        # for supervision
        pred = keras.layers.Conv2D(self._nclasses, kernel_size=(1, 1), padding='same',
                                   name=block_name + 'pred_')(x)
        
        # generate output for next stage in the network
        pred_back_x = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), padding='same',
                                          name=block_name + 'predback1_')(pred)
        x = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), padding='same',
                                name=block_name + 'conv2_')(x)
        output = keras.layers.Add()([pred_back_x, x, skipout])        
        
        return output, pred    

    @staticmethod
    def decayed_mse(y_true, y_pred):
        se = tf.square(y_pred - y_true)
        dacayed_se = tf.where(y_true==0, 0.2*se, se)
        loss = tf.reduce_mean(dacayed_se)
        return loss


    def build(self, input):
        # before hourglass
        x = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                                name='front_conv_7x7_x1')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = self.resblock(x, 128, 'front_res_x1_')
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = self.resblock(x, 128, 'front_res_x2_')
        x = self.resblock(x, 256, 'front_res_x3_')
        
        # hourglass module
        outputs = []
        for i in range(1, self._nstacks + 1):
            hgbout = self.hgblock(x, 256, 'hg' + str(i) + '_')
            x, pred = self.hghead(x, hgbout, 256, 'hg' + str(i) + '_head_')
            outputs.append(pred)

        model = keras.models.Model(inputs=input, outputs=outputs)
        
        return model

# model preparation.
input_resolution = [320, 320, 3]        
PR = PointRegression(2, 4)
model = PR.build(keras.layers.Input(shape=input_resolution))
#print(model.summary())

optimizer = keras.optimizers.RMSprop(lr=2.5e-4)

model.compile(optimizer=optimizer, loss=PR.decayed_mse)

# data preparation.
train_data = data_generator('/data/train80', 5)
valid_data = data_generator('/data/val80', 1)
train_data_generator = train_data.generator()

vallist = glob.glob('/data/val80/*.npy')
valgt = np.array([np.load(ele) for ele in vallist])
valimg = np.array([imageio.imread(ele.replace('.npy', '.jpg')) for ele in vallist])
valdataset = tf.data.Dataset.from_tensor_slices((valimg, (valgt, valgt)))

# callbacks definition.
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'hourglass_model.h5')

callbacks = [ keras.callbacks.TensorBoard(logdir),
              keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
              keras.callbacks.EarlyStopping(monitor='hg2_head_pred__loss', patience=15),
              Euldis_callback(valimg, valgt)
            ]

# training.
model.fit(train_data_generator, epochs=80, steps_per_epoch=train_data.get_batchNum,
          validation_data=valdataset.batch(1), validation_steps=valid_data.get_batchNum,
          callbacks=callbacks)
