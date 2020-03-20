import tensorflow as tf
import numpy as np
import cv2
import glob
import sys
import os
import matplotlib.pyplot as plt

def decayed_mse(y_true, y_pred):
    se = tf.square(y_pred - y_true)
    dacayed_se = tf.where(y_true==0, 0.2*se, se)
    loss = tf.reduce_mean(dacayed_se)
    return loss

if __name__ == '__main__':
    # data preparation
    images = glob.glob(os.path.join(sys.argv[1], '*.jpg'))
    dataset = np.array([cv2.resize(cv2.imread(img)[:,:,::-1], (320, 320)) for img in images]).astype(np.float64)
    outdir = 'show'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # model preparation
    model = tf.keras.models.load_model('tmpbest.h5', custom_objects={'decayed_mse': decayed_mse})

    # prediction
    _, preds = model.predict(dataset, verbose=1)

    # postprocess
    for img, pred in zip(images, preds):
        im = cv2.imread(img)
        h, w, _ = im.shape
        colors = np.linspace(0, 1, pred.shape[-1])
        for i in range(pred.shape[-1]):
            p = pred[:, :, i]
            pts_cor = np.mean(np.where(p == np.max(p)), axis=1)
            li = int(pts_cor[0] / 80.0 * h + 0.5)
            co = int(pts_cor[1] / 80.0 * w + 0.5)
            color = tuple(map(lambda x: x*255, plt.cm.rainbow(colors[i])))
            cv2.circle(im, (co, li), 5, color, -1)
        cv2.imwrite(os.path.join(outdir, os.path.basename(img)), im)
