import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class CardObjRegress:
    def __init__(self):
        self.img_width = 320
        self.img_height = 320
        self.gt_width = 80
        self.gt_height = 80
        self.sess = tf.Session()

        model_path = './models/KPHGregress.pb'
        graph = self.load_graph(model_path)
        sess = tf.Session(graph=graph)
        self.sess = sess

        self.input_node = self.sess.graph.get_tensor_by_name('input_1:0')
        self.pred = self.sess.graph.get_tensor_by_name('hg2_head_pred_/BiasAdd:0')

    def load_graph(self, frozen_graph_filename):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name = '',
            )
        return graph


    def check_valid(self, image, regs, im_name=''):
        IsValid = True
        if not ((np.array([np.cross(regs[1] - regs[0], regs[2] - regs[1]),
                np.cross(regs[2] - regs[1], regs[3] - regs[2]),
                np.cross(regs[3] - regs[2], regs[0] - regs[3])]) > 0).all() or
        (np.array([np.cross(regs[1] - regs[0], regs[2] - regs[1]),
                np.cross(regs[2] - regs[1], regs[3] - regs[2]),
                np.cross(regs[3] - regs[2], regs[0] - regs[3])]) < 0).all()):
            IsValid = False

        return IsValid


    def regress(self, img, visflag=False, im_name=''):
        img_r = cv2.resize(img, (self.img_width, self.img_height))
        input_node = self.input_node
        pred = self.pred
        result = self.sess.run(pred, feed_dict={input_node: np.array([img_r])})

        res = np.squeeze(result)
        kp = []
        colors = np.linspace(0, 1, 4)
        for i in range(4):
            pred = res[:, :, i]
            pts_cor = np.mean(np.where(pred == np.max(pred)), axis=1)
            L = pts_cor[0] / self.gt_height * img.shape[0]
            C = pts_cor[1] / self.gt_width * img.shape[1]
            kp.append([C, L])
            if visflag:
                Cint, Lint = int(C + 0.5), int(L + 0.5)
                color = tuple(map(lambda x: x*255, plt.cm.rainbow(colors[i])))
                cv2.circle(img, (Cint, Lint), 3, color, -1)

        if visflag:
            cv2.imwrite('./results/objreg/%s.jpg' % im_name, img)

        regs = np.array(kp)
        IsValid = self.check_valid(img, regs, im_name)

        return regs, IsValid


    def close(self):
        self.sess.close()
