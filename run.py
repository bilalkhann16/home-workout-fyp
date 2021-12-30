# test_image = TfPoseEstimator.draw_humans_1(image, bodys, imgcopy=False)
import argparse
import tensorflow as tf
import sys
import time
import logging
import cv2
import numpy as np
import tf_slim as slim
import vgg
from cpm import PafNet
import common
from tensblur.smoother import Smoother
from estimator import PoseEstimator, TfPoseEstimator
tf.compat.v1.disable_eager_execution()

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Webcam feed.')
    parser.add_argument('--checkpoint_path', type=str, default='/Users/bilalk/Desktop/FYP.nosync/unofficial-implement-of-openposet_v2/checkpoints/model_file/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='/Users/bilalk/Desktop/FYP.nosync/unofficial-implement-of-openposet_v2/checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--image', type=str, default=None)
    # parser.add_argument('--run_model', type=str, default='img')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--train_vgg', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=False)

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    logger.info('checkpoint_path: ' + checkpoint_path)

    with tf.name_scope('inputs'):
        raw_img = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        img_size = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    # Pre-Processing
    img_normalized = raw_img / 255 - 0.5
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    logger.info('initializing model...')
    net = PafNet(inputs_x=vgg_outputs, use_bn=args.use_bn)  #false  #setting the variables.
    hm_pre, cpm_pre, added_layers_out = net.gen_net()  #expanding it into two parts  5 aur 7 me divide krna hai.
    print ("hm_pre", hm_pre)
    print ("cpm_pre", cpm_pre)
    print ("added_layers_out", added_layers_out)
    
    print ("\n\n added layers", added_layers_out)
    print ("hm_up,cpm_up", hm_pre[5], cpm_pre[5])

    hm_up = tf.compat.v1.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.compat.v1.image.resize_area(cpm_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    
    print ("hm_up", hm_up)
    print ("cpm_up", cpm_up)
    
    smoother = Smoother({'data': hm_up}, 25, 3.0)  #blur the image.
    gaussian_heatMat = smoother.get_output()  #noise reduction/smoothing

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    # trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    # trainable_var_list = []
    trainable_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if args.train_vgg:
        trainable_var_list = trainable_var_list + tf.compat.v1.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

    restorer = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.compat.v1.train.Saver(trainable_var_list)
    logger.info('initialize session...')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.group(tf.compat.v1.global_variables_initializer()))
        logger.info('restoring vgg weights...')
        restorer.restore(sess, args.backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        # saver.restore(sess, args.checkpoint_path + 'model-55000.ckpt')
        logger.info('initialization done')
        if args.image is None:
            if args.video is not None:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0)
            _, image = cap.read()

            fps = cap.get(cv2.CAP_PROP_FPS)
            ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info('fps@%f' % fps)
            size = [320,320]
            h = int(654 * (ori_h / ori_w))
            time_n = time.time()
            while True:
                _, image = cap.read()
                img = np.array(cv2.resize(image, (256,144)))  #change this to change the fps
                # cv2.imshow('raw', img)
                img_corner = np.array(cv2.resize(image, (360, int(360*(ori_h/ori_w)))))
                img = img[np.newaxis, :]
                print ("img_corner", img_corner)
                # print ("image;", img)
                peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                     feed_dict={raw_img: img, img_size: size})
                
                print ("peaks",peaks[0])
                print ("heatmap",heatmap[0])
                print ("vector",vectormap[0])
                
                bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                # print ("bodys: " , bodys)
                # time.sleep(5)
                # cv2.imshow('firone', image)
                image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
                fps = round(1 / (time.time() - time_n), 2)
                image = cv2.putText(image, str(fps)+'fps', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
                time_n = time.time() 
                
                cv2.imshow('Output Image', image)
                # if args.save_video is not None:
                #     video_saver.write(image)
                cv2.waitKey(1)
                  
        else:   # to be deleted!
            print ("camera failed")