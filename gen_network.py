import sys
sys.path.insert(0, '/home/ubt/caffe/python')
import caffe
import os

from caffe import layers as L
from caffe import params as P

def conv(bottom, nout):
    """Returns Full Convolution Layer."""
    conv = L.Convolution(bottom, param=dict(lr_mult=1, decay_mult=1),
                         kernel_size=3, stride=2, num_output=nout, pad=1,
                         bias_term=False, weight_filler=dict(type='msra'))
    conv_bn = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0)] * 3, eps=1e-05, in_place=True)
    conv_scale = L.Scale(conv_bn, param=[dict(lr_mult=1, decay_mult=0)] * 2,
                         filler=dict(value=1), bias_term=True, bias_filler=dict(value=0), in_place=True)
    relu = L.ReLU(conv_scale, in_place=True)
    return conv, conv_bn, conv_scale, relu

def conv_dw(bottom, nout, stride=1):
    """Returns Depthwise Convolution Layer."""
    conv_dw = L.Convolution(bottom, param=dict(lr_mult=1, decay_mult=1),
                            kernel_size=3, stride=stride, num_output=nout, pad=1, group=nout,
                            bias_term=False, engine=P.Convolution.CAFFE, weight_filler=dict(type='msra'))
    conv_dw_bn = L.BatchNorm(conv_dw, param=[dict(lr_mult=0, decay_mult=0)] * 3, eps=1e-05, in_place=True)
    conv_dw_scale = L.Scale(conv_dw_bn, param=[dict(lr_mult=1, decay_mult=0)] * 2,
                            filler=dict(value=1), bias_term=True, bias_filler=dict(value=0), in_place=True)
    relu_dw = L.ReLU(conv_dw_scale, in_place=True)
    return conv_dw, conv_dw_bn, conv_dw_scale, relu_dw

def conv_pw(bottom, nout):
    """Returns Pointwise Convolution Layer."""
    conv_pw = L.Convolution(bottom, param=dict(lr_mult=1, decay_mult=1),
                            kernel_size=1, stride=1, num_output=nout, pad=0,
                            bias_term=False, weight_filler=dict(type='msra'))
    conv_pw_bn = L.BatchNorm(conv_pw, param=[dict(lr_mult=0, decay_mult=0)] * 3, eps=1e-05, in_place=True)
    conv_pw_scale = L.Scale(conv_pw_bn, param=[dict(lr_mult=1, decay_mult=0)] * 2,
                            filler=dict(value=1), bias_term=True, bias_filler=dict(value=0), in_place=True)
    relu_pw = L.ReLU(conv_pw_scale, in_place=True)
    return conv_pw, conv_pw_bn, conv_pw_scale, relu_pw

def mobilenet(data, label=None, num_classes=2):
    """Returns a NetSpec specifying MobileNet."""
    n = caffe.NetSpec()
    n["data"] = data

    # conv1
    n["conv1"], n["conv1/bn"], n["conv1/scale"], n["relu1"] = conv(n["data"], 32)

    # depthwise conv2_1
    n["conv2_1/dw"], n["conv2_1/dw/bn"], n["conv2_1/dw/scale"], n["relu2_1/dw"] = conv_dw(n["relu1"], 32)
    # pointwise conv2_1
    n["conv2_1/sep"], n["conv2_1/sep/bn"], n["conv2_1/sep/scale"], n["relu2_1/sep"] = conv_pw(n["relu2_1/dw"], 64)

    # depthwise conv2_2
    n["conv2_2/dw"], n["conv2_2/dw/bn"], n["conv2_2/dw/scale"], n["relu2_2/dw"] = conv_dw(n["relu2_1/sep"], 64, 2)
    # pointwise conv2_2
    n["conv2_2/sep"], n["conv2_2/sep/bn"], n["conv2_2/sep/scale"], n["relu2_2/sep"] = conv_pw(n["relu2_2/dw"], 128)

    # depthwise conv3_1
    n["conv3_1/dw"], n["conv3_1/dw/bn"], n["conv3_1/dw/scale"], n["relu3_1/dw"] = conv_dw(n["relu2_2/sep"], 128)
    # pointwise conv3_1
    n["conv3_1/sep"], n["conv3_1/sep/bn"], n["conv3_1/sep/scale"], n["relu3_1/sep"] = conv_pw(n["relu3_1/dw"], 128)

    # depthwise conv3_2
    n["conv3_2/dw"], n["conv3_2/dw/bn"], n["conv3_2/dw/scale"], n["relu3_2/dw"] = conv_dw(n["relu3_1/sep"], 128, 2)
    # pointwise conv3_2
    n["conv3_2/sep"], n["conv3_2/sep/bn"], n["conv3_2/sep/scale"], n["relu3_2/sep"] = conv_pw(n["relu3_2/dw"], 256)

    # depthwise conv4_1
    n["conv4_1/dw"], n["conv4_1/dw/bn"], n["conv4_1/dw/scale"], n["relu4_1/dw"] = conv_dw(n["relu3_2/sep"], 256)
    # pointwise conv4_1
    n["conv4_1/sep"], n["conv4_1/sep/bn"], n["conv4_1/sep/scale"], n["relu4_1/sep"] = conv_pw(n["relu4_1/dw"], 256)

    # depthwise conv4_2
    n["conv4_2/dw"], n["conv4_2/dw/bn"], n["conv4_2/dw/scale"], n["relu4_2/dw"] = conv_dw(n["relu4_1/sep"], 256, 2)
    # pointwise conv4_2
    n["conv4_2/sep"], n["conv4_2/sep/bn"], n["conv4_2/sep/scale"], n["relu4_2/sep"] = conv_pw(n["relu4_2/dw"], 512)

    # depthwise conv5_1
    n["conv5_1/dw"], n["conv5_1/dw/bn"], n["conv5_1/dw/scale"], n["relu5_1/dw"] = conv_dw(n["relu4_2/sep"], 512)
    # pointwise conv5_1
    n["conv5_1/sep"], n["conv5_1/sep/bn"], n["conv5_1/sep/scale"], n["relu5_1/sep"] = conv_pw(n["relu5_1/dw"], 512)

    # depthwise conv5_2
    n["conv5_2/dw"], n["conv5_2/dw/bn"], n["conv5_2/dw/scale"], n["relu5_2/dw"] = conv_dw(n["relu5_1/sep"], 512)
    # pointwise conv5_2
    n["conv5_2/sep"], n["conv5_2/sep/bn"], n["conv5_2/sep/scale"], n["relu5_2/sep"] = conv_pw(n["relu5_2/dw"], 512)

    # depthwise conv5_3
    n["conv5_3/dw"], n["conv5_3/dw/bn"], n["conv5_3/dw/scale"], n["relu5_3/dw"] = conv_dw(n["relu5_2/sep"], 512)
    # pointwise conv5_3
    n["conv5_3/sep"], n["conv5_3/sep/bn"], n["conv5_3/sep/scale"], n["relu5_3/sep"] = conv_pw(n["relu5_3/dw"], 512)

    # depthwise conv5_4
    n["conv5_4/dw"], n["conv5_4/dw/bn"], n["conv5_4/dw/scale"], n["relu5_4/dw"] = conv_dw(n["relu5_3/sep"], 512)
    # pointwise conv5_4
    n["conv5_4/sep"], n["conv5_4/sep/bn"], n["conv5_4/sep/scale"], n["relu5_4/sep"] = conv_pw(n["relu5_4/dw"], 512)

    # depthwise conv5_5
    n["conv5_5/dw"], n["conv5_5/dw/bn"], n["conv5_5/dw/scale"], n["relu5_5/dw"] = conv_dw(n["relu5_4/sep"], 512)
    # pointwise conv5_5
    n["conv5_5/sep"], n["conv5_5/sep/bn"], n["conv5_5/sep/scale"], n["relu5_5/sep"] = conv_pw(n["relu5_5/dw"], 512)

    # depthwise conv5_6
    n["conv5_6/dw"], n["conv5_6/dw/bn"], n["conv5_6/dw/scale"], n["relu5_6/dw"] = conv_dw(n["relu5_5/sep"], 512, 2)
    # pointwise conv5_6
    n["conv5_6/sep"], n["conv5_6/sep/bn"], n["conv5_6/sep/scale"], n["relu5_6/sep"] = conv_pw(n["relu5_6/dw"], 1024)

    # depthwise conv6
    n["conv6/dw"], n["conv6/dw/bn"], n["conv6/dw/scale"], n["relu6/dw"] = conv_dw(n["relu5_6/sep"], 1024)
    # pointwise conv6
    n["conv6/sep"], n["conv6/sep/bn"], n["conv6/sep/scale"], n["relu6/sep"] = conv_pw(n["relu6/dw"], 1024)

    # pool6
    n["pool6"] = L.Pooling(n["relu6/sep"], global_pooling=True, pool=P.Pooling.AVE)

    # fc7_ft
    n["fc7_ft"] = L.Convolution(n["pool6"], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             kernel_size=1, num_output=num_classes,
                             weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))

    if label is not None:
        n["label"] = label
        n["loss"] = L.SoftmaxWithLoss(n["fc7_ft"], n["label"])
        n["acc"] = L.Accuracy(n["fc7_ft"], n["label"])
    else:
        n["prob"] = L.Softmax(n["fc7_ft"])

    return n.to_proto()


def main():
    # Pretrained models on ImageNet
    weights = './mobilenet.caffemodel'
    assert os.path.exists(weights)

    # deploy
    deploy_data = L.Input(input_param=dict(shape=dict(dim=[1, 3, 224, 224])))
    deploy_proto = './deploy.prototxt'
    with open(deploy_proto, 'w') as f:
        f.write(str(mobilenet(data=deploy_data, num_classes=9)))
    deploy_net = caffe.Net(deploy_proto, weights, caffe.TEST)

    # train
    train_source = './data/train_shuffled.txt'
    train_transform_param = dict(scale=0.017, mirror=True, crop_size=224, mean_value=[103.94, 116.78, 123.68])
    train_data, train_label = L.ImageData(transform_param=train_transform_param, source=train_source, root_folder="./data/",
                                          batch_size=16, new_height=256, new_width=256, ntop=2)
    train_proto = './train.prototxt'
    with open(train_proto, 'w') as f:
        f.write(str(mobilenet(data=train_data, label=train_label, num_classes=9)))
    train_net = caffe.Net(train_proto, weights, caffe.TEST)

    # test
    test_source = './data/test_shuffled.txt'
    test_transform_param = dict(scale=0.017, mirror=False, crop_size=224, mean_value=[103.94, 116.78, 123.68])
    test_data, test_label = L.ImageData(transform_param=test_transform_param, source=test_source, root_folder="./data/",
                                          batch_size=2, new_height=256, new_width=256, ntop=2)
    test_proto = './test.prototxt'
    with open(test_proto, 'w') as f:
        f.write(str(mobilenet(data=test_data, label=test_label, num_classes=9)))
    test_net = caffe.Net(test_proto, weights, caffe.TEST)


if __name__ == '__main__':
    main()
