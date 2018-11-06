def make_net(phase='train'):
    n = caffe.NetSpec()
    if phase == 'train':
        batch_size = train_batch_size
    elif phase == 'test':
        batch_size = test_batch_size
    if phase != 'deploy':
        n.data = L.Data(source=join(data_source, phase+'-img'), backend=P.Data.LMDB, batch_size=batch_size,
                        transform_param=dict(mirror=1), ntop=1)
        n.label = L.Data(source=join(data_source, phase+'-count'), backend=P.Data.LMDB, batch_size=batch_size, ntop=1)
    else:   # deploy 阶段输入为一张待预测图片
        n.data = L.Input(shape=dict(dim=[1, 3, 320, 240]))
    # 第一列
    n.conv1_1 = L.Convolution(n.data, kernel_size=9, pad=4, num_output=16, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    if phase == 'train':
        n.bn1_1 = L.BatchNorm(n.conv1_1, use_global_stats=False, name='bn1_1')
    else:
        n.bn1_1 = L.BatchNorm(n.conv1_1, use_global_stats=True, name='bn1_1')
    n.scale1_1 = L.Scale(n.bn1_1, bias_term=True, name='scale1_1')

    n.PReLU1_1 = L.PReLU(n.scale1_1, in_place=True)

    
    #n.resh1_1 = L.Reshape(n.PReLU1_1, shape=dict(dim=[0, 0, 160, 140]))
    n.pool1_1 = L.Pooling(n.PReLU1_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.pool1_1_2 = L.Pooling(n.pool1_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.conv1_2 = L.Convolution(n.pool1_1, kernel_size=7, pad=3, num_output=32, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU1_2 = L.PReLU(n.conv1_2, in_place=True)
    if phase == 'train':
        n.bn1_2 = L.BatchNorm(n.PReLU1_2, use_global_stats=False, name='bn1_2')
    else:
        n.bn1_2 = L.BatchNorm(n.PReLU1_2, use_global_stats=True, name='bn1_2')
    n.scale1_2 = L.Scale(n.bn1_2, bias_term=True, name='scale1_2')
    n.pool1_2 = L.Pooling(n.scale1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.conv1_3 = L.Convolution(n.pool1_2, kernel_size=7, pad=3, num_output=16, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU1_3 = L.PReLU(n.conv1_3, in_place=True)

    n.conv1_4 = L.Convolution(n.PReLU1_3, kernel_size=7, pad=3, num_output=8, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU1_4 = L.PReLU(n.conv1_4, in_place=True)

    n.conv1_5 = L.Convolution(n.PReLU1_4, kernel_size=7, pad=3, num_output=8, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU1_5 = L.PReLU(n.conv1_5, in_place=True)

    # 第2列
    n.conv2_1 = L.Convolution(n.data, kernel_size=7, pad=3, num_output=24, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])


    if phase == 'train':
        n.bn2_1 = L.BatchNorm(n.conv2_1, use_global_stats=False, name='bn2_1')
    else:
        n.bn2_1 = L.BatchNorm(n.conv2_1, use_global_stats=True, name='bn2_1')
    n.scale2_1 = L.Scale(n.bn2_1, bias_term=True, name='scale2_1')
    n.PReLU2_1 = L.PReLU(n.scale2_1, in_place=True)

    

    n.pool2_1 = L.Pooling(n.PReLU2_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.pool2_1_2 = L.Pooling(n.pool2_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.conv2_2 = L.Convolution(n.pool2_1, kernel_size=5, pad=2, num_output=48, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU2_2 = L.PReLU(n.conv2_2, in_place=True)
    if phase == 'train':
        n.bn2_2 = L.BatchNorm(n.PReLU2_2, use_global_stats=False, name='bn2_2')
    else:
        n.bn2_2 = L.BatchNorm(n.PReLU2_2, use_global_stats=True, name='bn2_2')
    n.scale2_2 = L.Scale(n.bn2_2, bias_term=True, name='scale2_2')


    n.pool2_2 = L.Pooling(n.scale2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.conv2_3 = L.Convolution(n.pool2_2, kernel_size=5, pad=2, num_output=24, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU2_3 = L.PReLU(n.conv2_3, in_place=True)

    n.conv2_4 = L.Convolution(n.PReLU2_3, kernel_size=5, pad=2, num_output=12, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU2_4 = L.PReLU(n.conv2_4, in_place=True)

    n.conv2_5 = L.Convolution(n.PReLU2_4, kernel_size=5, pad=2, num_output=12, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU2_5 = L.PReLU(n.conv2_5, in_place=True)

    # 第3列
    n.conv3_1 = L.Convolution(n.data, kernel_size=5, pad=2, num_output=32, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])

    if phase == 'train':
        n.bn3_1 = L.BatchNorm(n.conv3_1, use_global_stats=False, name='bn3_1')
    else:
        n.bn3_1 = L.BatchNorm(n.conv3_1, use_global_stats=True, name='bn3_1')
    n.scale3_1 = L.Scale(n.bn3_1, bias_term=True, name='scale3_1')

    n.PReLU3_1 = L.PReLU(n.scale3_1, in_place=True)


    n.pool3_1 = L.Pooling(n.PReLU3_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.pool3_1_2 = L.Pooling(n.pool3_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.conv3_2 = L.Convolution(n.pool3_1, kernel_size=3, pad=1, num_output=64, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU3_2 = L.PReLU(n.conv3_2, in_place=True)
    if phase == 'train':
        n.bn3_2 = L.BatchNorm(n.PReLU3_2, use_global_stats=False, name='bn3_2')
    else:
        n.bn3_2 = L.BatchNorm(n.PReLU3_2, use_global_stats=True, name='bn3_2')
    n.scale3_2 = L.Scale(n.bn3_2, bias_term=True, name='scale3_2')


    n.pool3_2 = L.Pooling(n.scale3_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    n.conv3_3 = L.Convolution(n.pool3_2, kernel_size=3, pad=1, num_output=32, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU3_3 = L.PReLU(n.conv3_3, in_place=True)


    n.conv3_4 = L.Convolution(n.PReLU3_3, kernel_size=3, pad=1, num_output=16, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU3_4 = L.PReLU(n.conv3_4, in_place=True)


    n.conv3_5 = L.Convolution(n.PReLU3_4, kernel_size=3, pad=1, num_output=16, weight_filler=dict(type='xavier'),
                              param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])
    n.PReLU3_5 = L.PReLU(n.conv3_5, in_place=True)

    n.concat1 = L.Concat(n.pool1_1_2, n.pool2_1_2, n.pool3_1_2,
                        n.PReLU1_5, n.PReLU2_5, n.PReLU3_5, axis=1)
    #n.concat1 = L.Concat(n.PReLU1_4, n.PReLU2_4, n.PReLU3_4, axis=1)
    n.conv4 = L.Convolution(n.concat1, kernel_size=1, pad=2, num_output=10, weight_filler=dict(type='xavier'),
                            param=[dict(lr_mult=0.01, decay_mult=1), dict(lr_mult=0.02, decay_mult=0)])

    nout = leaf
    if nout > 0:
        assert (nout >= int(pow(2, treeDepth - 1) - 1))
        nout = nout
    else:
        if ntree == 1:
            nout = int(pow(2, treeDepth - 1) - 1)
        else:
            nout = int((pow(2, treeDepth - 1) - 1) * ntree * 2 / 3)
    # print 'nout', nout
    n.fc1 = L.InnerProduct(n.conv4, num_output=1000, bias_term=True,
                           weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0.1),
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.fc8 = L.InnerProduct(n.fc1, num_output=nout, bias_term=True, weight_filler=dict(type='xavier'),
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], name='fc8a')

    if phase == 'train':
        all_data_vec_length = int(nTrain / train_batch_size)
        n.loss = L.NeuralDecisionDLForestWithLoss(n.fc8, n.label,
                                                  param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                                                  neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree,
                                                                                    num_classes=maxCount - minCount + 1,
                                                                                    iter_times_class_label_distr=20,
                                                                                    iter_times_in_epoch=100,
                                                                                    all_data_vec_length=all_data_vec_length,
                                                                                    drop_out=False),
                                                  name='probloss')
    elif phase == 'test':
        n.pred = L.NeuralDecisionForest(n.fc8, n.label,
                                        neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree,
                                                                          num_classes=maxCount - minCount + 1),
                                        name='probloss')
        n.MAE = L.MAE(n.pred, n.label)
    elif phase == 'deploy':
        n.pred = L.NeuralDecisionForest(n.fc8, neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree,
                                                                                 num_classes=maxCount - minCount + 1),
                                        name='probloss')
    return n.to_proto()
