def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE
    s.type = 'SGD'
    s.display = 5
    #s.base_lr = 1e-1
    s.base_lr = 2e-2
    #s.base_lr = 2e-3
    s.lr_policy = "step"
    s.gamma = 0.5
    s.momentum = 0.9
    s.stepsize = 10000
    s.max_iter = maxIter
    s.snapshot = 5000
    snapshot_prefix = join(dirname(__file__), 'model_4')
    print snapshot_prefix
    if not isdir(snapshot_prefix):
        os.makedirs(snapshot_prefix)
    s.snapshot_prefix = snapshot_prefix
    s.train_net = join(tmp_dir, 'train.prototxt')
    s.test_net.append(join(tmp_dir, 'test.prototxt'))
    s.test_interval = maxIter + 1  # will test mannualy
    s.test_iter.append(test_iter)
    s.test_initialization = False
    # s.debug_info = True
    return s
