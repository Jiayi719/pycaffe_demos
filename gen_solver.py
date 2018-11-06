import sys
sys.path.insert(0, '/home/ubt/caffe/python')

from caffe.proto import caffe_pb2


def solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 40  # Test after every 40 training iterations.
        s.test_iter.append(2007)  # Test on 2007 batches each time we test.

    # # Specify location of the train network.
    # s.train_net = train_net_path

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    s.max_iter = 400  # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = 0.001

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'poly' the learning rate by a polynomial decay, i.e.
    # base_lr = (1 - iter / max_iter) ^ (power)
    s.lr_policy = 'poly'
    s.power = 1.0

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 1e-4

    # Display the current training loss every 40 iterations.
    s.display = 40
    s.average_loss = 40

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 100 iterations -- four times during training.
    s.snapshot = 100
    s.snapshot_prefix = 'snapshot/custom'

    # Train on the GPU. Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    return s

def main():
    solver_proto = './solver.prototxt'
    with open(solver_proto, 'w') as f:
        f.write(str(solver('./train.prototxt', './test.prototxt')))

if __name__ == '__main__':
    main()
