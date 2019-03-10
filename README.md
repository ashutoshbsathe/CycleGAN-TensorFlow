# CycleGAN-TensorFlow
TensorFlow implementation of CycleGAN. Modified to work with rectangular images

This repository holds the code and data I used for my CycleGAN experiment for converting gameplay of POP1 to POP2.

If you're unsure about how to run this, run `train.py` with flag `-h`
## Dependencies 
1. Python3
1. TensorFlow
## How to use 
1. Run `build_data.py` to build your datasets into tfrecords file.(Use `--help` as argument for more information)
1. Run `train.py`. This will load default settings defined in `train.py`.

`train.py` can be launched with custom flags as defined below:
```
flags:

.\train.py:
  --A: A tfrecords file for training, default:
    pop1topop2/tfrecords/pop1.tfrecords
    (default: 'pop1topop2/tfrecords/pop1.tfrecords')
  --B: B tfrecords file for training, default:
    pop1topop2/tfrecords/pop2.tfrecords
    (default: 'pop1topop2/tfrecords/pop2.tfrecords')
  --batch_size: Batch Size, default: 1
    (default: '1')
    (an integer)
  --beta1: Momentum of Adam, default: 0.5
    (default: '0.5')
    (a number)
  --image_height: Image Height, default: 200
    (default: '200')
    (an integer)
  --image_width: Image Width, default: 320
    (default: '320')
    (an integer)
  --lambda1: Weight of Forward Cycle Loss (A->B->A), default: 10
    (default: '10')
    (an integer)
  --lambda2: Weight of Backward Cycle Loss (B->A->B), default: 10
    (default: '10')
    (an integer)
  --learning_rate: Learning rate of Adam, default: 2e-4
    (default: '0.0002')
    (a number)
  --load_model: Folder of saved model for continuing the training, default: None
  --ngf: Number of gen filters in first conv layer, default: 64
    (default: '64')
    (an integer)
  --norm: Normalization [instance/batch], default: instance
    (default: 'instance')
  --pool_size: Size of Image Buffer that stores previously generated images,
    default: 50
    (default: '50')
    (an integer)
  --[no]reset_model: Allows you to reset computational graph of tensorflow,
    default: False
    (default: 'false')
  --[no]use_lsgan: Use LSGAN(MSE) or CrossEntropyLoss, default: True
    (default: 'true')
```
