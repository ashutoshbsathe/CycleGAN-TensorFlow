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

`train.py` can be launched with custom flags:
<details>
  <summary>Expand to view all flags</summary>
<pre>
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
</pre>
</details>

## Common Pitfalls :
1. Make sure that all of your images are in the shape that will be passed to the model. If some images are irregular, the reader may *squeeze* the image to make it of suitable size. This is generally not recommended. Possible solution is to have the image cropped in proper required resolution
1. If you see inversion of colors(dark colors in place of light colors) for a considerable amount of steps, you should stop the training and restart it. This is called model collapse and the results may look something like the image below:<br>
![Model Collapse](https://raw.githubusercontent.com/ashutoshbsathe/CycleGAN-TensorFlow/master/images/mode_collapse.png)
1. Poor results : CycleGANs are sensitive to initial weight initializations. If the results, are unsatisfactory, retraining the model from scratch might help. A lot of people online got incredible results on original datasets after 4th or 5th try so it's worth retrying.
