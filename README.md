# SINDy Autoencoder - PyTorch (Elastic Pendulum fork)
From original repo:

<blockquote>
PyTorch implementation of the SINDy Autoencoder from the paper "Data-driven discovery of coordinates and governing equations"
by Champion et al.
<br><br>
The original implementation is in TensorFlow and is located at: https://github.com/kpchamp/SindyAutoencoders <br>
The TensorFlow implementation was used as a reference and, for some components, code was directly copied over.
In the latter case, each file with copied code has a reference to which file in the original repository it was taken from.
</blockquote>
<br>

This fork is designed exclusively to produce/learn a simulated **elastic pendulum**.

Fully-connected (`src/models/SINDyAE_o2.py`) and convolutional (`src/models/SINDyConvAE_o2.py`) models are provided.

## Arguments
Each script in the training pipeline collects arguments using `cmd_line.py`. Create a master file with your arguments to simplify management, e.g.

**`args.txt`**
```text
--session_name 05-25-2025_0
--model SINDyConvAE_o2
--train_initial_conds 200
--val_initial_conds 20
--test_initial_conds 20
--timesteps 500
--batch_size 30
--epochs 500
--lambda_ddx 0
--lambda_ddz 0
--lambda_reg 0
```

## Datasets
To create the elastic pendulum datasets:

`cat args.txt | xargs python create_elastic_pendulum.py`.

See `cmd_line.py` for the full list of dataset creation args.

## Training
To train a model from scratch:

`cat args.txt | xargs python main.py`

To train from a checkpoint:

`cat args.txt | xargs python main.py --load_cp 1`

## Printing equations
To print the governing equations discovered by a model:

`cat args.txt | xargs python experiments.py`

## Creating movie
To produce an "input vs output" movie of a sample test set trajectory:

`cat args.txt | xargs python elastic_pendulum_AE_video.py`

The movie will appear in the `trained_models` directory.