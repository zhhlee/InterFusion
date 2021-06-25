# InterFusion

**KDD 2021: Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding**

InterFusion is an unsupervised MTS anomaly detection and interpretation method. It's core idea is to model the normal patterns of MTS using HVAE with jointly trained hierarchical stochastic latent variables, each of which explicitly learns low-dimensional inter-metric or temporal embeddings. You may refer to our paper for more details (coming soon).

## Getting Started

**Clone the repo**

```bash
git clone https://github.com/zhhlee/InterFusion.git && cd InterFusion
```

**Get data**

The datasets used in this paper are in folder ``data``. You may refer to ``data/Dataset Description`` for more details.

**Install dependencies (with python 3.6+)**

(virtualenv is recommended)

```bash
pip install -r requirements.txt
```

The code is tested under the following basic environments:

```
OS: Ubuntu 18.04
GPU: GTX 1080 Ti
Cuda: 9.0.176
Python: 3.6.6
```

**Run the code**

Please set the root directory of the project as your Python path.

For dataset ASD and SMD:

```bash
python algorithm/stack_train.py --dataset=omi-1			# training
python algorithm/stack_predict.py --load_model_dir=./results/stack_train/	# evaluation
```

For dataset SWaT and WADI (Note: you need to acquire these datasets first following ``data/Dataset Description`` and ``explib/raw_data_converter``):

SWaT:

```bash
python algorithm/stack_train.py --dataset=SWaT --train.train_start=21600 --train.valid_portion=0.1 --model.window_length=30 '--model.output_shape=[15, 15, 30]'	# training
python algorithm/stack_predict.py --load_model_dir=./results/stack_train/ --mcmc_track=False	# evaluation
```

WADI:

```bash
python algorithm/stack_train.py --dataset=WADI --train.train_start=259200 --train.max_train_size=789371 --train.valid_portion=0.1 --model.window_length=30 '--model.output_shape=[15, 15, 30]' # training
python algorithm/stack_predict.py --load_model_dir=./results/stack_train/ --mcmc_track=False	# evaluation
```

The default model configurations are in ``algorithm/InterFusion.py``, train configs in ``algorithm/stack_train.py``, and evaluation configs in ``algorithm/stack_predict.py``. You may overwrite the configs using command line args. For example:

```bash
python algorithm/stack_train.py --dataset=omi-1 --model.z_dim=5 --train.batch_size=128
python algorithm/stack_predict.py --load_model_dir=./results/stack_train/ --test_batch_size=100
```

**Run on your own dataset**

1. Put your train/test/label files under ``data/processed`` folder. e.g., ``ds_train.pkl``, ``ds_test.pkl``, ``ds_test_label.pkl`` with shape ``(train_length, feature_dim)``, ``(test_length, feature_dim)``, ``(test_length,)``, respectively. 
2. Put the interpretation files (optional) under ``data/interpretation_label`` folder.
3. Edit ``get_data_dim`` in ``algorithm/utils.py`` to add your dataset info.
4. Run the code following the instructions above.

**Results**

After running the algorithm, the results are shown in the ``results`` folder. The main results are:

```bash
Model: results/stack_train/result_params/
Training config: results/stack_train/config.json
Testing config: results/stack_predict/config.json
Testing statistics: results/stack_predict/result.json
```



If you find this code useful for your research, please cite our paper:

```bibTex
@inproceedings{li2021multi,
  title={Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding},
  author={Li, Zhihan and Zhao, Youjian and Han, Jiaqi and Su, Ya and Jiao, Rui and Wen, Xidao and Pei, Dan},
  booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```

