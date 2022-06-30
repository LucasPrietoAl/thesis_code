# thesis_code

This code is based on the [DGL implementation] (https://github.com/dmlc/dgl/blob/master/examples/pytorch/correct_and_smooth/README.md) of [Correct&Smooth] (https://github.com/CUAI/CorrectAndSmooth). 

* **LightGBM version of our method**

```bash
python main.py --model linear --dropout 0.5 --epochs 1000
python main.py --model linear --pretrain --correction-alpha 0.99 --smoothing-alpha 0.75 --correction-adj DA --autoscale
```


* **Linear version of our method**

```bash
python main.py --model linear --dropout 0.5 --epochs 1000
python main.py --model linear --pretrain --correction-alpha 0.98 --smoothing-alpha 0.90 --autoscale --aggregation Linear
```