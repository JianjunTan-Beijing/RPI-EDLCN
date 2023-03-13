# RPI-EDLCN
 An ensemble deep learning framework based on capsule network for ncRNA–protein interaction prediction

The _utils_, _data_ and _result_ directories contain model codes, tested data sets and generated results, respectively.
The depended python packages are listed in _requirements.txt_. The package versions should be followed by users in their environments to achieve the supposed performance.

## How to run

The program is in Python 3.7.0 using [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) backends. Use the below bash command to run RPI-EDLCN.

```bash
    python main.py -d dataset
```

The parameter of _dataset_ could be RPI1807,RPI2241 and NPInter v2.0. Then, RPI-EDLCN will perform 5-fold cross validation on the specific dataset.


## Three RPI datasets

The widely used RPI benchmark datasets are organized in the _data_ directory. 

Due to the limitation of the hardware conditions of the selected RNA secondary structure method, it can only predict the secondary structure of RNA with a length of no more than 1000 nucleotides, so we preprocessed the data.

                 Dataset    | #Positive pairs | #Negative pairs  |  RNAs  | Proteins | Reference

Original set    

                RPI1807             1807              1436          1078      3131        [1]
				RPI2241             2241              2241          841       2042        [2]
                NPInter v2.0        10412             10412         4636      449         [3]

Optimal set     

                RPI1807             652               221           646       868         [1]
				RPI2241             872               872           582       1190        [2]
                NPInter v2.0        3216              3216          1085      449         [3]
 
## Help

For any questions, feel free to contact me by tanjianjun@bjut.edu.cn or start an issue instead.


[1] Pan, X.Y.; Fan, Y.X.; Yan, J.C.; Shen, H.B. IPMiner: hidden ncRNA-protein interaction sequential pattern mining with stacked autoencoder for accurate computational prediction. Bmc Genomics 2016, 17. doi:ARTN 582 10.1186/s12864-016-2931-8.

[2] Peng, C.; Han, S.; Zhang, H.; Li, Y. RPITER: a hierarchical deep learning framework for ncRNA–protein interaction prediction. Int J Mol Sci. 2019;20(5):1070. doi: 10.3390/ijms20051070.

[3] Yuan, J.;Wu,W.; Xie, C.Y.; Zhao, G.G.; Zhao, Y.; Chen, R.S. NPInter v2.0: an updated database of ncRNA interactions. Nucleic Acids Research 2014, 42, D104–D108. doi:10.1093/nar/gkt1057.
## Reference:
RPI-EDLCN: an ensemble deep learning framework based on capsule network for ncRNA–protein interaction prediction