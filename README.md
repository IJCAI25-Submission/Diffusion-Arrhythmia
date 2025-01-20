# Label Efficient and Personalized Arrhythmia Diagnosis via Diffusion Models

To run the training and testing, you should download the dataset first by clicking [here](https://drive.google.com/file/d/1Btd4Pnnu3dbRu6WCsOpJbauLtJV3PdnW/view?usp=sharing)

After unzip the `Data.zip`, you wil get a `Data` fold with structure shown as follow:

```shell
├── Data
  ├── Dataset
  ├── Frequency
  └── Lorenz

```

You need to move all three subfolders to the root path of the project and you will get a file structure shown as follow: 

```shell
├── Baselines
│  ├── MoCo
│  ├── EffNet
│  └── Models
├── Dataset
│  ├── data_ChapmanShaoxing_segments
│  └── data_LTAF_segments
├── Frequency
├── Lorenz
├── Diffusion_Based
├── requirements.txt
└── run.sh
```

The `Baselines` fold includes the pre-training, fine-tuning, and testing code of two baselines EfficientECG and MoCo.

The `Diffusion_based` fold includes the fine-tuning and testing code of the proposed diffusion-based method.

You can install the required package using the command `pip install -r requirements.txt`.

Pre-trained models are also provided, to conduct test with the pre-trained model, use following command:

```shell
bash ./run.sh <method> <task>
```

`<method>` can be selected in `eff`, `moco`, and `diffEcg`.

`<task>` can be selected in `generalization` and `personalization`.
