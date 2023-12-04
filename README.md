# General Neural Network CNN Classificator Pytorch
The project involves the implementation of a Convolutional Neural Network (CNN) using the PyTorch library, with the central goal of providing a generic and flexible solution for any image dataset. The CNN architecture is designed to be easily adaptable to different domains and specific characteristics of datasets, enabling a wide range of applications. The code is structured in a modular way, facilitating the customization of hyperparameters, network architecture, and loss function. This approach aims to provide a versatile tool for researchers and developers seeking an efficient and adaptable solution for image classification tasks in various image contexts and domains.


## Datasets
Cat and Dog: https://www.kaggle.com/datasets/tongpython/cat-and-dog

Food Images (Food-101): https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101

Intel Image Classification: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data

## Download the weights of the trained network
Cat and Dog: https://drive.google.com/file/d/1n4_FN2VCA_OSZ-48i9vu8H9lkQguRcDa/view?usp=sharing

Food Images (Food-101): https://drive.google.com/file/d/1MUNP94VsOiKPIC3Gdt09loM_1RDOQTxz/view?usp=sharing

Intel Image Classification: https://drive.google.com/file/d/1t1HPU6Zq6g39dTxj4015MUAnnhJ6qHfB/view?usp=sharing


---
 
 # Executando o Arquivo inf721_dataset_builder.py

Para executar o script `inf721_dataset_builder.py`, siga as instruções abaixo.

## Pré-requisitos
Certifique-se de ter Python instalado em seu ambiente. Se ainda não tiver, você pode baixá-lo em [python.org](https://www.python.org/).

## Passos para Execução

1. Navegue até o diretório onde o script está localizado:

    ```bash
    cd caminho/do/repo
    ```

2. Execute o script `inf721_dataset_builder.py`, fornecendo os seguintes argumentos:

    - `--DIRECTORY_SOURCE_PATH`: Caminho do diretório do conjunto de dados.
    - `--PATH_DESTINATION` (opcional): Caminho de destino para os dados processados (padrão: 'fodo-./').

    Exemplo:

    ```bash
    python inf721_dataset_builder.py --DIRECTORY_SOURCE_PATH caminho/do/seu/dataset --PATH_DESTINATION caminho/de/destino
    ```

    Substitua `caminho/do/seu/dataset` pelo caminho real do seu conjunto de dados e `caminho/de/destino/` pelo caminho desejado para armazenar os dados processados.

4. Aguarde a conclusão da execução do script.

Isso é tudo! Agora você deve ter seus dados processados no caminho de destino especificado.

---

# Executando o Arquivo inf721_train.py

Agora que o conjunto de dados está preparado usando o script `inf721_dataset_builder.py`, siga as instruções abaixo para treinar o modelo com o script `inf721_train.py`.

## Passos para Execução

1. Execute o script `inf721_train.py` fornecendo os seguintes argumentos:

    - `--BATCH_SIZE`: Tamanho do lote para o treinamento (padrão: 32).
    - `--NUM_EPOCHS`: Número de épocas para o treinamento (padrão: 50).
    - `--MODEL_SAVE_PATH`: Caminho para salvar o modelo treinado (padrão: './').
    - `--MODEL_NAME`: Nome do modelo (padrão: 'model').
    - `--DATASET_NAME`: Nome do conjunto de dados (certifique-se de fornecer o mesmo nome usado na preparação do conjunto de dados).
    - `--DATASET_PATH`: Caminho do conjunto de dados (padrão: './data').
    - `--CLASS_FILE`: Caminho para o arquivo de rótulos de classes (padrão: './class_labels.txt').

    Exemplo:

    ```bash
    python inf721_train.py --BATCH_SIZE 32 --NUM_EPOCHS 50 --MODEL_SAVE_PATH caminho/do/salvar/modelo --MODEL_NAME meu_modelo --DATASET_NAME nome_do_seu_dataset --DATASET_PATH caminho/do/seu/dataset --CLASS_FILE caminho/do/seu/class_labels.txt
    ```

    Substitua os valores com os apropriados para o seu projeto.

3. Aguarde o término do treinamento do modelo.

Após a conclusão destes passos, você terá treinado seu modelo com os dados preparados anteriormente. Certifique-se de ajustar os caminhos e parâmetros conforme necessário para o seu projeto específico.


---

# Executando o Arquivo Python `inf721_inference.py`

Após treinar o modelo usando o script `inf721_train.py`, você pode realizar inferências em novas imagens usando o script `inf721_inference.py`. Siga as instruções abaixo:

## Passos para Execução

1. Certifique-se de ter concluído o treinamento do modelo usando o script `inf721_train.py`.

2. Execute o script `inf721_inference.py`, fornecendo os seguintes argumentos:

    - `--IMAGE_PATH`: Caminho para a imagem de entrada que você deseja realizar inferência.
    - `--MODEL_PATH`: Caminho para o arquivo do modelo treinado.
    - `--CLASS_LABELS_FILE`: Caminho para o arquivo de texto contendo rótulos de classe (padrão: 'class_labels.txt').

    Exemplo:

    ```bash
    python inf721_inference.py --IMAGE_PATH caminho/da/sua/imagem.jpg --MODEL_PATH caminho/do/seu/modelo/modelo.pth --CLASS_LABELS_FILE caminho/do/seu/class_labels.txt
    ```

    Substitua os valores com os apropriados para o seu projeto.

4. Aguarde a conclusão do processo de inferência.

Após a execução destes passos, você terá realizado inferências usando o modelo treinado. Certifique-se de ajustar os caminhos e parâmetros conforme necessário para o seu projeto específico.

