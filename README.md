# Train ResNet-152 with PyTorch using [Composer Library](https://www.mosaicml.com/composer) of MosaicML and Track Experiments with [Weights & Biases](https://wandb.ai/).

## Steps to run the code

1. Install Weights & Biases
   
    ```%pip install wandb -q```  
2. Install PyTorch Model Summary Library for getting layer by layer summary of a model

    ```pip install pytorch-model-summary```
3. Install PTFLOPS for counting floating point operations

    ```pip install ptflops```
4. Login to your WandB account so you can log all your metrics

    ```import wandb```
   
    ```wandb.login()```
5. Initialize a new run and specify your project name on WandB

    ```wandb.init(project="[YOUR_PROJECT_NAME]")```
6. Install Composer Library

   ```!pip install mosaicml```
7. Execute the [main.py](https://github.com/abdulsam/Better_Fatser_Models_with_Composer/blob/main/main.py) file.
