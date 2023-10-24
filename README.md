# betaSWWDirVAE

## Priorities
1. Check if existing github implementations of SWWAE and DirVAE are correct
2. Integrate DirVAE into SWWAE
3. Train model (**Unsupervised**) on small amount of data (CIFAR-10/100)
4. Get initial analysis (L2 norm of latent space, visualise latent space and compare to literature, etc)
   - Purpose of this is to verify that the properties from the seperate models are preserved when they are combined
6. upload model to GPU cloud and train (**Unsupervised**) on ImageNet (From Kaggle, downsample to 32x32)
7. Tune hyper parameters
8. Get final model
9. Take encoder part, attach to EfficientNet and train (**Supervised**) on CIFAR-10 and 100 datasets

## Tasks
### Timur
- Setup new cluster, clone repository

### Daniil and Baris
- Work together to figure out if SWWAE implementation is correct

### Riccardo
- figure out if DirVAE implementation is correct
