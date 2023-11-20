if [ ! -d "/workspace/weights" ]; then
    mkdir "/workspace/weights"
fi

if [ ! -d "/workspace/reconstructed_images" ]; then
    mkdir "/workspace/reconstructed_images"
fi

pip3 install matplotlib
pip3 install pandas

python3 dirvae_hyperparameter.py