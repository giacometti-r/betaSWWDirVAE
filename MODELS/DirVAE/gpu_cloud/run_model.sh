if [ ! -d "/weights" ]; then
    mkdir "/weights"
fi

if [ ! -d "/reconstructed_images" ]; then
    mkdir "/reconstructed_images"
fi

python3 dirvae_pytorch.py