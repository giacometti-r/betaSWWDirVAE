if [ ! -d "/weights" ]; then
    mkdir "/weights"
fi

if [ ! -d "/reconstructed_images" ]; then
    mkdir "/reconstructed_images"
fi

pip3 install matplotlib
pip3 install pandas

python3 dirvae_pytorch.py