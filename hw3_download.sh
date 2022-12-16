wget https://www.dropbox.com/s/iozgaiq3p0itkmi/model2.pth?dl=1 -O model.pth
python3 -c "import clip; clip.load('ViT-B/32'); clip.load('ViT-L/14')"
