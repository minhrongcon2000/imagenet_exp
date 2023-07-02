cd imagenet
mkdir train
mkdir val

kaggle datasets download -d sautkin/imagenet1k0
kaggle datasets download -d sautkin/imagenet1k1
kaggle datasets download -d sautkin/imagenet1k2
kaggle datasets download -d sautkin/imagenet1k3
kaggle datasets download -d sautkin/imagenet1kvalid

unzip imagenet1k0.zip -d train
rm imagenet1k0.zip
unzip imagenet1k1.zip -d train
rm imagenet1k1.zip
unzip imagenet1k2.zip -d train
rm imagenet1k2.zip
unzip imagenet1k3.zip -d train
rm imagenet1k3.zip
unzip imagenet1kvalid.zip -d val
rm imagenet1kvalid.zip