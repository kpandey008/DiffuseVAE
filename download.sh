FILE=$1

if  [ $FILE == "celeba" ]; then
    URL=https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
    ZIP_FILE=./celeba.zip
    gdown -O $ZIP_FILE $URL
    unzip $ZIP_FILE -d ./
    rm $ZIP_FILE

elif  [ $FILE == "celebamask-hq" ]; then
    URL=https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv
    ZIP_FILE=./celebamask_hq.zip
    gdown -O $ZIP_FILE $URL
    unzip $ZIP_FILE -d ./
    rm $ZIP_FILE

elif  [ $FILE == "celebahq" ]; then
    # Provided by the StarGAN v2 authors
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./celeba_hq.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./
    rm $ZIP_FILE

elif  [ $FILE == "afhq" ]; then
    # Provided by the StarGAN v2 authors
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./afhq.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./
    rm $ZIP_FILE

else
    echo "Available arguments are afhq, celeba, celebamask-hq and celebahq"
    exit 1

fi