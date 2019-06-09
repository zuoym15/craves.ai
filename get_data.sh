Root=http://www.cs.jhu.edu/~qiuwch/craves/
mkdir -p checkpoint/ saved_results/
wget -c $Root/checkpoint.pth.tar -O ./checkpoint/checkpoint.pth.tar
wget -c $Root/dataset/test_20181024.zip -O ./data/test_20181024.zip
unzip ./data/test_20181024.zip -d ./data/
