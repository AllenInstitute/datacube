rm -f $PWD/log/*
qsub -q informatics -t 1-$1 -d $PWD/log/ $PWD/../load_test.sh
