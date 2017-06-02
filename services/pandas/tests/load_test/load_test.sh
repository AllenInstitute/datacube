for i in $(seq $(nproc --a)); do
    $PWD/../dist/load_test $PWD/../trace.txt &
done
