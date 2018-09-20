for run in {1..16}
do
    #./dist/load_test <(paste <(cat trace.txt | cut -d' ' -f1) <(cat trace.txt | cut -d' ' -f2- | sort --random-sort)) ws://tdatacube:8080/ws >> $1 2>&1 &
    ./dist/load_test <(paste <(cat trace.txt | cut -d' ' -f1) <(cat trace.txt | cut -d' ' -f2- | sort --random-sort)) ws://tdatacube:8080/ws 2>&1 &
    #./dist/load_test trace.txt ws://localhost:8080/ws >> $1 2>&1 &
    #./dist/load_test trace.txt ws://localhost:8080/ws &
    #./dist/load_test trace.txt ws://localhost:8080/ws &
    #./dist/load_test <(paste <(cat trace.txt | cut -d' ' -f1) <(cat trace.txt | cut -d' ' -f2- | sort --random-sort)) ws://localhost:8080/ws &
done
