cat local-python-threaded-r2.txt | cut -d' ' -f2 | sort -n | cut -d'.' -f1 | grep -v '^$' | grep -v '[^0-9]' | uniq -c
