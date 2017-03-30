#!/bin/bash

until python pandas_service.py $@; do
    echo "Pandas service crashed with exit code $?.  Respawning..." >&2
    sleep 1
done
