#!/bin/bash

until python server.py $@; do
    echo "Datacube service crashed with exit code $?.  Respawning..." >&2
    sleep 1
done
