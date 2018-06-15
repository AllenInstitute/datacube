#!/bin/bash

until python -u locator.py $@; do
    echo "Service crashed with exit code $?.  Respawning..." >&2
    sleep 1
done
