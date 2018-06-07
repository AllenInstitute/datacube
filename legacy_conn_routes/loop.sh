
#!/bin/bash

until pipenv run python conn_bridge.py $@; do
    echo "Service crashed with exit code $?.  Respawning..." >&2
    sleep 1
done