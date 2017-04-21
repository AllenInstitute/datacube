#!/bin/bash
for dir in services/*; do (cd "$dir" && [[ -e "package.json" ]] && npm install); done
