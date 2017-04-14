#!/bin/bash
for dir in services/*; do (cd "$dir" && npm install); done
