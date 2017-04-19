babel src -d lib/amd --plugins transform-es2015-modules-amd
babel src -d lib/commonjs --plugins transform-es2015-modules-commonjs
babel src -d lib/systemjs --plugins transform-es2015-modules-systemjs
babel src -d lib/umd --plugins transform-es2015-modules-umd
BABEL_ENV=rollup rollup -c
