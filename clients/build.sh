node_modules/babel-cli/bin/babel.js src -d lib/es2015
node_modules/babel-cli/bin/babel.js src -d lib/amd --plugins transform-es2015-modules-amd
node_modules/babel-cli/bin/babel.js src -d lib/commonjs --plugins transform-es2015-modules-commonjs
node_modules/babel-cli/bin/babel.js src -d lib/systemjs --plugins transform-es2015-modules-systemjs
node_modules/babel-cli/bin/babel.js src -d lib/umd --plugins transform-es2015-modules-umd
BABEL_ENV=rollup node_modules/rollup/bin/rollup -c
