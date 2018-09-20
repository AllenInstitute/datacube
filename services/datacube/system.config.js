SystemJS.config({
  baseURL: './node_modules/',
  transpiler: 'plugin-babel',
  packages: {
    './node_modules/': {
      defaultJSExtensions: 'js'
    }
  },
  map: {
    "autobahn": './node_modules/autobahn-js-built/autobahn.min.js',
    "pako": './node_modules/pako/dist/pako.min.js',
    "text-encoding": './node_modules/text-encoding/lib/encoding.js',
    "plugin-babel": './node_modules/systemjs-plugin-babel/plugin-babel.js',
    "systemjs-babel-build": './node_modules/systemjs-plugin-babel/systemjs-babel-browser.js'
  }
});
