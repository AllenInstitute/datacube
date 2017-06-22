SystemJS.config({
  baseURL: '/node_modules/',
  packages: {
    '/node_modules/': {
      defaultJSExtensions: 'js'
    }
  },
  map: {
    "autobahn": '/node_modules/autobahn-js-built/autobahn.min.js',
    "pako": '/node_modules/pako/dist/pako.min.js',
    "text-encoding": '/node_modules/text-encoding/lib/encoding.js'
  }
});
