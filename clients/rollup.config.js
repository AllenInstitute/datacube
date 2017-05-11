import babel from 'rollup-plugin-babel';

export default {
  entry: 'src/all.js',
  dest: 'dist/clients.js',
  format: 'iife',
  moduleName: 'datacubejs',
  plugins: [
    babel({
      exclude: 'node_modules/**'
    })
  ]
};
