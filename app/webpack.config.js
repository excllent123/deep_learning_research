var webpack = require('webpack')

module.exports = {
    entry: './app.js',
    output: {
          path: __dirname,
          filename: 'bundle.js'
        },
    module: {
          loaders: [
	          {test: /\.css$/, loader: 'style!css'},
            {
              test: /\.js$/,
              exclude: /node_modules/,
              loaders: ['babel-loader'],
            }
	        ]
        }
}
