const path                      = require('path');
const SpeedMeasurePlugin        = require("speed-measure-webpack-plugin");
const { createProxyMiddleware } = require('http-proxy-middleware');
const webpack                   = require('webpack');
const fs                        = require('fs');

const smp = new SpeedMeasurePlugin();

// ------------------------------------- Step 1 - Define nodejs and python server details
console.log('\n ==================== NODEJS SERVER ==================\n');

const USE_DOCKER = process.env.USE_DOCKER && process.env.USE_DOCKER.toLowerCase() === 'true';
const USE_HTTPS  = process.env.USE_HTTPS  && process.env.USE_HTTPS.toLowerCase() === 'true';

console.log('USE_DOCKER:', USE_DOCKER);
console.log('USE_HTTPS:', USE_HTTPS);

const PORT_NODEJS = 50000
const PORT_PYTHON = 55000
const PORT_DICOM = 8042

let HOST_NODEJS = '0.0.0.0'
let HOST_PYTHON = '0.0.0.0'
let HOST_DICOM  = '0.0.0.0'

if (USE_DOCKER) {
  HOST_PYTHON='backend'
  HOST_DICOM='database'
}


// ------------------------------------- Step 2 - Define SSL certificates for python server
let pythonServerCert;
let pythonServerKey;
let NODEJS_SERVER_OPTIONS;
let SSL_ENABLED = false;

NODEJS_SERVER_OPTIONS = {type: 'http'} // regardless of http/https, you can still access orthanc via node.js frontend
if (USE_HTTPS) {
  try {
    pythonServerCert = fs.readFileSync(path.resolve(__dirname, '_keys', 'hostCert.pem'));
    pythonServerKey  = fs.readFileSync(path.resolve(__dirname, '_keys', 'hostKey.pem'));
    NODEJS_SERVER_OPTIONS = {type: 'https', options: { key: pythonServerKey, cert: pythonServerCert }}
  } catch (error) {
    console.error('\n - [webpack.config.js] Error reading certificate or key file:', error);
    console.error('\n - [webpack.config.js] Using HTTP server instead of HTTPS !!');
    NODEJS_SERVER_OPTIONS = {type: 'http'}
  }
}
if (NODEJS_SERVER_OPTIONS.type === 'https') {
  SSL_ENABLED = true;
}
const serverProtocol = SSL_ENABLED ? 'https' : 'http';

// ------------------------------------- Step 3 - Define webpack configuration
module.exports = smp.wrap({
  entry: './interactive-frontend.js',
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, 'dist'),
  },
  devtool: 'inline-source-map',
  devServer: {
    static: path.join(__dirname, 'dist'),
    compress: true,
    host: HOST_NODEJS, 
    port: PORT_NODEJS,
    // hot: true,
    client: {overlay: false,},
    headers: {
      "Access-Control-Allow-Origin": "*", // not a soln for CORS on orthanc
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",   
      "Cross-Origin-Resource-Policy": "cross-origin",  
    },
    server: NODEJS_SERVER_OPTIONS,
    setupMiddlewares: function(middlewares, devServer) {
      
      // Add middleware for Orthanc server
      const endpointsOrthanc = ['/dicom-web', '/patients', '/studies', '/series', '/instances']; // List your endpoints here
      endpointsOrthanc.forEach((endpointOrthanc) => {
        devServer.app.use(
          endpointOrthanc,
          createProxyMiddleware({
            target: `http://${HOST_DICOM}:${PORT_DICOM}${endpointOrthanc}`,
            changeOrigin: true,
            onProxyRes: function(proxyRes) {
              proxyRes.headers['Cross-Origin-Opener-Policy'] = 'same-origin';
              proxyRes.headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
            },
          })
        );
      });

      // Add middleware for Python server
      const endpointsPython = ['/prepare', '/process', '/serverHealth', '/uploadManualRefinement', '/uploadScrolledSliceIdxs','/closeSession']; // List your endpoints here
      endpointsPython.forEach((endpointPython) => {
        devServer.app.use(
          endpointPython,
          createProxyMiddleware({
            target: `${serverProtocol}://${HOST_PYTHON}:${PORT_PYTHON}${endpointPython}`,
            changeOrigin: true,
            // secure: false, // If you want to accept self-signed certificates
            // ssl: {
            //   cert: pythonServerCert,
            //   key: pythonServerKey, // Private key file
            // },
            // onProxyReq: (proxyReq, req, res) => {
            //   // Additional proxy request configurations if needed
            //   proxyReq.headers['Cross-Origin-Opener-Policy'] = 'same-origin';
            //   proxyReq.headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
            // },
            onProxyRes: function(proxyRes) {
              proxyRes.headers['Cross-Origin-Opener-Policy'] = 'same-origin';
              proxyRes.headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
            },
          })
        );
      });

      return middlewares;
    },
  },
  experiments: {
    asyncWebAssembly: true,
  },
  module: {
    rules: [
      {
        test: /\.wasm$/,
        // use: ['wasm-loader'], //to deal with ERROR in ./node_modules/@icr/polyseg-wasm/dist/ICRPolySeg.wasm 1:0
        // type: 'javascript/auto',
        type: 'asset/resource',
      },
    ],
    unknownContextCritical: false,
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env.NETLIFY': JSON.stringify(process.env.NETLIFY), // to use dicoms from https://d3t6nz73ql33tx.cloudfront.net/dicomweb when on netlify
      'process.env.CONTEXT': JSON.stringify(process.env.CONTEXT),
    }),
  ],
});

// ------------------------------------- Step 4 - Print server details
const os = require('os');
// Function to get the IP address
function getIPAddress() {
  const interfaces = os.networkInterfaces();
  for (let iface in interfaces) {
      for (let alias of interfaces[iface]) {
          if (alias.family === 'IPv4' && !alias.internal) {
              return alias.address;
          }
      }
  }
  return '0.0.0.0';
}

console.log('\n ======================================\n');
console.log(` --> Server running at ${NODEJS_SERVER_OPTIONS.type}://${getIPAddress()}:${PORT_NODEJS}/ (SSL_ENABLED: ${SSL_ENABLED})`);
console.log('   --> [net::ERR_CONNECTION_REFUSED]       Server is inaccessible !!')
console.log('   --> [net::ERR_BLOCKED_BY_CLIENT]        Make sure to remove addBlockers !!')
console.log('   --> [net::ERR_CERT_AUTHORITY_INVALID]   Make sure to allow self-signed certificates !!')
console.log('   --> [net::ERR_CERT_COMMON_NAME_INVALID] Try to set chrome://flags/#allow-insecure-localhost to Enabled !!')
console.log('   --> [net::ERR_NAME_NOT_RESOLVED]        Dont know !!')

console.log('\n ======================================\n');

/**
2025-04-02 15:16:25,374 | INFO | Incoming request: 172.20.0.1:39052 => GET /serverHealth
2025-04-02 15:16:25,387 | INFO | 172.20.0.1:39052 => GET /serverHealth => 200 (200) [0.0s]
2025-04-02 15:16:26,332 | INFO | Incoming request: 172.20.0.4:51144 => GET /serverHealth/
2025-04-02 15:16:26,338 | INFO | 172.20.0.4:51144 => GET /serverHealth/ => 307 (307) [0.0s]
 */