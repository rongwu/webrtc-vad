const utils = require('./utils')

console.log('hello worker');
var intervalTime;

worker.postMessage({
  msg: 'hello from worker: ' + utils.test(),
  buffer: utils.str2ab('hello arrayBuffer from worker')
})

worker.onMessage((msg) => {
  
})