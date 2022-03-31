const vad = require("./vad/webrtc_vad.js");
const app = getApp();

//获取全局录音器
const recorder = wx.getRecorderManager()

let recordOptions = {
  duration: 600000,//录音的时长单位ms
  sampleRate: 16000,//采样率
  numberOfChannels: 1,//录音通道数
  encodeBitRate: 48000,//编码码率
  format: 'pcm',//音频格式，默认是16比特一个采样,
  frameSize: 5, //单位KB.  160ms一个帧， 一个帧的字节数是： 160ms *  16000  * 2bytes / 1000ms = 5120bytes。   5120/1024=5
  }
  

Page({
  data: {
    recording: false,  // 正在录音
    recordStatus: 0,   // 状态： 0 - 录音中 1- 翻译中 2 - 翻译完成/二次翻译
  },

  initRecord: function() {

    console.log('init record')

    //开始录音
    recorder.onStart(() => {
      this.isStart = true;
      console.log('onStart:');
    })
    
    //结束事件
    recorder.onStop((res) => {
      this.isStart = false;
      console.log('onStop:path=',res.tempFilePath,' duration=',res.duration,' size=',res.fileSize)
    })

    recorder.onFrameRecorded((res) => {
      vad.push(res.frameBuffer);
    })


    // 识别错误事件
    recorder.onError((err) => {
      console.log('onError: ', err)
    })
  },

  onLoad: function () {
    this.isTouchEnd = true;
    this.initRecord();
    app.getRecordAuth();
    vad.init();

  },
  onReady: function() {
   
  },
  onTouchStart: function(e){
    if(!this.isTouchEnd) return;
    this.isTouchEnd = false;
    this.recording = true;

    if(!this.isStart)
      recorder.start(recordOptions);
  },
  onTouchEnd: function(){
    if(this.isTouchEnd) return;
    this.isTouchEnd = true;
    recorder.stop();
  },
  concatenate: function(...arrays) {
    let totalLen = 0;
    for (let arr of arrays)
        totalLen += arr.byteLength;

    let res = new Uint8Array(totalLen)
    let offset = 0

    for (let arr of arrays) {
        let uint8Arr = new Uint8Array(arr)
        res.set(uint8Arr, offset)
        offset += arr.byteLength
    }
    return res.buffer
  }
})