//app.js

const utils = require('./utils/util.js')

App({
  onLaunch: function () {
    this.createNewWorker();
    wx.getStorage({
      key: 'history',
      success: (res) => {
          this.globalData.history = res.data
      },
      fail: (res) => {
          console.log("get storage failed")
          console.log(res)
          this.globalData.history = []
      }
    })

  },
  // 权限询问
  getRecordAuth: function() {
    wx.getSetting({
      success(res) {
        console.log("succ")
        console.log(res)
        if (!res.authSetting['scope.record']) {
          wx.authorize({
            scope: 'scope.record',
            success() {
                // 用户已经同意小程序使用录音功能，后续调用 wx.startRecord 接口不会弹窗询问
                console.log("succ auth")
            }, fail() {
                console.log("fail auth")
            }
          })
        } else {
          console.log("record has been authed")
        }
      }, fail(res) {
          console.log("fail")
          console.log(res)
      }
    })
  },

  onHide: function () {
    wx.stopBackgroundAudio()
  },
  createNewWorker: function() {
    this.worker = wx.createWorker('workers/request/index.js', {
      useExperimentalWorker: true
    })
    // 监听worker被系统回收事件
    this.worker.onProcessKilled(() => {
      // 重新创建一个worker
      console.log("onProcessKilled");
      createNewWorker()
    })
  },
  globalData: {

    history: [],
  }
})