//app.js
var api = require('/utils/api.js');
var loginUrl = api.getLoginUrl();
App({
  data: {
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo')
  },
  onLaunch: function () {
    //调用API从本地缓存中获取数据
    var logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs);
    //1.静默操作获取用户信息 调用wx.login
    var that = this;
    // 登录
    wx.login({
      success: res => {
        // 发送 res.code 到后台换取 openId, sessionKey, unionId
        wx.request({
          url: loginUrl,
          data: {
            code: res.code
          },
          success: function (res) {
            if (res.data.code == 200) {
              that.data.openId = res.data.data.openid
            }
          },
        })
      }
    })
  },
  globalData: {
    userInfo: null,
    backUserInfo: null
  }
})