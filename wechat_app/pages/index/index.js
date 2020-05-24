//index.js
//获取应用实例
var app = getApp()
Page({
  data: {
    motto: '无需安装 触手可及 用完即走 无需卸载',
    userInfo: {}
  },
  //事件处理函数
  bindViewTap: function() {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },
  sample:function(){
    console.log('input')
      wx.navigateTo({
      url: '../sample/sample',
      fail:function(res){
        console.info(res)
      }
    })
  },
  onLoad: function () {
    var getAppWXUserInfo = app.globalData.userInfo;
    this.setData({
      userInfo: getAppWXUserInfo,
      hasUserInfo: true,
      openId: getAppWXUserInfo.openId,
      nickName: getAppWXUserInfo.nickName,
    })
  }
})
