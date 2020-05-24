var app = getApp();
var api = require('../../utils/api.js');
var qcUrl = api.getQcUrl();
Page({
  data: {
    motto: '速算识别',
    images: {},
    info: "",
    classifyCnt: "",
    confidence: "",
    label: "",
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    result: [],
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    remark: ""
  },
  onShareAppMessage: function () {
    return {
      title: '速算智能批改识别',
      path: '/pages/quickcalculation/quickcalculation',
      success: function (res) {
        if (res.errMsg == 'shareAppMessage:ok') {
          wx.showToast({
            title: '分享成功',
            icon: 'success',
            duration: 500
          });
        }
      },
      fail: function (res) {
        if (res.errMsg == 'shareAppMessage:fail cancel') {
          wx.showToast({
            title: '分享取消',
            icon: 'loading',
            duration: 500
          })
        }
      }
    }
  },
  clear: function (event) {
    console.info(event);
    wx.clearStorage();
  },
  //事件处理函数
  bindViewTap: function () {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },
  uploads: function () {
    var that = this
    wx.chooseImage({
      count: 1, // 默认9
      sizeType: ['compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success: function (res) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        //console.log( res )
        that.setData({
          img: res.tempFilePaths[0],
          info: "",
          classifyCnt: "",
          confidence: "",
          label: "",
        }),
          wx.showLoading({
            title: "努力分析中...",
            mask: true
          }),
          wx.uploadFile({
          url: qcUrl,
            filePath: res.tempFilePaths[0],
            header: {
              'content-type': 'multipart/form-data'
            },
            name: 'file',
            formData: {
              'openId': that.data.openId,
              'nickName': that.data.nickName
            },
            success: function (res) {
              wx.hideLoading();
              var data = res.data;
              var str = JSON.parse(data);
              if (str.code == "0") {
                let dataList = str.results;
                that.setData({
                  result: dataList
                })
              } else if (str.code == "1") {
                that.setData({
                  info: str.msg
                })
              } else {
                that.setData({
                  info: "Sorry 小程序远走高飞了",
                })
              }
            },
            fail: function (res) {
              wx.hideLoading();
              console.log(res);
              that.setData({
                info: '小程序离家出走了稍后再试',
              })
            }
          })
      }
    })
  },
  onLoad: function () {
    var openIdKey = app.data.openId;
    this.setData({
      openId: openIdKey
    })
  }
});