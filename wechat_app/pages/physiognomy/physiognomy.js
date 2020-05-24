var app = getApp();
var api = require('../../utils/api.js');
var physiognomyUrl = api.getPhysiognomyUrl();
Page({
  data: {
    motto: '面相分析',
    images: {},
    info: "",
    icrName: "",
    score: "",
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    baikeUrl: "",
    imageUrl: "",
    description: "",
    remark: "",
    physiognomy:null
  },
  onShareAppMessage: function () {
    return {
      title: '面相分析小程序',
      path: '/pages/physiognomy/physiognomy',
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
          icrName: "",
          score: "",
          baikeUrl: "",
          imageUrl: "",
          description: "",
          physiognomy: null
        }),
          wx.showLoading({
            title: "努力分析中...",
            mask: true
          }),
          wx.uploadFile({
            url: physiognomyUrl,
            filePath: res.tempFilePaths[0],
            header: {
              'content-type': 'multipart/form-data'
            },
            name: 'file',
            formData: {
              'openId': that.data.openId
            },
            success: function (res) {
              wx.hideLoading();
              var data = res.data;
              var str = JSON.parse(data);
              if (str.recognitionResult.status == 0) {
                that.setData({
                  physiognomy: str.recognitionResult
                })
                console.log(that.data.physiognomy.length)
              } else if (str.recognitionResult.status == 1) {
                that.setData({
                  info: "未检测到人脸"
                })
              } else if (str.recognitionResult.status == 2) {
                that.setData({
                  info: "请上传单人照片"
                })
              } else if (str.recognitionResult.status == 3) {
                that.setData({
                  info: "人脸不完整"
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