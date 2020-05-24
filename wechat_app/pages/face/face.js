var app = getApp();
var api = require('../../utils/api.js');
var faceUrl = api.getFaceUrl();
Page({
  data: {
    motto: '检测人脸',
    images: {},
    info: "",
    age: "",
    beauty: "",
    expression:"",
    faceShape:"",
    gender:"",
    glasses:"",
    raceType:"",
    remark: "等待1-2秒查看分析",
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId:"",
    nickName:"",
    canIUse: wx.canIUse('button.open-type.getUserInfo')
  },
  onShareAppMessage: function () {
    return {
      title: '颜值分析小程序',
      path: '/pages/face/face',
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
    var that = this;
    console.log(that);
    wx.chooseImage({
      count: 1, // 默认9
      sizeType: ['compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success: function (res) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        //console.log( res )
        that.setData({
          img: res.tempFilePaths[0],
          age: "",
          beauty: "",
          expression: "",
          faceShape: "",
          gender: "",
          glasses: "",
          raceType: "",
          info:""
        }),
          wx.showLoading({
            title: "魅力年龄分析中...",
            mask: true
          }),
        wx.uploadFile({
          url: faceUrl,
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
            var data = res.data;
            var str = JSON.parse(data);
            if (str.code=="0") {
              that.setData({
                age: str.age,
                beauty:str.beauty.substring(0,5),
                expression: str.expression,
                faceShape: str.faceShape,
                gender: str.gender,
                glasses: str.glasses,
                raceType: str.raceType
              })
            } else if (str.code == "1"){
              that.setData({
                info: 'Sorry '+str.msg
              })
            } else {
              that.setData({
                info: 'Sorry 小程序远走高飞了'
              })
            }
            wx.hideLoading();
          },
          fail: function (res) {
            wx.hideLoading();
            wx.showModal({
              title: '温馨提示',
              content: 'Sorry 小程序离家出走了',
              showCancel: false
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