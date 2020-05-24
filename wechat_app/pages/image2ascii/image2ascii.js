var app = getApp();
var api = require('../../utils/api.js');
var image2asciiurl = api.getImage2asciiurl();
Page({
  data: {
    motto: '图片转字符图片',
    images: {},
    info: "",
    img: '',
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    remark:"原图宽高不要过大哦，目前base64没有压缩，过大小程序会解析失败",
    canIUse: wx.canIUse('button.open-type.getUserInfo')
  },
  onShareAppMessage: function () {
    return {
      title: '图片转字符图片',
      path: '/pages/image2ascii/image2ascii',
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
        if (res.tempFiles[0].size>1024*1024) {
          wx.showToast({
            title: '图片文件过大哦',
            icon: 'none',
            mask: true,
            duration: 1500
          })
        } else {
          that.setData({
            img: res.tempFilePaths[0],
            info: "",
          }),
          wx.showLoading({
            title: "努力转换中...",
            mask: true
          }),
          wx.uploadFile({
            url: image2asciiurl,
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
              console.log(str);
              if (str.code == 0) {
                console.log(str.code == 0);
                that.setData({
                  img:'data:image/jpg;base64,' + str.image
                })
              } else if (str.code == "1") {
                that.setData({
                  info: 'Sorry ' + str.msg
                })
              } else {
                that.setData({
                  info: 'Sorry 小程序远走高飞了'
                })
              }
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
  },
  /**
   * 点击查看图片，可以进行保存
   */
  preview: function (e) {
    var that = this;
    if (that.data.img==""){
      wx.showToast({
        title: '快选择图片吧',
        icon: 'none',
        mask: true,
        duration: 1000
      })
    }else{
      wx.previewImage({
        urls: [that.data.img],
        current: that.data.img
      })
    }
  }
});