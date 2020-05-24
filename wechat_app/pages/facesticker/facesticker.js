var app = getApp();
var api = require('../../utils/api.js');
var facestickerurl = api.getFacestickerurl();
Page({
  data: {
    motto: '腾讯优图',
    images: {},
    img: '',
    remark: "",
    model: 0,
    tempFilePaths: '',
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    facemergeData: [
      { "id": "1", "url": "https://yourself.com/facesticker/1.png", "text": "NewDay" },
      { "id": "2", "url": "https://yourself.com/facesticker/2.png", "text": "欢乐球吃球1" },
      { "id": "3", "url": "https://yourself.com/facesticker/3.jpg", "text": "Bonvoyage" },
      { "id": "4", "url": "https://yourself.com/facesticker/4.png", "text": "Enjoy" },
      { "id": "5", "url": "https://yourself.com/facesticker/5.png", "text": "ChickenSpring" },
      { "id": "6", "url": "https://yourself.com/facesticker/6.png", "text": "ChristmasShow" },
      { "id": "7", "url": "https://yourself.com/facesticker/7.png", "text": "ChristmasSnow" },
      { "id": "8", "url": "https://yourself.com/facesticker/8.jpg", "text": "CircleCat" },
      { "id": "9", "url": "https://yourself.com/facesticker/9.jpg", "text": "CircleMouse" },
      { "id": "10", "url": "https://yourself.com/facesticker/10.jpg", "text": "CirclePig" },
      { "id": "11", "url": "https://yourself.com/facesticker/11.png", "text": "Comicmn" },
      { "id": "12", "url": "https://yourself.com/facesticker/12.jpg", "text": "CuteBaby" },
      { "id": "13", "url": "https://yourself.com/facesticker/13.jpg", "text": "Envolope" },
      { "id": "14", "url": "https://yourself.com/facesticker/14.jpg", "text": "Fairytale" },
      { "id": "15", "url": "https://yourself.com/facesticker/15.jpg", "text": "GoodNight" },
      { "id": "16", "url": "https://yourself.com/facesticker/16.jpg", "text": "HalloweenNight" },
      { "id": "17", "url": "https://yourself.com/facesticker/17.jpg", "text": "1LovelyDay" },
      { "id": "18", "url": "https://yourself.com/facesticker/18.png", "text": "Newyear2017" },
      { "id": "19", "url": "https://yourself.com/facesticker/19.png", "text": "PinkSunny" },
      { "id": "20", "url": "https://yourself.com/facesticker/20.jpg", "text": "KIRAKIRA" },
      { "id": "21", "url": "https://yourself.com/facesticker/21.jpg", "text": "欢乐球吃球2" },
      { "id": "22", "url": "https://yourself.com/facesticker/22.png", "text": "SnowWhite" },
      { "id": "23", "url": "https://yourself.com/facesticker/23.png", "text": "SuperStar" },
      { "id": "24", "url": "https://yourself.com/facesticker/24.png", "text": "WonderfulWork" },
      { "id": "25", "url": "https://yourself.com/facesticker/25.png", "text": "Cold" },
      { "id": "26", "url": "https://yourself.com/facesticker/26.png", "text": "狼人杀守卫" },
      { "id": "27", "url": "https://yourself.com/facesticker/27.png", "text": "狼人杀猎人" },
      { "id": "28", "url": "https://yourself.com/facesticker/28.png", "text": "狼人杀预言家" },
      { "id": "29", "url": "https://yourself.com/facesticker/29.png", "text": "狼人杀村民" },
      { "id": "30", "url": "https://yourself.com/facesticker/30.png", "text": "狼人杀女巫" },
      { "id": "31", "url": "https://yourself.com/facesticker/31.png", "text": "狼人杀狼人" }]
  },
  onShareAppMessage: function () {
    return {
      title: '来选个大头贴吧',
      path: '/pages/facesticker/facesticker',
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
  onFaceMerge: function (res) {
    var that = this;
    console.log(that);
    var modelId = res.currentTarget.dataset.id;
    if (!that.data.tempFilePaths) {
      wx.showToast({
        title: '快选择图片吧',
        icon: 'none',
        mask: true,
        duration: 1000
      })
    } else {
      that.setData({
        model: modelId
      })
      wx.showToast({
        title: '智能处理中',
        icon: 'loading',
        mask: true,
        duration: 20000
      })
      wx.uploadFile({
        url: facestickerurl,
        filePath: that.data.tempFilePaths[0],
        header: {
          'content-type': 'multipart/form-data'
        },
        name: 'file',
        formData: {
          model: modelId
        },
        success: function (res) {
          var data = res.data;
          var str = JSON.parse(data);
          console.log(str.ret);
          if (str.ret == 0) {
            that.setData({
              img: 'data:image/png;base64,' + str.data.image,
            })
          } else if (str.ret == 16402) {
            wx.showModal({
              title: '温馨提示',
              content: '图片中不包含人脸哦',
              showCancel: false
            })
          } else {
            wx.showModal({
              title: '温馨提示',
              content: '服务器远走高飞了',
              showCancel: false
            })
          }
          wx.hideToast();
        },
        fail: function (res) {
          wx.hideToast();
          wx.hideLoading();
          wx.showModal({
            title: '上传失败',
            content: '服务器远走高飞了',
            showCancel: false
          })
        }
      })
    }
  },
  chooseImage: function () {
    var that = this;
    wx.chooseImage({
      count: 1,
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: function (res) {
        console.log(res);
        if (res.tempFiles[0].size > 500 * 1024) {
          wx.showToast({
            title: '图片文件过大哦',
            icon: 'none',
            mask: true,
            duration: 1500
          })
        } else {
          that.setData({
            img: res.tempFilePaths[0],
            tempFilePaths: res.tempFilePaths
          })
        }
      },
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
    wx.previewImage({
      urls: [that.data.img],
      current: that.data.img
    })
  }
});