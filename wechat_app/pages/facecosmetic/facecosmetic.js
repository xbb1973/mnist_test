var app = getApp();
var api = require('../../utils/api.js');
var facecosmeticiurl = api.getFacecosmeticiurl();
Page({
  data: {
    motto: '腾讯优图',
    images: {},
    img: '',
    remark: "",
    model: 0,
    tempFilePaths: '',
    userInfo: {},
    currentTab:0,
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    facecosmeticJP: [
      { "id": "1", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/1.png", "text": "芭比粉" },
      { "id": "2", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/2.png", "text": "清透" },
      { "id": "3", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/3.png", "text": "烟灰" },
      { "id": "4", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/4.png", "text": "自然" },
      { "id": "5", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/5.png", "text": "樱花粉" },
      { "id": "6", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/6.png", "text": "原宿红" }],
    facecosmeticKO: [
      { "id": "7", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/7.png", "text": "闪亮" },
      { "id": "8", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/8.png", "text": "粉紫" },
      { "id": "9", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/9.png", "text": "粉嫩" },
      { "id": "10", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/10.png", "text": "自然" },
      { "id": "11", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/11.png", "text": "清透" },
      { "id": "12", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/12.png", "text": "大地色" },
      { "id": "13", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/13.png", "text": "玫瑰" }],
    facecosmeticNL: [
      { "id": "14", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/14.png", "text": "自然" },
      { "id": "15", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/15.png", "text": "清透" },
      { "id": "16", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/16.png", "text": "桃粉" },
      { "id": "17", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/17.png", "text": "橘粉" },
      { "id": "18", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/18.png", "text": "春夏" },
      { "id": "19", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/19.png", "text": "秋冬" }],
    facecosmeticTH: [
      { "id": "20", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/20.png", "text": "经典复古" },
      { "id": "21", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/21.png", "text": "性感混血" },
      { "id": "22", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/22.png", "text": "炫彩明眸" },
      { "id": "23", "url": "https://wximage-1251091977.cos.ap-beijing.myqcloud.com/facecosmetic/23.png", "text": "紫色魅惑" }]
  },
  onShareAppMessage: function () {
    return {
      title: '快来看看自己适合什么妆',
      path: '/pages/facecosmetic/facecosmetic',
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
  //点击切换
  clickTab: function (e) {
    var that = this;
    if (this.data.currentTab === e.target.dataset.current) {
      return false;
    } else {
      that.setData({
        currentTab: e.target.dataset.current
      })
    }
  },
  onFaceMerge: function (res) {
    var that = this;
    var modelId = res.currentTarget.dataset.id;
    console.info('modelId='+modelId);
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
        url: facecosmeticiurl,
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
    var openIdKey = app.data.openId;
    this.setData({
      openId: openIdKey
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