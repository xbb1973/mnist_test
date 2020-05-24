var app = getApp();
var api = require('../../utils/api.js');
var facedecorationurl = api.getFacedecorationurl();
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
      { "id": "1", "url": "https://yourself.com/facedecoration/1.jpg", "text": "埃及妆" },
      { "id": "2", "url": "https://yourself.com/facedecoration/2.jpg", "text": "巴西土著妆" },
      { "id": "3", "url": "https://yourself.com/facedecoration/3.jpg", "text": "灰姑娘妆" },
      { "id": "4", "url": "https://yourself.com/facedecoration/4.jpg", "text": "恶魔妆" },
      { "id": "5", "url": "https://yourself.com/facedecoration/5.jpg", "text": "武媚娘妆" },
      { "id": "6", "url": "https://yourself.com/facedecoration/6.jpg", "text": "星光薰衣草" },
      { "id": "7", "url": "https://yourself.com/facedecoration/7.jpg", "text": "花千骨" },
      { "id": "8", "url": "https://yourself.com/facedecoration/8.jpg", "text": "僵尸妆" },
      { "id": "9", "url": "https://yourself.com/facedecoration/9.jpg", "text": "爱国妆" },
      { "id": "10", "url": "https://yourself.com/facedecoration/10.jpg", "text": "小胡子妆" },
      { "id": "11", "url": "https://yourself.com/facedecoration/11.jpg", "text": "美羊羊妆" },
      { "id": "12", "url": "https://yourself.com/facedecoration/12.jpg", "text": "火影鸣人妆" },
      { "id": "13", "url": "https://yourself.com/facedecoration/13.jpg", "text": "刀马旦妆" },
      { "id": "14", "url": "https://yourself.com/facedecoration/14.jpg", "text": "泡泡妆" },
      { "id": "15", "url": "https://yourself.com/facedecoration/15.jpg", "text": "桃花妆" },
      { "id": "16", "url": "https://yourself.com/facedecoration/16.jpg", "text": "女皇妆" },
      { "id": "17", "url": "https://yourself.com/facedecoration/17.jpg", "text": "权志龙" },
      { "id": "18", "url": "https://yourself.com/facedecoration/18.jpg", "text": "撩妹妆" },
      { "id": "19", "url": "https://yourself.com/facedecoration/19.jpg", "text": "印第安妆" },
      { "id": "20", "url": "https://yourself.com/facedecoration/20.jpg", "text": "印度妆" },
      { "id": "21", "url": "https://yourself.com/facedecoration/21.jpg", "text": "萌兔妆" },
      { "id": "22", "url": "https://yourself.com/facedecoration/22.jpg", "text": "大圣妆" }]
  },
  onShareAppMessage: function () {
    return {
      title: '一键变妆',
      path: '/pages/facedecoration/facedecoration',
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
        url: facedecorationurl,
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