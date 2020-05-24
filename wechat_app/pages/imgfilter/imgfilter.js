var app = getApp();
var api = require('../../utils/api.js');
var vsionimgfilterurl = api.getVsionimgfilterurl();
var ptuimgfilterurl = api.getPtuimgfilterurl();
Page({
  data: {
    motto: '腾讯优图',
    images: {},
    img: '',
    remark: "",
    model: 0,
    tempFilePaths: '',
    userInfo: {},
    currentTab: 0,
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    imgfilterPT: [
      { "id": "1", "url": "https://yourself.com/ptuimgfilter/1.png", "text": "黛紫" },
      { "id": "2", "url": "https://yourself.com/ptuimgfilter/2.png", "text": "岩井" },
      { "id": "3", "url": "https://yourself.com/ptuimgfilter/3.png", "text": "粉嫩" },
      { "id": "4", "url": "https://yourself.com/ptuimgfilter/4.png", "text": "错觉" },
      { "id": "5", "url": "https://yourself.com/ptuimgfilter/5.png", "text": "暖阳" },
      { "id": "6", "url": "https://yourself.com/ptuimgfilter/6.png", "text": "浪漫" },
      { "id": "7", "url": "https://yourself.com/ptuimgfilter/7.gif", "text": "蔷薇" },
      { "id": "8", "url": "https://yourself.com/ptuimgfilter/8.gif", "text": "睡莲" },
      { "id": "9", "url": "https://yourself.com/ptuimgfilter/9.gif", "text": "糖果玫瑰" },
      { "id": "10", "url": "https://yourself.com/ptuimgfilter/10.gif", "text": "新叶" },
      { "id": "11", "url": "https://yourself.com/ptuimgfilter/11.gif", "text": "尤加利" },
      { "id": "12", "url": "https://yourself.com/ptuimgfilter/12.png", "text": "墨" },
      { "id": "13", "url": "https://yourself.com/ptuimgfilter/13.png", "text": "玫瑰初雪" },
      { "id": "14", "url": "https://yourself.com/ptuimgfilter/14.png", "text": "樱桃布丁" },
      { "id": "15", "url": "https://yourself.com/ptuimgfilter/15.png", "text": "白茶" },
      { "id": "16", "url": "https://yourself.com/ptuimgfilter/16.png", "text": "甜薄荷" },
      { "id": "17", "url": "https://yourself.com/ptuimgfilter/17.png", "text": "樱红" },
      { "id": "18", "url": "https://yourself.com/ptuimgfilter/18.png", "text": "圣代" },
      { "id": "19", "url": "https://yourself.com/ptuimgfilter/19.png", "text": "莫斯科" },
      { "id": "20", "url": "https://yourself.com/ptuimgfilter/20.png", "text": "冲绳" },
      { "id": "21", "url": "https://yourself.com/ptuimgfilter/21.png", "text": "粉碧" },
      { "id": "22", "url": "https://yourself.com/ptuimgfilter/22.png", "text": "地中海" },
      { "id": "23", "url": "https://yourself.com/ptuimgfilter/23.png", "text": "首尔" },
      { "id": "24", "url": "https://yourself.com/ptuimgfilter/24.png", "text": "佛罗伦萨" },
      { "id": "25", "url": "https://yourself.com/ptuimgfilter/25.png", "text": "札幌" },
      { "id": "26", "url": "https://yourself.com/ptuimgfilter/26.png", "text": "栀子" },
      { "id": "27", "url": "https://yourself.com/ptuimgfilter/27.png", "text": "东京" },
      { "id": "28", "url": "https://yourself.com/ptuimgfilter/28.png", "text": "昭和" },
      { "id": "29", "url": "https://yourself.com/ptuimgfilter/29.gif", "text": "自然" },
      { "id": "30", "url": "https://yourself.com/ptuimgfilter/30.png", "text": "清逸" },
      { "id": "31", "url": "https://yourself.com/ptuimgfilter/31.png", "text": "染" },
      { "id": "32", "url": "https://yourself.com/ptuimgfilter/32.png", "text": "甜美" }],
    imgfilterVision: [
      { "id": "1", "url": "https://yourself.com/visionimgfilter/1.jpg", "text": "1" },
      { "id": "2", "url": "https://yourself.com/visionimgfilter/2.jpg", "text": "2" },
      { "id": "3", "url": "https://yourself.com/visionimgfilter/3.jpg", "text": "3" },
      { "id": "4", "url": "https://yourself.com/visionimgfilter/4.jpg", "text": "4" },
      { "id": "5", "url": "https://yourself.com/visionimgfilter/5.jpg", "text": "5" },
      { "id": "6", "url": "https://yourself.com/visionimgfilter/6.jpg", "text": "6" },
      { "id": "7", "url": "https://yourself.com/visionimgfilter/7.jpg", "text": "7" },
      { "id": "8", "url": "https://yourself.com/visionimgfilter/8.jpg", "text": "8" },
      { "id": "9", "url": "https://yourself.com/visionimgfilter/9.jpg", "text": "9" },
      { "id": "10", "url": "https://yourself.com/visionimgfilter/10.jpg", "text": "10" },
      { "id": "11", "url": "https://yourself.com/visionimgfilter/11.jpg", "text": "11" },
      { "id": "12", "url": "https://yourself.com/visionimgfilter/12.jpg", "text": "12" },
      { "id": "13", "url": "https://yourself.com/visionimgfilter/13.jpg", "text": "13" },
      { "id": "14", "url": "https://yourself.com/visionimgfilter/14.jpg", "text": "14" },
      { "id": "15", "url": "https://yourself.com/visionimgfilter/15.jpg", "text": "15" },
      { "id": "16", "url": "https://yourself.com/visionimgfilter/16.jpg", "text": "16" },
      { "id": "17", "url": "https://yourself.com/visionimgfilter/17.jpg", "text": "17" },
      { "id": "18", "url": "https://yourself.com/visionimgfilter/18.jpg", "text": "18" },
      { "id": "19", "url": "https://yourself.com/visionimgfilter/19.jpg", "text": "19" },
      { "id": "20", "url": "https://yourself.com/visionimgfilter/20.jpg", "text": "20" },
      { "id": "21", "url": "https://yourself.com/visionimgfilter/21.jpg", "text": "21" },
      { "id": "22", "url": "https://yourself.com/visionimgfilter/22.jpg", "text": "22" },
      { "id": "23", "url": "https://yourself.com/visionimgfilter/23.jpg", "text": "23" },
      { "id": "24", "url": "https://yourself.com/visionimgfilter/24.jpg", "text": "24" },
      { "id": "25", "url": "https://yourself.com/visionimgfilter/25.jpg", "text": "25" },
      { "id": "26", "url": "https://yourself.com/visionimgfilter/26.jpg", "text": "26" },
      { "id": "27", "url": "https://yourself.com/visionimgfilter/27.jpg", "text": "27" },
      { "id": "28", "url": "https://yourself.com/visionimgfilter/28.jpg", "text": "28" },
      { "id": "29", "url": "https://yourself.com/visionimgfilter/29.jpg", "text": "29" },
      { "id": "30", "url": "https://yourself.com/visionimgfilter/30.jpg", "text": "30" },
      { "id": "31", "url": "https://yourself.com/visionimgfilter/31.jpg", "text": "31" },
      { "id": "32", "url": "https://yourself.com/visionimgfilter/32.jpg", "text": "32" },
      { "id": "33", "url": "https://yourself.com/visionimgfilter/33.jpg", "text": "33" },
      { "id": "34", "url": "https://yourself.com/visionimgfilter/34.jpg", "text": "34" },
      { "id": "35", "url": "https://yourself.com/visionimgfilter/35.jpg", "text": "35" },
      { "id": "36", "url": "https://yourself.com/visionimgfilter/36.jpg", "text": "36" },
      { "id": "37", "url": "https://yourself.com/visionimgfilter/37.jpg", "text": "37" },
      { "id": "38", "url": "https://yourself.com/visionimgfilter/38.jpg", "text": "38" },
      { "id": "39", "url": "https://yourself.com/visionimgfilter/39.jpg", "text": "39" },
      { "id": "40", "url": "https://yourself.com/visionimgfilter/40.jpg", "text": "40" },
      { "id": "41", "url": "https://yourself.com/visionimgfilter/41.jpg", "text": "41" },
      { "id": "42", "url": "https://yourself.com/visionimgfilter/42.jpg", "text": "42" },
      { "id": "43", "url": "https://yourself.com/visionimgfilter/43.jpg", "text": "43" },
      { "id": "44", "url": "https://yourself.com/visionimgfilter/44.jpg", "text": "44" },
      { "id": "45", "url": "https://yourself.com/visionimgfilter/45.jpg", "text": "45" },
      { "id": "46", "url": "https://yourself.com/visionimgfilter/46.jpg", "text": "46" },
      { "id": "47", "url": "https://yourself.com/visionimgfilter/47.jpg", "text": "47" },
      { "id": "48", "url": "https://yourself.com/visionimgfilter/48.jpg", "text": "48" },
      { "id": "49", "url": "https://yourself.com/visionimgfilter/49.jpg", "text": "49" },
      { "id": "50", "url": "https://yourself.com/visionimgfilter/50.jpg", "text": "50" },
      { "id": "51", "url": "https://yourself.com/visionimgfilter/51.jpg", "text": "51" },
      { "id": "52", "url": "https://yourself.com/visionimgfilter/52.jpg", "text": "52" },
      { "id": "53", "url": "https://yourself.com/visionimgfilter/53.jpg", "text": "53" },
      { "id": "54", "url": "https://yourself.com/visionimgfilter/54.jpg", "text": "54" },
      { "id": "55", "url": "https://yourself.com/visionimgfilter/55.jpg", "text": "55" },
      { "id": "56", "url": "https://yourself.com/visionimgfilter/56.jpg", "text": "56" },
      { "id": "57", "url": "https://yourself.com/visionimgfilter/57.jpg", "text": "57" },
      { "id": "58", "url": "https://yourself.com/visionimgfilter/58.jpg", "text": "58" },
      { "id": "59", "url": "https://yourself.com/visionimgfilter/59.jpg", "text": "59" },
      { "id": "60", "url": "https://yourself.com/visionimgfilter/60.jpg", "text": "60" },
      { "id": "61", "url": "https://yourself.com/visionimgfilter/61.jpg", "text": "61" },
      { "id": "62", "url": "https://yourself.com/visionimgfilter/62.jpg", "text": "62" },
      { "id": "63", "url": "https://yourself.com/visionimgfilter/63.jpg", "text": "63" },
      { "id": "64", "url": "https://yourself.com/visionimgfilter/64.jpg", "text": "64" },
      { "id": "65", "url": "https://yourself.com/visionimgfilter/65.jpg", "text": "65" }]
  },
  onShareAppMessage: function () {
    return {
      title: '给图片加个滤镜',
      path: '/pages/imgfilter/imgfilter',
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
    var url = '';
    if (this.data.currentTab=='0'){
      url = ptuimgfilterurl;
    }else{
      url = vsionimgfilterurl;
    }
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
        url: url,
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