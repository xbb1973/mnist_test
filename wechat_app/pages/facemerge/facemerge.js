var app = getApp();
var api = require('../../utils/api.js');
var facemergeUrl = api.getFacemergeUrl();
Page({
  data: {
    motto: '腾讯优图',
    images: {},
    img:'',
    remark: "",
    model:0,
    tempFilePaths:'',
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    facemergeData: [
      { "id": "1", "url": "https://替换成自己的图片路径/facemerge/1.png", "text": "奇迹" },
      { "id": "2", "url": "https://替换成自己的图片路径/facemerge/2.png", "text": "压岁钱" },
      { "id": "3", "url": "https://替换成自己的图片路径/facemerge/3.png", "text": "范蠡" },
      { "id": "4", "url": "https://替换成自己的图片路径/facemerge/4.png", "text": "李白" },
      { "id": "5", "url": "https://替换成自己的图片路径/facemerge/5.png", "text": "孙尚香" },
      { "id": "6", "url": "https://替换成自己的图片路径/facemerge/6.png", "text": "花无缺" },
      { "id": "7", "url": "https://替换成自己的图片路径/facemerge/7.png", "text": "西施" },
      { "id": "8", "url": "https://替换成自己的图片路径/facemerge/8.png", "text": "杨玉环" },
      { "id": "9", "url": "https://替换成自己的图片路径/facemerge/9.png", "text": "白浅" },
      { "id": "10", "url": "https://替换成自己的图片路径/facemerge/10.png", "text": "凤九" },
      { "id": "11", "url": "https://替换成自己的图片路径/facemerge/11.png", "text": "夜华" },
      { "id": "12", "url": "https://替换成自己的图片路径/facemerge/12.png", "text": "年年有余" },
      { "id": "13", "url": "https://替换成自己的图片路径/facemerge/13.png", "text": "新年萌萌哒" },
      { "id": "14", "url": "https://替换成自己的图片路径/facemerge/14.png", "text": "王者荣耀荆轲" },
      { "id": "15", "url": "https://替换成自己的图片路径/facemerge/15.png", "text": "王者荣耀李白" },
      { "id": "16", "url": "https://替换成自己的图片路径/facemerge/16.png", "text": "王者荣耀哪吒" },
      { "id": "17", "url": "https://替换成自己的图片路径/facemerge/17.png", "text": "王者荣耀王昭君" },
      { "id": "18", "url": "https://替换成自己的图片路径/facemerge/18.png", "text": "王者荣耀甄姬" },
      { "id": "19", "url": "https://替换成自己的图片路径/facemerge/19.png", "text": "王者荣耀诸葛亮" },
      { "id": "20", "url": "https://替换成自己的图片路径/facemerge/20.png", "text": "赵灵儿" },
      { "id": "21", "url": "https://替换成自己的图片路径/facemerge/21.png", "text": " 李逍遥" },
      { "id": "22", "url": "https://替换成自己的图片路径/facemerge/22.png", "text": "爆炸头" },
      { "id": "23", "url": "https://替换成自己的图片路径/facemerge/23.png", "text": "村姑" },
      { "id": "24", "url": "https://替换成自己的图片路径/facemerge/24.png", "text": "光头" },
      { "id": "25", "url": "https://替换成自己的图片路径/facemerge/25.png", "text": "呵呵哒" },
      { "id": "26", "url": "https://替换成自己的图片路径/facemerge/26.png", "text": "肌肉" },
      { "id": "27", "url": "https://替换成自己的图片路径/facemerge/27.png", "text": "肉山" },
      { "id": "28", "url": "https://替换成自己的图片路径/facemerge/28.png", "text": "机智" },
      { "id": "29", "url": "https://替换成自己的图片路径/facemerge/29.png", "text": "1927年军装(男)" },
      { "id": "30", "url": "https://替换成自己的图片路径/facemerge/30.png", "text": "1927年军装(女)" },
      { "id": "31", "url": "https://替换成自己的图片路径/facemerge/31.png", "text": "1929年军装(男)" },
      { "id": "32", "url": "https://替换成自己的图片路径/facemerge/32.png", "text": "1929年军装(女)" },
      { "id": "33", "url": "https://替换成自己的图片路径/facemerge/33.png", "text": "1937年军装(男)" },
      { "id": "34", "url": "https://替换成自己的图片路径/facemerge/34.png", "text": "1937年军装(女)" },
      { "id": "35", "url": "https://替换成自己的图片路径/facemerge/35.png", "text": "1948年军装(男)" },
      { "id": "36", "url": "https://替换成自己的图片路径/facemerge/36.png", "text": "1948年军装(女)" },
      { "id": "37", "url": "https://替换成自己的图片路径/facemerge/37.png", "text": "1950年军装(男)" },
      { "id": "38", "url": "https://替换成自己的图片路径/facemerge/38.png", "text": "1950年军装(女)" },
      { "id": "39", "url": "https://替换成自己的图片路径/facemerge/39.png", "text": "1955年军装(男)" },
      { "id": "40", "url": "https://替换成自己的图片路径/facemerge/40.png", "text": "1955年军装(女)" },
      { "id": "41", "url": "https://替换成自己的图片路径/facemerge/41.png", "text": "1965年军装(男)" },
      { "id": "42", "url": "https://替换成自己的图片路径/facemerge/42.png", "text": "1965年军装(女)" },
      { "id": "43", "url": "https://替换成自己的图片路径/facemerge/43.png", "text": "1985年军装(男)" },
      { "id": "44", "url": "https://替换成自己的图片路径/facemerge/44.png", "text": "1985年军装(女)" },
      { "id": "45", "url": "https://替换成自己的图片路径/facemerge/45.png", "text": "1987年军装(男)" },
      { "id": "46", "url": "https://替换成自己的图片路径/facemerge/46.png", "text": "1987年军装(女)" },
      { "id": "47", "url": "https://替换成自己的图片路径/facemerge/47.png", "text": "1999年军装(男)" },
      { "id": "48", "url": "https://替换成自己的图片路径/facemerge/48.png", "text": "1999年军装(女)" },
      { "id": "49", "url": "https://替换成自己的图片路径/facemerge/49.png", "text": "2007年军装(男)" },
      { "id": "50", "url": "https://替换成自己的图片路径/facemerge/50.png", "text": "2007年军装(女)" }]
  },
  onShareAppMessage: function () {
    return {
      title: '快来和我一起变脸吧',
      path: '/pages/facemerge/facemerge',
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
  onFaceMerge:function(res){
    var that = this;
    console.log(that);
    var modelId = res.currentTarget.dataset.id;
    if (!that.data.tempFilePaths){
      wx.showToast({
        title: '快选择图片吧',
        icon:'none',
        mask:true,
        duration:1000
      })
    } else {
      that.setData({
        model:modelId
      })
      wx.showToast({
        title: '智能处理中',
        icon:'loading',
        mask:true,
        duration:20000
      })
      wx.uploadFile({
        url: facemergeUrl,
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
          if(str.ret==0){
            that.setData({
              img: 'data:image/png;base64,' + str.data.image,
            })
          } else if (str.ret==16402){
            wx.showModal({
              title: '温馨提示',
              content: '图片中不包含人脸哦',
              showCancel: false   
            })
          }else{
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
            showCancel:false
          })
        }
      })
    }
  },
  chooseImage:function(){
    var that = this;
    wx.chooseImage({
	   count:1,
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'], 
      success: function(res) {
          console.log(res);
          if(res.tempFilePaths[0].size>500*1024){
            wx.showToast({
              title: '图片文件过大哦',
              icon:'none',
              mask:true,
              duration:1500
            })
          } else {
             that.setData({
               img:res.tempFilePaths[0],
               tempFilePaths:res.tempFilePaths
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
  preview:function(e){
    var that = this;
    wx.previewImage({
      urls: [that.data.img],
      current:that.data.img
    })
  }
});