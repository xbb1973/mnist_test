var app = getApp();
var api = require('../../utils/api.js');
var faceageuploadUrl = api.getFaceAgeUploadUrl();
var faceagecoresUrl = api.getFaceAgeCoresUrl();
Page({
  data: { 
    motto: '我们两个相似吗？',
    images: {},
    remark: "",
    score:"",
    scores: "",
    info:"",
    imgA:"",
    imgB:"",
    tempFilePathsA:"",
    tempFilePathsB:"",
    imgNameA:"",
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    imgNameB: ""
  },
  onShareAppMessage: function () {
    return {
      title: '快来看我跟你有几分相似',
      path: '/pages/facedetectcrossageface/facedetectcrossageface',
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
  chooseImageA: function () {
    var that = this;
    wx.chooseImage({
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: function (res) {
        console.log(res);
        if (res.tempFiles[0].size > 1024 * 1024) {
          wx.showToast({
            title: '图片文件过大哦',
            image: '../../image/big.png',
            mask: true,
            duration: 1500
          })
        } else {
          that.setData({
            imgA: res.tempFilePaths[0],
            tempFilePathsA: res.tempFilePaths,
            score: "",
            scores: "",
            info: "",
          })
          var uploadA = wx.uploadFile({
            url: faceageuploadUrl,
            filePath: that.data.tempFilePathsA[0],
            header: {
              'content-type': 'multipart/form-data'
            },
            name: 'file',
            success: function (res) {
              var data = res.data;
              console.info(res);
              var str = JSON.parse(data);
              that.setData({
                imgNameA: str.fileName,
                score: "",
                scores: "",
                info: "",
              })
            },
            fail: function (res) {
              wx.showModal({
                title: '上传失败',
                content: '服务器远走高飞了',
                showCancel: false
              })
            }
          })
          uploadA.onProgressUpdate((res) => { 
            var progress = res.progress;
            if (progress==100){
              wx.hideToast();
                wx.showToast({
                title: '上传完成',
                icon: 'success',
                mask: true,
                duration: 1500
              })
            }else{
              wx.showToast({
                title: '正在上传' + progress + '%',
                icon: 'loading',
                mask: true
              })
            }
          })
        }
      },
    })
  },
  chooseImageB: function () {
    var that = this;
    wx.chooseImage({
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: function (res) {
        console.log(res);
        console.log(res.tempFiles[0].size);
        if (res.tempFiles[0].size > 1024 * 1024) {
          wx.showToast({
            title: '图片文件过大哦',
            image:'../../image/big.png',
            mask: true,
            duration: 1500
          })
        } else {
          that.setData({
            imgB: res.tempFilePaths[0],
            tempFilePathsB:res.tempFilePaths,
            score: "",
            scores: "",
            info: "",
          })
         var uploadB= wx.uploadFile({
           url: faceageuploadUrl,
            filePath: that.data.tempFilePathsB[0],
            header: {
              'content-type': 'multipart/form-data'
            },
            name: 'file',
            success: function (res) {
              var data = res.data;
              console.log(res);
              var str = JSON.parse(data);
              that.setData({
                imgNameB: str.fileName,
                score: "",
                scores: "",
                info: "",
              })
              wx.hideToast();
            },
            fail: function (res) {
              wx.showModal({
                title: '上传失败',
                content: '服务器远走高飞了',
                showCancel: false
              })
            }
          })
         uploadB.onProgressUpdate((res) => {
           var progress = res.progress;
           if (progress == 100) {
             wx.hideToast();
             wx.showToast({
               title: '上传完成',
               icon: 'success',
               mask: true,
               duration: 1500
             })
           } else {
             wx.showToast({
               title: '正在上传' + progress+'%',
               icon: 'loading',
               mask: true
             })
           }
         })
        }
      },
    })
  },
  onFaceAge: function (res) {
    var that = this;
    console.log(that);
    if (!that.data.tempFilePathsA || !that.data.tempFilePathsB) {
      wx.showToast({
        title: '请选择图片哦',
        icon: 'none',
        mask: true,
        duration: 1000
      })
    } else {
      wx.showToast({
        title: '结果马上就来...',
        icon: 'loading',
        mask: true,
        duration: 20000
      })
      wx.request({
       url:faceagecoresUrl,
        data:{
          imgNameA: that.data.imgNameA,
          imgNameB: that.data.imgNameB,
          'openId': that.data.openId,
          'nickName': that.data.nickName
        },
        success: function (res) {
          var data = res.data;
          var str = data;
          if (str.ret==0){
            that.setData({
              score: str.data.degree,
              info:str.data.info,
              scores:str.data.scores

            })
          }else if(str.ret==16402){
            wx.showModal({
              title: '温馨提示',
              content: '图片中不包含人脸哦',
              showCancel: false
            })
          }else{
            wx.showModal({
              title: '温馨提示',
              content: str.msg,
              showCancel: false
            })
          }
          wx.hideToast();
        },
        fail:function(res){
          wx.hideToast();
          console.log('============'+res.data)
        }
      })
    }
  },
  chooseImage: function () {
    var that = this;
    wx.chooseImage({
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: function (res) {
        console.log(res);
        if (res.tempFilePaths[0].size > 500 * 1024) {
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
  }
});