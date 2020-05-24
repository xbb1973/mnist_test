var app = getApp();
var api = require('../../utils/api.js');
var charmUrl = api.getCharmUrl();
Page( {
  data: {
    motto: '腾讯优图',
    images: {},
    info:"",
    age:"",
    beauty:"",
    expression:"",
    glasses:"",
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    remark:""
  },
    onShareAppMessage: function () {
    return {
      title: '魅力值小程序',
      path: '/pages/chram/chram',
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
  bindViewTap: function() {
    wx.navigateTo( {
      url: '../logs/logs'
    })
  },
  uploads: function() {
    var that =this
    wx.chooseImage( {
      count: 1, // 默认9
      sizeType: ['compressed' ], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: [ 'album', 'camera' ], // 可以指定来源是相册还是相机，默认二者都有
      success: function( res ) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        that.setData({
          img: res.tempFilePaths[0],
          age: "",
          beauty: "",
          expression: "",
          glasses: "",
        }),
          wx.showLoading({
          title: "魅力年龄分析中...",
          mask: true
          }),
        wx.uploadFile( {
          url: charmUrl,
          filePath: res.tempFilePaths[ 0 ],
          header:{
             'content-type':'multipart/form-data'
          },
          name: 'file',
          formData: {
            'openId': that.data.openId,
            'nickName': that.data.nickName
          },
          success: function(res) {
            wx.hideLoading();
            var data = res.data;
            var str=JSON.parse(data);
            console.log(str);
            if (str.code==0) {
              that.setData({
                age: str.age,
                beauty: str.beauty,
                expression:str.expression,
                glasses:str.glasses
              })
            } else if (str.code == "1") {
              that.setData({
                info: 'Sorry ' + str.msg
              })
            }else {
              that.setData({
                info: 'Sorry 小程序远走高飞了'
              })
            }
          },
          fail:function(res){
            wx.hideLoading();
            console.log( res )
            that.setData({
              info: '小程序离家出走了稍后再试'
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