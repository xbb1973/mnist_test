// pages/someinto/someinto.js
var app = getApp();
var api = require('../../utils/api.js');
var skinUrl = api.getFaceskinurl();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    uploadimgs: '',
    showimages: '0',
    imageSrc: 'https://heiyunke.com/789.jpg',
    width: "130px",
    height: "130px",
    percent: "75",
    count: 0,
    countTimer: null,
    setdatas: [{
      image: "../../image/stain.png",
      name: "色斑",
      percent: "0"
    }, {
      image: "../../image/acne.png",
      name: "青春痘",
      percent: "0"
    }, {
      image: "../../image/dark_circle.png",
      name: "黑眼圈",
      percent: "0"
    }]
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    var openIdKey = app.data.openId;
    this.setData({
      openId: openIdKey
    })
    let that = this;
    that.countInterval(that.data.percent);
    that.drawProgressbg();

  },
  previewImage: function () {

    wx.previewImage({
      urls: [this.data.imageSrc]
    });
  },

  drawProgressbg: function () {
    var t = wx.createCanvasContext("canvasProgressbg", this);
    t.setLineWidth(8),
      t.setStrokeStyle("#e5e5e5"),
      t.setLineCap("round"),
      t.beginPath(),
      t.arc(65, 65, 61, 0, 2 * Math.PI, false),
      t.stroke(),
      t.draw();
  },
  countInterval: function (percent) {
    var that = this;
    this.countTimer = setInterval(function () {
      if (that.data.count <= percent) {
        that.drawCircle(that.data.count / 50)
        that.data.count = that.data.count + .5;
        that.setData({
          percent: parseInt(that.data.count)
        })
      } else {
        that.drawCircle(percent / 50)
        that.setData({
          percent: parseInt(percent)
        })
        clearInterval(that.countTimer)

      }
    }, 4);
  },
  drawCircle: function (t) {
    var e = wx.createCanvasContext("canvasProgress", this);
    e.setLineWidth(8),
      e.setStrokeStyle("#00BFFF"),
      e.beginPath(),
      e.arc(65, 65, 61, -Math.PI / 2, t * Math.PI - Math.PI / 2, false),
      e.stroke(),
      e.draw();
  },
  //图片分析
  uploads: function () {
    var that = this
    var stain = "setdatas[" + 0 + "].percent";
    var acne = "setdatas[" + 1 + "].percent";
    var darkCircle = "setdatas[" + 2 + "].percent";
    wx.chooseImage({
      count: 1, // 默认9
      sizeType: ['compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success: function (res) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        //console.log( res )
        that.setData({
          img: res.tempFilePaths[0],
          code: 1
        }),
        that.countInterval(that.data.percent);
        that.drawProgressbg();
        wx.showLoading({
          title: "肤质分析中...",
          mask: true
        }),
          wx.uploadFile({
            url: skinUrl,
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
              console.info(str);
              if (str.code == "0") {
                that.setData({
                  code: 0,
                  percent: str.health,
                  [stain]: str.stain,
                  [acne]: str.acne,
                  [darkCircle]: str.darkCircle
                })
                that.countInterval(str.health);
                that.drawProgressbg();
              } else if (str.code == "5") {
                that.setData({
                  info: str.msg
                })
              } else if (str.code == "1") {
                that.setData({
                  info: str.msg
                })
              } else {
                that.setData({
                  info: "Sorry 小程序远走高飞了",
                })
              }
            },
            fail: function (res) {
              wx.hideLoading();
              that.setData({
                info: '小程序离家出走了稍后再试',
              })
            }
          })
      }
    })
  },
  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {
    return {
      title: '皮肤肤质分析小程序',
      path: '/pages/skin/skin',
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
})