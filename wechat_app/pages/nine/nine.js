var app = getApp();
// pages/ninegrid/ninegrid.js
Page({
  /**
   * 页面的初始数据
   */
  data: {
    routers: [
      {
        id: '0',
        name: '菜品识别',
        url: '../../pages/dish/dish',
        icon: '../../image/dishNine.png'
      },
      {
        id: '1',
        name: '车型识别',
        url: '../../pages/car/car',
        icon: '../../image/carNine.png'
      },
      {
        id: '2',
        name: '植物识别',
        url: '../../pages/plant/plant',
        icon: '../../image/plantNine.png'
      },
      {
        id: '13',
        name: '花卉识别',
        url: '../../pages/flower/flower',
        icon: '../../image/flower.png'
      },
      // {
      //   id: '4',
      //   name: '图像主体识别',
      //   url: '../../pages/subject/subject',
      //   icon: '../../image/image_HL.png'
      // },
      {
        id: '19',
        name: '红酒识别',
        url: '../../pages/redwine/redwine',
        icon: '../../image/redwine.png'
      },
      {
        id: '20',
        name: '景区识别',
        url: '../../pages/landmark/landmark',
        icon: '../../image/landmark.png'
      },
      // {
      //   id: '21',
      //   name: '驾驶行为分析',
      //   url: '../../pages/driverbehavior/driverbehavior',
      //   icon: '../../image/driverbehavior.png'
      // },
      {
        id: '23',
        name: '面相分析',
        url: '../../pages/physiognomy/physiognomy',
        icon: '../../image/physiognomy.png'
      },
      {
        id: '22',
        name: '速算题目识别',
        url: '../../pages/quickcalculation/quickcalculation',
        icon: '../../image/quickcalculation.png'
      },
      {
        id: '12',
        name: '快乐相似脸',
        url: '../../pages/facedetectcrossageface/facedetectcrossageface',
        icon: '../../image/ageNine.png'
      },
      // {
      // id: '5',
      //   name: '车辆定损识别',
      //   url: '../../pages/cardamage/cardamage',
      //   icon: '../../image/damageNine.png'
      // },
      {
        id: '3',
        name: '动物识别',
        url: '../../pages/animal/animal',
        icon: '../../image/animalNine.png'
      },
      {
        id: '6',
        name: '肤质分析',
        url: '../../pages/skin/skin',
        icon: '../../image/skin.png'
      },
      {
        id: '7',
        name: '品牌LOGO识别',
        url: '../../pages/logo/logo',
        icon: '../../image/tag_HL.png'
      },
      // {
      //   id: '8',
      //   name: '相似相同图像搜索',
      //   url: '../../pages/resemble/resemble',
      //   icon: '../../image/equal_HL.png'
      // },
      {
        id: '14',
        name: '图片转字符图片',
        url: '../../pages/image2ascii/image2ascii',
        icon: '../../image/ascii_HL.png'
      },
      {
        id: '9',
        name: '食材识别',
        url: '../../pages/ingredient/ingredient',
        icon: '../../image/ingredient_HL.png'
      },
      {
        id: '10',
        name: '手势识别',
        url: '../../pages/youtuHT/youtuHT',
        icon: '../../image/youtuHT.png'
      },
      {
        id: '11',
        name: '手写文字识别',
        url: '../../pages/youtuHW/youtuHW',
        icon: '../../image/youtuHW.png'
      },
      {
        id: '15',
        name: '人脸美妆',
        url: '../../pages/facecosmetic/facecosmetic',
        icon: '../../image/facecosmetic.png'
      },
      {
        id: '16',
        name: '人脸变妆',
        url: '../../pages/facedecoration/facedecoration',
        icon: '../../image/facedecoration.png'
      },
      {
        id: '17',
        name: '滤镜特效',
        url: '../../pages/imgfilter/imgfilter',
        icon: '../../image/imgfilter.png'
      },
      {
        id: '18',
        name: '大头贴',
        url: '../../pages/facesticker/facesticker',
        icon: '../../image/facesticker.png'
      }
    ]
  },
  toPage: function (event) {
    var route = event.currentTarget.id;
    if (route == 0) {
      wx.navigateTo({
        url: '/pages/dish/dish',
      })
    } else if (route == 1) {
      wx.navigateTo({
        url: '/pages/car/car',
      })
    } else if (route == 2) {
      wx.navigateTo({
        url: '/pages/plant/plant',
      })
    } else if (route == 3) {
      wx.navigateTo({
        url: '/pages/animal/animal',
      })
    } else if (route == 4) {
      wx.showModal({
        title: '功能说明',
        content: '检测图像中的主体位置，没啥好玩的。就不开放了。',
        showCancel: false,
      })
    } else if (route == 7) {
      wx.navigateTo({
        url: '/pages/logo/logo',
      })
    } else if (route == 9){
      wx.navigateTo({
        url: '/pages/ingredient/ingredient',
      })
    } else if (route == 10){
      wx.navigateTo({
        url: '/pages/youtuHT/youtuHT',
      })
    } else if (route == 11){
      wx.navigateTo({
        url: '/pages/youtuHW/youtuHW',
      })
    } else if (route == 12){
      wx.navigateTo({
        url: '/pages/facedetectcrossageface/facedetectcrossageface',
      })
    } else if (route == 13) {
      wx.navigateTo({
        url: '/pages/flower/flower',
      })
    } else if (route == 14){
      wx.navigateTo({
        url: '/pages/image2ascii/image2ascii',
      })
    } else if (route == 15) {
      wx.navigateTo({
        url: '/pages/facecosmetic/facecosmetic',
      })
    } else if (route == 16) {
      wx.navigateTo({
        url: '/pages/facedecoration/facedecoration',
      })
    } else if (route == 17) {
      wx.navigateTo({
        url: '/pages/imgfilter/imgfilter',
      })
    } else if (route == 18) {
      wx.navigateTo({
        url: '/pages/facesticker/facesticker',
      })
    } else if (route == 6) {
      wx.navigateTo({
        url: '/pages/skin/skin',
      })
    } else if (route == 19){
      wx.navigateTo({
        url: '/pages/redwine/redwine',
      })
    } else if (route == 20) {
      wx.navigateTo({
        url: '/pages/landmark/landmark',
      })
    } else if (route == 21) {
      wx.showModal({
        title: '功能说明',
        content: '功能待完善，即将开放',
        showCancel: false,
      })
    } else if (route == 22) {
      wx.navigateTo({
        url: '/pages/quickcalculation/quickcalculation',
      })
    } else if(route==23){
      wx.navigateTo({
        url: '/pages/physiognomy/physiognomy',
      })
    }else {
      wx.showModal({
        title: '温馨提示',
        content: '功能暂未开放，请点击别的功能试试',
        showCancel: false,
        confirmText: '好的',
      })
    }
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function () {
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
      return{
        title:'X-AI玩转图像、文字、人脸识别'
      }
  }
})