var app = getApp();
// pages/ocr/ocr.js
var ocrtext="";
var api = require('../../utils/api.js');
var BdOcrUrl = api.getBdOcrUrl();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    array: ['通用OCR', '身份证OCR（正面）', '身份证OCR（背面）', '银行卡OCR', '手写字体识别','通用OCR（含位置信息版）', '通用OCR（含生僻字版）', '通用OCR（高精度版）', '通用OCR（含位置高精度版）','驾驶证OCR', '行驶证OCR', '网图OCR', '营业执照OCR', '车牌OCR', '彩票OCR', '公式OCR', '通用票据OCR', '表格OCR（提交）', '表格OCR（获取）'],
    index: 0,
    words:"",
    wordsesultNum:"",
    address:"",
    sex:"",
    birth:"",
    idCardNum:"",
    imageStatus:"",
    name:"",
    nation:"",
    riskType:"",
    authority:"",
    expiryDate:"",
    issueDate:"",
    bankCardNumber:"",
    bankCardType:"",
    bankName:"",
    userInfo: {},
    backUserInfo: {},//后台得到的微信用户信息
    hasUserInfo: false,
    openId: "",
    nickName: "",
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    info:""
  },
  bindPickerChange:function(e){
    this.setData({
      index: e.detail.value,
      words: "",
      wordsesultNum: "",
      address: "",
      birth: "",
      idCardNum: "",
      imageStatus: "",
      name: "",
      nation: "",
      riskType: "",
      authority: "",
      expiryDate: "",
      issueDate: "",
      bankCardNumber: "",
      bankCardType: "",
      info: "",
      bankName: ""
    })
    console.log('picker发送选择改变，携带值为', e.detail.value);
    var ocrindex = e.detail.value;
    console.info(ocrindex);
    if (ocrindex != '0' && ocrindex != '1' && ocrindex != '2' && ocrindex != '3' && ocrindex != '4') {
      wx.showModal({
        title: '温馨提示',
        showCancel: false,
        content: '目前只有通用OCR,银行卡识别,身份证正反面识别,手写字体识别可用 ',
        success: function (res) {
          this.setData({
            ocrindex:0
          })
        },
        fail: function (res) {
          this.setData({
            ocrindex:'0',
          })
        }
      })
    } 
  },
  copyTEXT: function (e) {
    var that = this;
    var ocrType = that.data.index;
    var dataText = "";
    if(ocrType=="0"||ocrType=="4"){
      dataText = that.data.words;
    } else if (ocrType == "1"){
      dataText = that.data.name + "\n" + that.data.sex + "\n" + that.data.nation + that.data.birth + "\n" + that.data.address + "\n" + that.data.idCardNum;
    } else if (ocrType == "2") {
      dataText = that.data.authority + "\n" + that.data.expiryDate + "\n" + that.data.issueDate;
    } else if (ocrType == "3") {
      dataText = that.data.bankName + "\n" + that.data.bankCardType + "\n" + that.data.bankCardNumber;
    } else {
      dataText = "疯狂给小帅丶打Call";
    }
    wx.setClipboardData({
      data: dataText,
      success: function (res) {
        wx.showModal({
          title: '提示',
          content: '复制成功',
          showCancel: false
        })
      }
    })
  },
  clear:function(){
    var that = this;
    console.info(that);
    this.setData({
      words: "",
      wordsesultNum: "",
      address: "",
      birth: "",
      sex: "",
      idCardNum: "",
      imageStatus: "",
      name: "",
      nation: "",
      riskType: "",
      authority: "",
      expiryDate: "",
      issueDate: "",
      bankCardNumber: "",
      bankCardType: "",
      info: "",
      bankName: ""
    })  
  },
  uploads: function () {
    var that = this;
    console.info(that);
    var ocrindex = that.data.index;
    console.info(ocrindex);
    if (ocrindex != '0' && ocrindex != '1' && ocrindex != '2' && ocrindex != '3' && ocrindex != '4') {
      wx.showModal({
        title: '友情提示',
        showCancel:false,
        content: '目前只有通用OCR,银行卡识别,身份证正反面识别,手写字体识别可用 ',
        success:function(res){
          that.setData({
            ocrindex: 0,
          })
        },
        fail:function(res){
          that.setData({
            ocrindex: 0,
          })
        }
      })
    }else{
      console.info(ocrindex);
    wx.chooseImage({
      count: 1, // 默认9
      sizeType: ['compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success: function (res) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        //console.log( res )
        that.setData({
          img: res.tempFilePaths[0],
          words: "",
          sex: "",
          wordsesultNum: "",
          address: "",
          birth: "",
          idCardNum: "",
          imageStatus: "",
          name: "",
          nation: "",
          riskType: "",
          authority: "",
          expiryDate: "",
          issueDate: "",
          bankCardNumber: "",
          bankCardType: "",
          info: "",
          bankName: ""
        })
        wx.showLoading({
          title: "努力识别中...",
          mask: true
        }),
        wx.uploadFile({
          url: BdOcrUrl,
          filePath: res.tempFilePaths[0],
          header: {
            'content-type': 'multipart/form-data'
          },
          name: 'file',
          formData: {
            'ocrtype': ocrindex,
            'openId': that.data.openId,
            'nickName': that.data.nickName
          },
          success: function (res) {
            wx.hideLoading();
            console.info(res);
            var data = res.data;
            var str = JSON.parse(data);
            if(str.code=="0"){
              if(str.ocrType=="ocr"){
                that.setData({
                  words: str.words,
                  wordsesultNum: str.wordsesultNum
                })
              } else if (str.ocrType == "idcardf"){
                that.setData({
                  address: str.address,
                  birth: str.birth,
                  idCardNum: str.idCardNum,
                  imageStatus: str.imageStatus,
                  name: str.name,
                  nation: str.nation,
                  riskType: str.riskType,
                  sex: str.sex
                })
              } else if (str.ocrType == "idcardb") {
                that.setData({
                  authority: str.authority,
                  expiryDate: str.expiryDate,
                  imageStatus: str.imageStatus,
                  issueDate: str.issueDate,
                  riskType: str.riskType
                })
              } else if (str.ocrType == "bank") {
                that.setData({
                  bankCardNumber: str.bankCardNumber,
                  bankCardType: str.bankCardType,
                  bankName: str.bankName
                })
              } else if (str.ocrType == "handwriting") {
                that.setData({
                  words: str.words,
                  wordsesultNum: str.wordsesultNum
                })
              } else{
                that.setData({
                  info:"未能识别出文字"
                })
              }
            } else if (str.code == "5"){
              that.setData({
               info:"未能识别出相关内容"
              })
            } else{
              that.setData({
                info: "Sorry 小程序远走高飞了",
              })
            }
          },
          fail: function (res) {
            wx.hideLoading();
            console.log(res);
            that.setData({
              info: '小程序离家出走了稍后再试',
            })
          }
        })
      }
    })
    }
  },
  onShareAppMessage: function () {
    return {
      title: 'OCR识别',
      path: '/pages/ocr/ocr',
      imageUrl:'https://www.xsshome.cn/timg.jpg',
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
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function () {
    var openIdKey = app.data.openId;
    this.setData({
      openId: openIdKey
    })
  }
})