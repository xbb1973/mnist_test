//域名地址
var prefix = '自己的域名';
//接口授权码
var authCode = '自己的接口授权码';
//接口访问类型
var clientType= 'wsc';
/*接口地址开始*/
//授权获取微信用户信息url
const oauthurl = prefix + '/wcsp/oauth';
const loginurl = prefix + '/wcsp/login';
//颜值分析url
const faceurl = prefix+'/rest/face/detect?clientType='+clientType+'&apiType=face&authCode='+authCode;
//魅力值分析url
const charmurl = prefix + '/rest/youtu/detect?clientType=' + clientType + '&apiType=face&authCode=' + authCode;
//菜品识别url
const dishurl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=dish&authCode=' + authCode;
//车型识别url
const carurl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=car&authCode=' + authCode;
//植物识别url
const planturl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=plant&authCode=' + authCode;
//动物识别url
const animalurl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=animal&authCode=' + authCode;
//logo识别url
const logourl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=logo&authCode=' + authCode;
//食材识别url
const ingredienturl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=ingredient&authCode=' + authCode;
//花卉识别url
const flowerturl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=flower&authCode=' + authCode;
//手写识别url
const hwurl = prefix + '/rest/youtu/detect?clientType=' + clientType + '&apiType=hw&authCode=' + authCode;
//手势识别url
const hturl = prefix + '/rest/youtu/detect?clientType=' + clientType + '&apiType=ht&authCode=' + authCode;
//人脸融合(疯狂变脸)url
const facemergeurl = prefix + '/rest/ptu/facemerge?clientType=' + clientType + '&authCode=' + authCode;
//图片转字符图片url
const image2asciiurl = prefix + '/rest/ias/image2ascii?clientType=' + clientType + '&authCode=' + authCode;
//人脸对比上传图片url
const faceageuploadurl = prefix + '/rest/ptu/uploadFA?clientType=' + clientType + '&authCode=' + authCode;
//人脸对比url
const faceagecoresurl = prefix + '/rest/ptu/detectFA?clientType=' + clientType + '&authCode=' + authCode;
//文字识别bd url
const bdocrsurl = prefix + '/rest/ocr/detect?clientType=' + clientType + '&authCode=' + authCode;
//人脸美妆url
const facecosmeticurl = prefix + '/rest/ptu/facecosmetic?clientType=' + clientType + '&authCode=' + authCode;
//人脸变妆url
const facedecorationurl = prefix + '/rest/ptu/facedecoration?clientType=' + clientType + '&authCode=' + authCode;
//人脸滤镜url
const ptuimgfilterurl = prefix + '/rest/ptu/ptuimgfilter?clientType=' + clientType + '&authCode=' + authCode;
//图片滤镜url
const visionimgfilterurl = prefix + '/rest/ptu/visionimgfilter?clientType=' + clientType + '&authCode=' + authCode;
//大头贴url
const facestickerurl = prefix + '/rest/ptu/facesticker?clientType=' + clientType + '&authCode=' + authCode;
//肤质分析
const faceskinurl = prefix + '/rest/fpp/detect?clientType=' + clientType + '&authCode=' + authCode;
//地标景区识别url
const landmarkurl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=landmark&authCode=' + authCode;
//红酒识别url
const redwineurl = prefix + '/rest/icr/detect?clientType=' + clientType + '&apiType=redwine&authCode=' + authCode;
//速算识别url
const qcurl = prefix + '/rest/youtu/detect?clientType=' + clientType + '&apiType=qc&authCode=' + authCode;
//速算识别url
const physiognomyurl = prefix + '/rest/physiognomy/face?clientType=' + clientType + '&authCode=' + authCode;
/*常量函数*/
function getOauthUrl() {
  return oauthurl;
}
function getPhysiognomyUrl() {
  return physiognomyurl;
}
function getLoginUrl() {
  return loginurl;
}
function getFaceUrl(){
  return faceurl;
}
function getCharmUrl() {
  return charmurl;
}
function getDishUrl() {
  return dishurl;
}
function getCarUrl() {
  return carurl;
}
function getPlantUrl() {
  return planturl;
}
function getAnimalUrl() {
  return animalurl;
}
function getLogoUrl() {
  return logourl;
}
function getIngredientUrl() {
  return ingredienturl;
}
function getHwUrl() {
  return hwurl;
}
function getHtUrl() {
  return hturl;
}
function getFacemergeUrl() {
  return facemergeurl;
}
function getFaceAgeUploadUrl() {
  return faceageuploadurl;
}
function getFaceAgeCoresUrl() {
  return faceagecoresurl;
}
function getBdOcrUrl() {
  return bdocrsurl;
}
function getFlowerUrl() {
  return flowerturl;
}
function getImage2asciiurl(){
  return image2asciiurl;
}
function getFacecosmeticiurl() {
  return facecosmeticurl;
}
function getFacedecorationurl() {
  return facedecorationurl;
}
function getPtuimgfilterurl() {
  return ptuimgfilterurl;
}
function getVsionimgfilterurl() {
  return visionimgfilterurl;
}
function getFacestickerurl() {
  return facestickerurl;
}
function getFaceskinurl(){
  return faceskinurl;
}
function getLandmarkUrl() {
  return landmarkurl;
}
function getRedwineUrl() {
  return redwineurl;
}
function getQcUrl() {
  return qcurl;
}
/*暴露常量函数*/
module.exports.getOauthUrl = getOauthUrl;
module.exports.getFaceUrl = getFaceUrl;
module.exports.getCharmUrl = getCharmUrl;
module.exports.getDishUrl = getDishUrl;
module.exports.getCarUrl = getCarUrl;
module.exports.getPlantUrl = getPlantUrl;
module.exports.getAnimalUrl = getAnimalUrl;
module.exports.getLogoUrl = getLogoUrl;
module.exports.getIngredientUrl = getIngredientUrl;
module.exports.getHwUrl = getHwUrl;
module.exports.getHtUrl = getHtUrl;
module.exports.getFacemergeUrl = getFacemergeUrl;
module.exports.getFaceAgeUploadUrl = getFaceAgeUploadUrl;
module.exports.getFaceAgeCoresUrl = getFaceAgeCoresUrl;
module.exports.getBdOcrUrl = getBdOcrUrl;
module.exports.getFlowerUrl = getFlowerUrl;
module.exports.getImage2asciiurl = getImage2asciiurl;
module.exports.getFacecosmeticiurl = getFacecosmeticiurl;
module.exports.getFacedecorationurl = getFacedecorationurl;
module.exports.getPtuimgfilterurl = getPtuimgfilterurl;
module.exports.getVsionimgfilterurl = getVsionimgfilterurl;
module.exports.getFacestickerurl = getFacestickerurl;
module.exports.getFaceskinurl = getFaceskinurl;
module.exports.getLandmarkUrl = getLandmarkUrl;
module.exports.getRedwineUrl = getRedwineUrl;
module.exports.getQcUrl = getQcUrl;
module.exports.getLoginUrl = getLoginUrl;
module.exports.getPhysiognomyUrl = getPhysiognomyUrl;