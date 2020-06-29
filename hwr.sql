/*
 Navicat Premium Data Transfer

 Source Server         : docker-mysql02
 Source Server Type    : MySQL
 Source Server Version : 80020
 Source Host           : localhost:3305
 Source Schema         : hwr

 Target Server Type    : MySQL
 Target Server Version : 80020
 File Encoding         : 65001

 Date: 28/06/2020 06:37:23
*/

create database hwr;
use hwr;
/*
 Navicat Premium Data Transfer

 Source Server         : docker-mysql02
 Source Server Type    : MySQL
 Source Server Version : 80020
 Source Host           : localhost:3305
 Source Schema         : hwr

 Target Server Type    : MySQL
 Target Server Version : 80020
 File Encoding         : 65001

 Date: 28/06/2020 06:37:23
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ckpt
-- ----------------------------
DROP TABLE IF EXISTS `ckpt`;
CREATE TABLE `ckpt` (
                        `ckpt_id` bigint NOT NULL AUTO_INCREMENT,
                        `train_id` int DEFAULT NULL,
                        `dataset_id` int DEFAULT NULL,
                        `ckpt_path` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `create_time` datetime DEFAULT NULL,
                        `ckpt_name` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `ckpt_tag` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `ckpt_status` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `epoch` int DEFAULT NULL,
                        `step` int DEFAULT NULL,
                        `acc` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `loss` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `modal` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        `comment` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                        PRIMARY KEY (`ckpt_id`)
) ENGINE=InnoDB AUTO_INCREMENT=424 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;



-- ----------------------------
-- Table structure for dataset
-- ----------------------------
DROP TABLE IF EXISTS `dataset`;
CREATE TABLE `dataset` (
                           `dataset_id` int NOT NULL AUTO_INCREMENT,
                           `dataset_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                           `dataset_size` bigint DEFAULT NULL,
                           `dataset_update_time` datetime DEFAULT NULL,
                           `dataset_create_time` datetime DEFAULT NULL,
                           `dataset_path` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                           `dataset_comment` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                           `dataset_tag` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                           `dataset_status` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                           `dataset_version` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                           PRIMARY KEY (`dataset_id`)
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;


-- ----------------------------
-- Table structure for train
-- ----------------------------
DROP TABLE IF EXISTS `train`;
CREATE TABLE `train` (
                         `train_id` int NOT NULL AUTO_INCREMENT,
                         `train_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                         `train_create_time` datetime DEFAULT NULL,
                         `train_update_time` datetime DEFAULT NULL,
                         `train_record_num` bigint DEFAULT NULL,
                         `train_status` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                         `train_dataset_id` int DEFAULT NULL,
                         `train_ckpt_id` int DEFAULT NULL,
                         `train_task_id` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                         `train_took_seconds` bigint DEFAULT NULL,
                         `train_epochs` int DEFAULT NULL,
                         `train_epoch` int DEFAULT NULL,
                         `train_steps` int DEFAULT NULL,
                         `train_step` int DEFAULT NULL,
                         `train_acc` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                         `train_loss` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                         `train_comment` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                         `train_modal` varchar(255) CHARACTER SET utf8 COLLATE utf8_bin DEFAULT NULL,
                         PRIMARY KEY (`train_id`)
) ENGINE=InnoDB AUTO_INCREMENT=195 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;


-- ----------------------------
-- Table structure for train_record
-- ----------------------------
DROP TABLE IF EXISTS `train_record`;
CREATE TABLE `train_record` (
                                `record_id` bigint NOT NULL AUTO_INCREMENT,
                                `train_id` int DEFAULT NULL,
                                `update_time` datetime DEFAULT NULL,
                                `train_dataset_id` int DEFAULT NULL,
                                `train_ckpt_id` int DEFAULT NULL,
                                `train_task_id` int DEFAULT NULL,
                                `train_epochs` int DEFAULT NULL,
                                `train_epoch` int DEFAULT NULL,
                                `train_steps` int DEFAULT NULL,
                                `train_step` int DEFAULT NULL,
                                `train_acc` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                                `train_loss` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                                `train_status` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                                `train_modal` varchar(255) COLLATE utf8_bin DEFAULT NULL,
                                PRIMARY KEY (`record_id`)
) ENGINE=InnoDB AUTO_INCREMENT=11117 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;


SET FOREIGN_KEY_CHECKS = 1;
