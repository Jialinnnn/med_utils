import json
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict

class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)
        self.dict_list = list()
        self.loss_rpn_bbox = list()
        self.loss_rpn_cls = list()
        self.loss_bbox = list()
        self.loss_cls = list()
        self.loss = list()
        self.acc = list()
        self.mAP = list()
        self.mAP_s = list()
        self.mAP_m = list()
        self.mAP_l = list()
        self.mAP_50 = list()
        self.mAP_75 = list()

    def load_data(self):
        for line in self.log:
            info = json.loads(line)
            # print('info:', info)
            # if info['mode'] == 'train':
            self.dict_list.append(info)

        for i in range(0, len(self.dict_list)):
            if 'coco/bbox_mAP' in self.dict_list[i]:
                mAP_value = dict(self.dict_list[i])['coco/bbox_mAP']
                mAP_s_value = dict(self.dict_list[i])['coco/bbox_mAP_s']
                mAP_m_value = dict(self.dict_list[i])['coco/bbox_mAP_m']
                mAP_l_value = dict(self.dict_list[i])['coco/bbox_mAP_l']
                mAP_50_value = dict(self.dict_list[i])['coco/bbox_mAP_50']
                mAP_75_value = dict(self.dict_list[i])['coco/bbox_mAP_75']
                self.mAP.append(mAP_value)
                self.mAP_s.append(mAP_s_value)
                self.mAP_m.append(mAP_m_value)
                self.mAP_l.append(mAP_l_value)
                self.mAP_50.append(mAP_50_value)
                self.mAP_75.append(mAP_75_value)
            else:
            # for value, key in dict(self.dict_list[i]).items():
                # 读取每一行的信息
                loss_dfl_value = dict(self.dict_list[i])['loss_dfl']
                # loss_rpn_bbox_value = dict(self.dict_list[i])['loss_rpn_bbox']
                loss_bbox_value = dict(self.dict_list[i])['loss_bbox']
                loss_cls_value = dict(self.dict_list[i])['loss_cls']
                loss_value = dict(self.dict_list[i])['loss']
                # acc_value = dict(self.dict_list[i])['acc']
                # 将其保存至对应列表中
                self.loss_rpn_cls.append(loss_dfl_value)
                # self.loss_rpn_bbox.append(loss_rpn_bbox_value)
                self.loss_bbox.append(loss_bbox_value)
                self.loss_cls.append(loss_cls_value)
                self.loss.append(loss_value)
                # self.acc.append(acc_value)

        # # 清除list中的重复项
        # self.loss_rpn_cls = list(OrderedDict.fromkeys(self.loss_rpn_cls))
        # self.loss_rpn_bbox = list(OrderedDict.fromkeys(self.loss_rpn_bbox))
        # self.loss_bbox = list(OrderedDict.fromkeys(self.loss_bbox))
        # self.loss_cls = list(OrderedDict.fromkeys(self.loss_cls))
        # self.loss = list(OrderedDict.fromkeys(self.loss))
        # self.acc = list(OrderedDict.fromkeys(self.acc))
        # self.mAP = list(OrderedDict.fromkeys(self.mAP))
        # self.mAP_s = list(OrderedDict.fromkeys(self.mAP_s))
        # self.mAP_m = list(OrderedDict.fromkeys(self.mAP_m))
        # self.mAP_l = list(OrderedDict.fromkeys(self.mAP_l))

    def show_chart(self):
        x = range(len(self.loss_rpn_cls))  # 即x轴
        plt.figure(1)
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('iters', fontsize=15)  # x轴标签
        plt.ylabel('loss', fontsize=15)  # y轴标签
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x, self.loss_rpn_cls, linewidth=1, linestyle="solid", label="loss_DFL")
        # plt.plot(x, self.loss_rpn_bbox, linewidth=1, linestyle="solid", label="loss_rpn_bbox")
        plt.plot(x, self.loss_bbox, linewidth=1, linestyle="solid", label="loss_GIoU")
        plt.plot(x, self.loss_cls, linewidth=1, linestyle="solid", label="los_QFL")
        plt.plot(x, self.loss, linewidth=1, linestyle="solid", label="loss")
        plt.legend(fontsize=10)
        plt.title('Loss curve', fontsize=20)
        plt.savefig('../result/fig/gfl_r101_fpn_loss_train.png')
        plt.show()
        # plt.close()
        print("successful save train loss curve! ")

        x_mAP = range(len(self.mAP))  # 即x轴
        plt.figure(2)
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('epochs', fontsize=15)  # x轴标签
        plt.ylabel('mAP', fontsize=15)  # y轴标签
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_mAP, self.mAP, linewidth=1, linestyle="solid", label="AP_co")
        plt.plot(x_mAP, self.mAP_50, linewidth=1, linestyle="solid", label="AP_50")
        plt.plot(x_mAP, self.mAP_75, linewidth=1, linestyle="solid", label="AP_75")
        plt.plot(x_mAP, self.mAP_s, linewidth=1, linestyle="solid", label="AP_small")
        plt.plot(x_mAP, self.mAP_m, linewidth=1, linestyle="solid", label="AP_medium")
        plt.plot(x_mAP, self.mAP_l, linewidth=1, linestyle="solid", label="AP_large")
        plt.legend(fontsize=10)
        plt.title('Train mAP', fontsize=20)
        plt.savefig('../result/fig/gfl_r101_fpn_mAP_train.png')
        plt.show()
        # plt.close()
        print("successful save mAP curve! ")


if __name__ == '__main__':
    json_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # 返回上层目录
    json_path = json_root + "/work_dirs/gfl_r101_fpn_1x_coco/20240322_221432_43.8/vis_data/20240322_221432.json"  # 数据集路径
    # json_path = json_root + "/work_dirs/gfl_r101_fpn_1x_coco/20240517_205946/vis_data/20240517_205946.json"
# detection/MMDetection-main/tools/work_dirs/gfl_r101_fpn_1x_coco/20240517_205946/vis_data/20240517_205946.json
#     json_path = json_root + "/work_dirs/gfl_r101_fpn_1x_coco/20240520_104515/vis_data/20240520_104515.json"
# detection / MMDetection - main / tools / work_dirs / gfl_r101_fpn_1x_coco / 20240520_104515 / vis_data / 20240520_104515.json
#     json_path = json_root + "/work_dirs/gfl_r101_fpn_1x_coco/20240521_162424/vis_data/20240521_162424.json"
# detection / MMDetection - main / tools / work_dirs / gfl_r101_fpn_1x_coco / 20240521_162424 / vis_data / 20240521_162424.json
    x = visualize_mmdetection(json_path)
    x.load_data()
    x.show_chart()


# import json
# import matplotlib.pyplot as plt
# import sys
# import os
# from collections import OrderedDict
#
# class visualize_mmdetection():
#     def __init__(self, path):
#         self.log = open(path)
#         self.dict_list = list()
#         # self.loss_rpn_bbox = list()
#         # self.loss_rpn_cls = list()
#         # self.loss_bbox = list()
#         # self.loss_cls = list()
#         self.loss = list()
#         # self.acc = list()
#         # self.mAP = list()
#         # self.mAP_s = list()
#         # self.mAP_m = list()
#         # self.mAP_l = list()
#         self.mAP_50 = list()
#         # self.mAP_75 = list()
#
#     def load_data(self):
#         for line in self.log:
#             info = json.loads(line)
#             # print('info:', info)
#             # if info['mode'] == 'train':
#             self.dict_list.append(info)
#
#         for i in range(0, len(self.dict_list)):
#             if 'coco/bbox_mAP' in self.dict_list[i]:
#                 # mAP_value = dict(self.dict_list[i])['coco/bbox_mAP']
#                 # mAP_s_value = dict(self.dict_list[i])['coco/bbox_mAP_s']
#                 # mAP_m_value = dict(self.dict_list[i])['coco/bbox_mAP_m']
#                 # mAP_l_value = dict(self.dict_list[i])['coco/bbox_mAP_l']
#                 mAP_50_value = dict(self.dict_list[i])['coco/bbox_mAP_50']
#                 # mAP_75_value = dict(self.dict_list[i])['coco/bbox_mAP_75']
#                 # self.mAP.append(mAP_value)
#                 # self.mAP_s.append(mAP_s_value)
#                 # self.mAP_m.append(mAP_m_value)
#                 # self.mAP_l.append(mAP_l_value)
#                 self.mAP_50.append(mAP_50_value)
#                 # self.mAP_75.append(mAP_75_value)
#             else:
# #             # for value, key in dict(self.dict_list[i]).items():
#                 # 读取每一行的信息
#                 # loss_dfl_value = dict(self.dict_list[i])['loss_dfl']
#                 # # loss_rpn_bbox_value = dict(self.dict_list[i])['loss_rpn_bbox']
#                 # loss_bbox_value = dict(self.dict_list[i])['loss_bbox']
#                 # loss_cls_value = dict(self.dict_list[i])['loss_cls']
#                 loss_value = dict(self.dict_list[i])['loss']
#                 # acc_value = dict(self.dict_list[i])['acc']
#                 # 将其保存至对应列表中
#                 # self.loss_rpn_cls.append(loss_dfl_value)
#                 # # self.loss_rpn_bbox.append(loss_rpn_bbox_value)
#                 # self.loss_bbox.append(loss_bbox_value)
#                 # self.loss_cls.append(loss_cls_value)
#                 self.loss.append(loss_value)
#                 # self.acc.append(acc_value)
#
#         # # 清除list中的重复项
#         # self.loss_rpn_cls = list(OrderedDict.fromkeys(self.loss_rpn_cls))
#         # self.loss_rpn_bbox = list(OrderedDict.fromkeys(self.loss_rpn_bbox))
#         # self.loss_bbox = list(OrderedDict.fromkeys(self.loss_bbox))
#         # self.loss_cls = list(OrderedDict.fromkeys(self.loss_cls))
#         # self.loss = list(OrderedDict.fromkeys(self.loss))
#         # self.acc = list(OrderedDict.fromkeys(self.acc))
#         # self.mAP = list(OrderedDict.fromkeys(self.mAP))
#         # self.mAP_s = list(OrderedDict.fromkeys(self.mAP_s))
#         # self.mAP_m = list(OrderedDict.fromkeys(self.mAP_m))
#         # self.mAP_l = list(OrderedDict.fromkeys(self.mAP_l))
#
#     def show_chart(self):
#         x = range(len(self.loss_rpn_cls))  # 即x轴
#         plt.figure(1)
#         ax = plt.axes()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         plt.xlabel('iters')  # x轴标签
#         plt.ylabel('loss')  # y轴标签
#         # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#         # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#         plt.plot(x, self.loss_rpn_cls, linewidth=1, linestyle="solid", label="loss_DFL")
#         # plt.plot(x, self.loss_rpn_bbox, linewidth=1, linestyle="solid", label="loss_rpn_bbox")
#         plt.plot(x, self.loss_bbox, linewidth=1, linestyle="solid", label="loss_GIoU")
#         plt.plot(x, self.loss_cls, linewidth=1, linestyle="solid", label="los_QFL")
#         plt.plot(x, self.loss, linewidth=1, linestyle="solid", label="loss")
#         plt.legend()
#         plt.title('Loss curve')
#         plt.savefig('../result/fig/gfl_r101_fpn_loss.png')
#         plt.show()
#         # plt.close()
#         print("successful save train loss curve! ")
#
#         x_mAP = range(len(self.mAP))  # 即x轴
#         plt.figure(2)
#         ax = plt.axes()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         plt.xlabel('epochs')  # x轴标签
#         plt.ylabel('mAP')  # y轴标签
#         # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#         # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#         plt.plot(x_mAP, self.mAP, linewidth=1, linestyle="solid", label="AP_co")
#         plt.plot(x_mAP, self.mAP_50, linewidth=1, linestyle="solid", label="AP_50")
#         plt.plot(x_mAP, self.mAP_75, linewidth=1, linestyle="solid", label="AP_75")
#         plt.plot(x_mAP, self.mAP_s, linewidth=1, linestyle="solid", label="AP_small")
#         plt.plot(x_mAP, self.mAP_m, linewidth=1, linestyle="solid", label="AP_medium")
#         plt.plot(x_mAP, self.mAP_l, linewidth=1, linestyle="solid", label="AP_large")
#         plt.legend()
#         plt.title('Eval mAP')
#         plt.savefig('../result/fig/gfl_r101_fpn_mAP.png')
#         plt.show()
#         # plt.close()
#         print("successful save mAP curve! ")
#
#
# def plot(ap1,ap2):
#     x_mAP = range(len(ap2))  # 即x轴
#     plt.figure(2)
#     ax = plt.axes()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.xlabel('epochs')  # x轴标签
#     plt.ylabel('mAP')  # y轴标签
#     # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#     # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#     plt.plot(x_mAP, ap1, linewidth=1, linestyle="solid", label="ResNet101 train AP50 with transfer learning")
#     plt.plot(x_mAP, ap2, linewidth=1, linestyle="solid", label="ResNet101 train AP50 without transfer learning")
#     plt.legend()
#     plt.title('Eval mAP')
#     plt.savefig('../result/fig/transfer_learningR50.png')
#     plt.show()
#     # plt.close()
#     print("successful save mAP curve! ")
#
# def plotloss(loss1,loss2):
#     x_loss = range(len(loss2))  # 即x轴
#     plt.figure(3)
#     ax = plt.axes()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.xlabel('iters')  # x轴标签
#     plt.ylabel('Loss')  # y轴标签
#     # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#     # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#     plt.plot(x_loss, loss1, linewidth=1, linestyle="solid", label="ResNet101 train loss with transfer learning")
#     plt.plot(x_loss, loss2, linewidth=1, linestyle="solid", label="ResNet101 train loss without transfer learning")
#     plt.legend()
#     plt.title('Loss curve')
#     plt.savefig('../result/fig/transfer_learningR50_loss.png')
#     plt.show()
#     # plt.close()
#     print("successful save loss curve! ")
#
# if __name__ == '__main__':
#     json_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # 返回上层目录
#     # json_path1 = json_root + "/work_dirs/gfl_r50_fpn_1x_coco/20240324_203047_r50/vis_data/20240324_203047.json"  # 数据集路径
#     # json_path2 = json_root + "/work_dirs/gfl_r50_fpn_1x_coco/20240325_090357_r50no/vis_data/20240325_090357.json"  # 数据集路径
#
#     json_path1 = json_root + "/work_dirs/gfl_r50_fpn_1x_coco/20240324_203047_r50/vis_data/20240324_203047.json"  # 数据集路径
#     json_path2 = json_root + "/work_dirs/gfl_r50_fpn_1x_coco/20240325_090357_r50no/vis_data/20240325_090357.json"  # 数据集路径
#
#     x = visualize_mmdetection(json_path1)
#     x2 = visualize_mmdetection(json_path2)
#
#     x.load_data()
#     x2.load_data()
#     plot(x.mAP_50, x2.mAP_50)
#     plotloss(x.loss, x2.loss)
#     # x.show_chart()
#
# # detection/MMDetection-main/tools/work_dirs/gfl_r50_fpn_1x_coco/20240324_203047_r50/vis_data/20240324_203047.json
# # detection/MMDetection-main/tools/work_dirs/gfl_r50_fpn_1x_coco/20240325_090357_r50no/vis_data/20240325_090357.json
# # detection/MMDetection-main/tools/work_dirs/gfl_r50_fpn_1x_coco/20240326_151332_r34/vis_data/20240326_151332.json
# # detection/MMDetection-main/tools/work_dirs/gfl_r50_fpn_1x_coco/20240323_095111_noprer101/vis_data/20240323_095111.json
# # detection/MMDetection-main/tools/work_dirs/gfl_r50_fpn_1x_coco/20240515_211323_r50/vis_data/20240515_211323.json