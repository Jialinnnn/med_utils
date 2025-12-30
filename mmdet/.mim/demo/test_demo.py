# from mmdet.apis import init_detector, inference_detector
# # model.show_result
# # Choose to use a config
# config_path = './configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
#
# # Setup a checkpoint file to load
# checkpoint = './weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
#
# model = init_detector(config_path, checkpoint, device='cuda:0')
#
# img = 'demo.jpg'
# result = inference_detector(model, img)
# model.show_result(model, img, result)

import mmdet
print(mmdet.__version__)