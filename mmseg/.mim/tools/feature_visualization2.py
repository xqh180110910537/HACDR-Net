import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = 'feature_map',name = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heat_maps, (224, 224))
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, (320, 320))
                heatmap=cv2.cvtColor(heatmap,cv2.COLOR_RGB2BGR)
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                # cv2.imshow("1", superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                plt.imshow(superimposed_img)
                plt.axis('off')  # 关掉坐标轴
                plt.savefig('./demo/out/2_1.jpg', bbox_inches='tight', pad_inches=0)
                plt.show()
                # cv2.imwrite('./imgs/6.jpg',superimposed_img)
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            heatmap = cv2.resize(heatmap, (320, 320))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')

                # 下面这些是对特征图进行保存，使用时取消注释
                cv2.imshow("1",superimposed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                # i=i+1
