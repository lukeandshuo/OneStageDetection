from util import Anchors,Bboxes
import numpy as np



def postprocessing_fn(classifications,localizations,img_size):
    anchors_bboxes = Anchors.multibox_anchors(img_size=img_size)
    classes,scores,bboxes = Bboxes.bboxes_select(classifications,localizations,anchors_bboxes,select_threshold=0.5)
    ref_bboxes = np.array([0.,0.,1.,1.])
    bboxes = Bboxes.bboxes_clip(ref_bboxes,bboxes)
    classes,scores,bboxes = Bboxes.bboxes_sort(classes,scores,bboxes,top_k=400)
    classes,scores,bboxes = Bboxes.bboxes_nms(classes,scores,bboxes, nms_threshold=0.45)

    return classes,scores,bboxes
