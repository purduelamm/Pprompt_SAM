import cv2 as cv
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def sam_loader(checkpoint: str, model_type: str, device: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    mask_predictor = SamPredictor(sam_model=sam)

    return mask_predictor


def segmentation_visualizer(dst: np.array, masks: np.array, blend: float = 0.5) -> None:
    cyan = np.full_like(dst,(255,255,0))
    img_cyan = cv.addWeighted(dst, blend, cyan, 1-blend, 0)

    c,h,w = masks.shape
    masks = 255 * masks.astype(np.uint8).reshape(h,w,c)
    result = np.where(masks==255, img_cyan, dst)

    cv.imshow('result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()   

    return
