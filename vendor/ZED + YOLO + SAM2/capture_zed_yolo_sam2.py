import os
import cv2
import torch
import numpy as np

import pyzed.sl as sl

from ultralytics import YOLO


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



# =====================================================
# Mask 后处理
# =====================================================


def postprocess_mask(mask):

    mask = (
        mask.astype(np.uint8)
    ) * 255


    # --------------------------
    # 最大连通区域
    # --------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )


    if num_labels > 1:

        largest = (
            1 +
            np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
        )

        mask = np.where(
            labels == largest,
            255,
            0
        ).astype(np.uint8)



    # --------------------------
    # 填洞
    # --------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7,7)
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )



    # --------------------------
    # 去毛刺
    # --------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3,3)
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )



    # 平滑

    mask=cv2.GaussianBlur(
        mask,
        (3,3),
        0
    )


    _,mask=cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )


    return mask.astype(bool)




# =====================================================
# 保存目录
# =====================================================


save_root="my_zed_data"


rgb_dir=os.path.join(
    save_root,
    "rgb"
)


depth_dir=os.path.join(
    save_root,
    "depth"
)


mask_dir=os.path.join(
    save_root,
    "masks"
)


os.makedirs(rgb_dir,exist_ok=True)
os.makedirs(depth_dir,exist_ok=True)
os.makedirs(mask_dir,exist_ok=True)




# =====================================================
# YOLO
# =====================================================


yolo_model=YOLO(
    r"C:\Users\24954\Desktop\BundleSDF-master\runs\detect\train-4\weights\best.pt"
)



# =====================================================
# SAM2
# =====================================================


device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



sam2=build_sam2(
    r"C:\Users\24954\sam2\sam2\configs\sam2.1\sam2.1_hiera_t.yaml",

    r"C:\Users\24954\sam2\checkpoints\sam2.1_hiera_tiny.pt",

    device=device
)


predictor=SAM2ImagePredictor(
    sam2
)




# =====================================================
# ZED Camera
# =====================================================


zed=sl.Camera()



init=sl.InitParameters()


# 分辨率
init.camera_resolution=sl.RESOLUTION.HD720


# FPS

init.camera_fps=30



# 深度模式
init.depth_mode=sl.DEPTH_MODE.PERFORMANCE


# 单位 mm

init.coordinate_units=sl.UNIT.MILLIMETER




status=zed.open(init)



if status != sl.ERROR_CODE.SUCCESS:

    print(
        "ZED open failed",
        status
    )

    exit()



print(
    "ZED opened"
)



# =====================================================
# 获取内参
# =====================================================


camera_info=zed.get_camera_information()


calib=camera_info.camera_configuration.calibration_parameters



fx=calib.left_cam.fx
fy=calib.left_cam.fy

cx=calib.left_cam.cx
cy=calib.left_cam.cy



K=np.array(
[
    [fx,0,cx],
    [0,fy,cy],
    [0,0,1]
],
dtype=np.float32
)



np.savetxt(
    os.path.join(
        save_root,
        "cam_K.txt"
    ),
    K,
    fmt="%.6f"
)



print(
    "Camera intrinsic saved"
)


print(K)



# =====================================================
# ZED Mat
# =====================================================


image_zed=sl.Mat()

depth_zed=sl.Mat()



index=0



# =====================================================
# 主循环
# =====================================================


while True:


    if zed.grab()==sl.ERROR_CODE.SUCCESS:



        # --------------------------
        # RGB
        # --------------------------

        zed.retrieve_image(
            image_zed,
            sl.VIEW.LEFT
        )


        image=image_zed.get_data()


        # BGRA -> BGR

        image=cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )



        # --------------------------
        # Depth
        # --------------------------


        zed.retrieve_measure(
            depth_zed,
            sl.MEASURE.DEPTH
        )


        depth=depth_zed.get_data()



        depth=np.asarray(
            depth,
            dtype=np.float32
        )


        # NaN处理

        depth[np.isnan(depth)] = 0




        image_rgb=cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )



        # =================================================
        # YOLO
        # =================================================


        results=yolo_model.predict(

            source=image,

            conf=0.7,

            verbose=False
        )



        boxes=results[0].boxes.xyxy.cpu().numpy()



        output=image.copy()



        final_mask=np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )



        if len(boxes)>0:



            predictor.set_image(
                image_rgb
            )


            colors=[
                (255,0,0),
                (0,255,0),
                (0,0,255),
                (255,255,0),
                (255,0,255)
            ]



            for i,box in enumerate(boxes):



                masks, scores, logits = predictor.predict(

                    point_coords=None,

                    point_labels=None,

                    box=box[None,:],

                    multimask_output=False
                )



                mask=np.squeeze(
                    masks[0]
                )


                mask=postprocess_mask(
                    mask
                )



                final_mask[mask]=255



                color=colors[
                    i%len(colors)
                ]



                x1,y1,x2,y2=map(
                    int,
                    box
                )



                cv2.rectangle(

                    output,

                    (x1,y1),

                    (x2,y2),

                    color,

                    2
                )



                output[mask]=(
                    output[mask]*0.5+
                    np.array(color)*0.5
                ).astype(
                    np.uint8
                )



        # =================================================
        # 显示
        # =================================================


        cv2.imshow(
            "ZED YOLO SAM2",
            output
        )



        key=cv2.waitKey(1)



        # =================================================
        # 保存
        # =================================================


        if key==ord('s'):


            name=f"{index:06d}.png"



            cv2.imwrite(
                os.path.join(
                    rgb_dir,
                    name
                ),
                image
            )



            # 保存16bit深度

            depth_save=np.clip(
                depth,
                0,
                65535
            ).astype(
                np.uint16
            )



            cv2.imwrite(

                os.path.join(
                    depth_dir,
                    name
                ),

                depth_save
            )



            cv2.imwrite(

                os.path.join(
                    mask_dir,
                    name
                ),

                final_mask
            )



            print(
                "Saved",
                name
            )



            index+=1



        if key==27:
            break



zed.close()


cv2.destroyAllWindows()
