import sys
from pathlib import Path
import cv2
import numpy as np
from zed_yolo_sam2_icp_reconstruct import select_target
from zed_yolo_sam2_live import ROOT, load_yolo, load_sam2
p=Path(sys.argv[1]); bgr=cv2.imread(str(p)); y=load_yolo(ROOT/'best.pt'); s=load_sam2(ROOT/'sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml',ROOT/'sam2/checkpoints/sam2.1_hiera_tiny.pt')
r=y.predict(bgr,conf=.65,verbose=False)[0]; x=select_target(r,s,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),'Wing and Body')
if x is None: raise SystemExit('No target found')
m=x[3]; out=ROOT/'datasets/remote_capture/masks'/p.with_suffix('.png').name; out.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(out),(m*255).astype(np.uint8)); o=bgr.copy(); o[m]=(o[m]*.5+np.array((0,255,0))*.5).astype(np.uint8); cv2.imwrite(str(ROOT/'datasets/remote_capture/preview'/p.name),o); print(out)
