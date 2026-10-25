"""Live webcam overlay: emotion, gaze, top AUs. Run from project root:
   python src/live_demo.py        (press q to quit)"""
import os, cv2, numpy as np, torch
from pathlib import Path
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
W = PROJECT_ROOT / "OpenFace-3.0" / "weights"
os.chdir(W.parent)   # RetinaFace hardcodes torch.load("./weights/mobilenetV1X0.25_pretrain.tar")

class ArrayFaceDetector(FaceDetector):
    """get_face() for BGR ndarrays (webcam/video frames), not just file paths.
    Fixes shipped bug: preprocess_image hardcodes cv2.imread (breaks detect-video)."""
    def preprocess_image(self, image, resize=1.0):
        img_raw = image if isinstance(image, np.ndarray) else cv2.imread(image)
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0).to(self.device), img_raw

EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

print("loading models ...")
det = ArrayFaceDetector(model_path=str(W / "Alignment_RetinaFace.pth"), device="cpu")
mtl = MultitaskPredictor(model_path=str(W / "MTL_backbone.pth"), device="cpu")   # AU/gaze/emotion branch only

cap = cv2.VideoCapture(0)                       # 0 = built-in FaceTime camera
if not cap.isOpened():
    raise SystemExit("could not open camera 0 (check System Settings > Privacy > Camera)")
print("camera open, press q in the video window to quit")
misses = 0
while True:
    ok, frame = cap.read()
    if not ok:                                  # macOS camera often drops the first frames while warming up
        misses += 1
        if misses > 100:
            raise SystemExit("camera opened but returned no frames")
        cv2.waitKey(30)
        continue
    misses = 0
    face, dets = det.get_face(frame)
    if face is not None:
        emo_logits, gaze, aus = mtl.predict(face)
        emo = EMOTIONS[int(torch.argmax(emo_logits, dim=1))]
        aus = aus.squeeze(0)                    # unwrap batch dim: (1,8) -> (8,)
        top = torch.topk(aus, 3)
        x1, y1, x2, y2 = dets[0][:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{emo}  gaze({gaze[0][0]:+.0f},{gaze[0][1]:+.0f})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "AUs " + " ".join(f"#{i}:{v:.2f}" for i, v in
                    zip(top.indices.tolist(), top.values.tolist())),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("openface live  (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()