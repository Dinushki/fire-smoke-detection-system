import cv2
import numpy as np
import os

# ---------- INPUT IMAGE ----------
img_path = "data/Assets/fire_1.jpg"
  # change image here

image = cv2.imread(img_path)
if image is None:
    print("❌ Image not loaded")
    exit()

image = cv2.resize(image, (640, 480))
output = image.copy()

# ---------- PREPROCESSING ----------
# ---------- ADVANCED PREPROCESSING ----------

# Convert to LAB (better lighting handling)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)

lab = cv2.merge((l, a, b))
enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Bilateral filter (noise removal but keep edges)
filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

# Convert to HSV
hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

# Split channels
h, s, v = cv2.split(hsv)


detected_type = "Normal"
detected = False

# =====================================================
# FIRE DETECTION (Color-based)
# =====================================================

lower_fire = np.array([10, 120, 150])
upper_fire = np.array([35, 255, 255])
fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

# Remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

fire_contours, _ = cv2.findContours(
    fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in fire_contours:
    area = cv2.contourArea(cnt)

    if area > 1000:   # increased threshold
        x, y, w, h = cv2.boundingRect(cnt)

        # Additional validation: check brightness variance
        roi = v[y:y+h, x:x+w]
        brightness_variance = np.var(roi)

        if brightness_variance > 200:   # fire flickers → high variance
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, "Fire", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            detected_type = "Fire"
            detected = True


# =====================================================
# IMPROVED SMOKE DETECTION
# =====================================================

if not detected:

    # Smoke = low saturation + medium/high brightness
    lower_smoke = np.array([0, 0, 100])
    upper_smoke = np.array([180, 60, 230])
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

    smoke_contours, _ = cv2.findContours(
        smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in smoke_contours:
        area = cv2.contourArea(cnt)

        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)

            # Smoke has low edge density (soft texture)
            edges = cv2.Canny(filtered[y:y+h, x:x+w], 50, 150)
            edge_density = np.sum(edges > 0)

            if edge_density < 8000:
                cv2.rectangle(output, (x, y), (x + w, y + h),
                              (200, 200, 200), 2)
                cv2.putText(output, "Smoke", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 2)
                detected_type = "Smoke"
                detected = True


# =====================================================
# NORMAL IMAGE
# =====================================================
if not detected:
    cv2.putText(output, "Normal Image", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

# ---------- DISPLAY ----------
cv2.imshow("Fire & Smoke Detection System", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- SAVE ----------
os.makedirs("outputs", exist_ok=True)
filename = os.path.basename(img_path)
cv2.imwrite(f"outputs/result_{filename}", output)

print(f"✅ Classification Result: {detected_type}")
