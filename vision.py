from roboflow import Roboflow
from datetime import date

# ==============================
# ROBOFLOW CONFIGURATION
# ==============================

ROBOFLOW_API_KEY = "19DSNUIvMHxH1Li1RENl"
PROJECT_ID = "plant-diseases-9mchz-i17o4"
MODEL_VERSION = 1

# Initialize model once (important for performance)
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_ID)
model = project.version(MODEL_VERSION).model


# ==============================
# DISEASE SOLUTIONS DATABASE
# ==============================

SOLUTIONS = {
    "Potato Early blight":
        "Apply Mancozeb fungicide every 7 days and remove infected leaves.",
    
    "Potato Late blight":
        "Use Metalaxyl fungicide immediately and avoid excess moisture.",
    
    "Potato healthy":
        "Crop is healthy. Maintain proper irrigation and nutrition.",
    
    "Tomato Leaf Mold":
        "Improve air circulation and apply Chlorothalonil spray.",
    
    "Tomato healthy":
        "Crop is healthy. No treatment required."
}


# ==============================
# MAIN SUPERVISOR CLASS
# ==============================

class SupervisorAgent:

    async def analyze_image_parallel(self, image_path, city, crop):

        try:
            print("🔄 Sending image to Roboflow...")

            result = model.predict(image_path).json()

            print("📦 Raw Prediction:", result)

            # =========================
            # HANDLE NO PREDICTION
            # =========================

            if not result.get("predictions"):
                disease = "Healthy"
                confidence = 0.0
            else:
                prediction = result["predictions"][0]

                # Roboflow classification models use "top"
                disease = prediction.get("top") or prediction.get("class")

                confidence = prediction.get("confidence", 0) * 100

            # =========================
            # SEVERITY LOGIC
            # =========================

            if confidence > 85:
                severity = "High"
            elif confidence > 60:
                severity = "Medium"
            else:
                severity = "Low"

            # =========================
            # SOLUTION FETCH
            # =========================

            solution = SOLUTIONS.get(
                disease,
                "Consult agricultural expert for further inspection."
            )

            # =========================
            # RETURN FINAL STRUCTURE
            # =========================

            return {
                "vision_result": {
                    "disease": disease,
                    "confidence": round(confidence, 2),
                    "severity": severity,
                    "recommendation": solution
                },
                "weather": {
                    "city": city,
                    "temperature": 25,
                    "humidity": 60,
                    "condition": "Clear",
                    "wind_speed": 2.5
                },
                "mandi_prices": {
                    "commodity": crop,
                    "city": city,
                    "date": date.today().isoformat(),
                    "prices": [
                        {
                            "market": "Local Market",
                            "min_price": 1100,
                            "max_price": 1300,
                            "modal_price": 1200
                        }
                    ],
                    "source": "Demo Data"
                }
            }

        except Exception as e:

            print("❌ Error:", e)

            return {
                "vision_result": {
                    "disease": "Unable to analyze",
                    "confidence": 0,
                    "severity": "Unknown",
                    "recommendation": str(e)
                }
            }