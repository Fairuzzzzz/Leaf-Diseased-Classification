import uvicorn
from fastapi import FastAPI, File, UploadFile
from inference_onnx import ONNXPrediction

app = FastAPI(docs_url="/")

class_mapping = {
    0: "Pepper_Bell_Bacterial_spot",
    1: "Pepper_Bell_Healthy",
    2: "Potato_Early_Blight",
    3: "Potato_Healthy",
    4: "Potato_Late_Blight",
    5: "Tomato_Target_Spot",
    6: "Tomato_Mosaic_Virus",
    7: "Tomato_YellowLeaf_Curl_Virus",
    8: "Tomato_Bacterial_Spot",
    9: "Tomato_Early_Blight",
    10: "Tomato_Healthy",
    11: "Tomato_Late_Blight",
    12: "Tomato_Leaf_Mold",
    13: "Tomato_Septoria_Leaf_Spot",
    14: "Tomato_Spider_Mites_Two_Spotted_Spider_Mites"
}

onnx_model = 'leaf_model.onnx'
leaf_classifier = ONNXPrediction(onnx_model, class_mapping)

@app.post("/prediction/")
async def prediction(file: UploadFile = File(...)):
    contents = await file.read()

    prediction_class_name, prediction_class_prob, prediction_class_idx = (
        leaf_classifier.prediction(contents)
    )
    return {
        "Predicted Class Index": prediction_class_idx,
        "Predicted Class Prob": prediction_class_prob,
        "Predicted Class Names": prediction_class_name
    }

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
