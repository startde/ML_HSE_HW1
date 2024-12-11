from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional
import pickle
from custom_classes import DropDuplicates, DropDuplicateDescriptions, ResetIndex, ExtractNumericValues, TorqueProcessor, ConvertToInt, ProcessName, ReplaceOutliersAndZeros, AddPowerPerEngine, FillNaWithZeros, CustomEncoder, CustomScaler
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder
import warnings
import logging

logging.basicConfig(level=logging.WARNING)

with open('pipeline.pickle', 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

with open('model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

app = FastAPI()

@app.post("/predict_item")
async def predict_item(item: Item):
    try:
        input_df = pd.DataFrame([item.model_dump()])
        logging.info(f"Input data:\n{input_df}")
        
        # Применяем пайплайн
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_transformed = pipeline.named_steps['preprocessor'].transform(input_df)
            for warning in w:
                if issubclass(warning.category, UserWarning):
                    logging.warning(warning.message)
        
        logging.info(f"Transformed data shape: {X_transformed.shape}")
        logging.debug(f"Transformed data:\n{X_transformed}")

        if len(X_transformed.shape) != 2:
            X_transformed = pd.DataFrame(X_transformed).values        

        prediction = pipeline.predict(input_df)
        
        # Преобразуем предсказанные значения обратно к исходному масштабу
        predicted_price = np.exp(prediction[0])

        return {"predicted_price": predicted_price}
    except Exception as e:
        logging.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_transformed = pipeline.named_steps['preprocessor'].transform(df)
            for warning in w:
                if issubclass(warning.category, UserWarning):
                    logging.warning(warning.message)
        
        expected_feature_count = pipeline.named_steps['preprocessor'].transform(df.iloc[:1]).shape[1]
        logging.info(f"Expected feature count: {expected_feature_count}")
        
        if X_transformed.shape[1] != expected_feature_count:
            raise ValueError(f"Mismatch in feature count: Expected {expected_feature_count}, got {X_transformed.shape[1]}")
        
        predictions = pipeline.predict(df)
        
        # Преобразуем предсказанные значения обратно к исходному масштабу
        predicted_prices = np.exp(predictions)
        
        # Добавляем предсказанные цены в DataFrame
        df['predicted_price'] = predicted_prices
        
        # Преобразуем DataFrame обратно в CSV
        csv_content = df.to_csv(index=False)
        
        return Response(content=csv_content, media_type="text/csv")
    except Exception as e:
        logging.error(f"Error during batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")