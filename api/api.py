from fastapi import FastAPI

from config.settings import DataInferenceConfig, DatasetConfig, SlicingConfig

app = FastAPI()


slicingConfig = SlicingConfig(
    slicing_mode="asahi",
    tile_size=(640, 640),
    overlap=128,
    min_object_coverage=0.5
)
dataConfig = DatasetConfig(
    input_path="./dataset",
    output_path="./output",
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)

inferenceConfig = DataInferenceConfig(
        slicing_mode="asahi",
        suppression="nms",
        dataset_path="./dataset",
        models_path="./models",
        output_results_path="./output",
        batch_size=32,
        num_workers=4,
        save_original_annotations=True
    )
    


@app.get("/")
async def read_root():
    return {"Test":"API is running"}

# slicing
@app.post("/slicing/single_image")
async def read_root(img_path):
    isSliced = False
    
    # splita imagem
    
    return isSliced

@app.post("/slicing/dataset")
async def read_root(dataset_path):
    isSliced = False
    
    # splita dataset
    
    return isSliced

# validate
@app.post("/validate/")
async def read_root(dataset_path):
    isValidated = False
    
    # valida dataset
    
    return isValidated   

# utils

@app.post("/imgshow/show_slice_frontiers")
async def read_root(img_path):
    isExibited = False
    # valida dataset
    
    return isExibited   

@app.post("/imgshow/{index_slice}")
async def read_root(img_path):
    pass
