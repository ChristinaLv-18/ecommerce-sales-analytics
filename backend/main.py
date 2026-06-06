from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class item(BaseModel):
    x: float
    y: float
@app.post("/sum")
def calc_sum(item: item):
    res= item.x + item.y
    return{"result": res}
