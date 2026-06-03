from ultralytics import YOLO

def train():
    model = YOLO("yolo26n.pt")
    model.train("data")
    
if __name__ == "__main__":
    train()