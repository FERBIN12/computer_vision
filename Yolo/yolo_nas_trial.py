# import the needed libs
from super_gradients.training import models
from torchinfo import summary

# initializind the pre-trined model
yolo_nas_1=models.get("yolo_nas_l",pretrained_weights="coco")

summary(model=yolo_nas_1, 
        input_size=(16, 3, 640, 640),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
url = "https://previews.123rf.com/images/freeograph/freeograph2011/freeograph201100150/158301822-group-of-friends-gathering-around-table-at-home.jpg"
yolo_nas_1.predict(url, conf=0.25).show()