from roboflow import Roboflow

rf = Roboflow(api_key="RXPrptY6yay3qGSPPXU6")

project = rf.workspace("abdelaadimkhriss").project("ocean-plastics-waste-detection-float-plastics")

dataset = project.version(13).download("yolov8", location="./dataset")

print("Done! Dataset saved to:", dataset.location)