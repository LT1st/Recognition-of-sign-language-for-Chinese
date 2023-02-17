from Deduction import Deduction
from Camera import Camera
import json

# with open('dictionary.json', 'r') as f:
#     dictionary = json.load(f)

if __name__ == "__main__":
    camera = Camera(Deduction('./3dcnn_20.pth'))
    camera.work()

    # model = Deduction('./3dcnn_20.pth')
    # print(model.identification('./input'))
