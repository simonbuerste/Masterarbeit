# import necessary packages
from GUI import PhotoGui
from imutils.video import VideoStream
import argparse
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
                help="path to output directory to store snapshots")

args = vars(ap.parse_args())

# initialize video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=True, resolution=(224,224)).start()
time.sleep(2.0)

#start the app
app = PhotoGui(vs, args["output"])
app.root.mainloop()