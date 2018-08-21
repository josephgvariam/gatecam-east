from flask import Flask, request, Response, render_template, send_file
import time
import os
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import shutil

app = Flask(__name__, static_url_path='/static')
app.config['PROPAGATE_EXCEPTIONS'] = True

# Uncomment the line below to provide path to tesseract manually
#pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


args = {
    'min_confidence': 0.5,
    'width': 640,
    'height': 640,
    'east': 'frozen_east_text_detection.pb'
}

@app.route('/')
def hello_world():
    return render_template('index.html')


def copydebugimages(path):
    shutil.rmtree('static/debug')
    os.mkdir('static/debug')
    copied = []

    for root, dirs, files in os.walk(path):
        for currentFile in files:
            exts = ('.jpg')
            if currentFile.lower().endswith(exts):
                shutil.copy(os.path.join(root, currentFile), 'static/debug/')
                copied.append(currentFile)

    return copied

def getdebugimages():
    copied = []

    for root, dirs, files in os.walk('static/debug/'):
        for currentFile in files:
            exts = ('.jpg')
            if currentFile.lower().endswith(exts):
                copied.append(currentFile)

    return copied

@app.route('/debug')
def debug():
    files = getdebugimages()
    return render_template('debug.html', files=files)


@app.route('/processimage', methods=["POST"])
def processimage():
    now = time.strftime("%Y%m%d-%H%M%S")
    path = 'images/' + now
    os.makedirs(path)

    f = request.files['file']
    inputFilePath = path + '/input.jpg'
    f.save(inputFilePath)

    boxes = text_detect(inputFilePath)

    textBoxes = text_recognize(inputFilePath, boxes)
    #print(textBoxes)
    drawTextBoxes(inputFilePath, textBoxes)

    copydebugimages(path)

    resultFilePath = path + '/result.jpg'
    return send_file(resultFilePath, mimetype='image/jpeg')

def drawTextBoxes(inputFilePath, textBoxes):
    img = cv2.imread(inputFilePath)
    for box in textBoxes:
        cv2.rectangle(img, (box['startX'], box['startY']), (box['endX'], box['endY']), (0, 255, 0), 1)
        cv2.putText(img, box['text'], (box['startX'], box['endY'] - box['startY']), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    n = inputFilePath.rfind('/')
    resultFilePath = inputFilePath[:n] + '/result.jpg'
    cv2.imwrite(resultFilePath, img)

def text_recognize(inputFilePath, boxes):
    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    config = ('-l eng --oem 1 --psm 3')

    textBoxes = []
    img = cv2.imread(inputFilePath)

    start = time.time()
    cropCount = 1
    n = inputFilePath.rfind('/')
    cropFilePathPre = inputFilePath[:n]

    for box in boxes:
        crop = img[box['startY'] : box['endY'], box['startX'] : box['endX'], : ]

        cropFilePath = cropFilePathPre + '/crop' + str(cropCount) + '.jpg'
        cv2.imwrite(cropFilePath, crop)

        text = pytesseract.image_to_string(crop, config=config)
        print('crop saved: '+cropFilePath+', text: '+text)

        textBoxes.append({
            'box': box,
            'text': text
        })

        cropCount += 1

    end = time.time()
    print("[INFO] text recognition took {:.6f} seconds".format(end - start))

    return textBoxes


def text_detect(inputFilePath):
    # load the input image and grab the image dimensions
    image = cv2.imread(inputFilePath)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    scaledBoxes = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

        scaledBoxes.append({
            'startX': startX,
            'startY': startY,
            'endX': endX,
            'endY': endY
        })

    # show the output image
    #cv2.imshow("Text Detection", orig)
    #cv2.waitKey(0)

    return scaledBoxes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
