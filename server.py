from flask import Flask, request, Response, render_template, send_file
import time
import os

app = Flask(__name__, static_url_path='/static')
app.config['PROPAGATE_EXCEPTIONS'] = True

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/processimage', methods=["POST"])
def processimage():
    now = time.strftime("%Y%m%d-%H%M%S")

    path = 'images/' + now
    os.makedirs(path)

    f = request.files['file']
    f.save(path + '/image.jpg')

    return send_file(path + '/result.jpg', mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
