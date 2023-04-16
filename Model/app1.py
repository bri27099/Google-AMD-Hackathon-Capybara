from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index4.html')

@app.route('/capture', methods=['POST'])
def capture():
    image_data = request.form['image']
    encoded_data = image_data.split(',')[1]
    with open('actual1  .jpg', 'wb') as f:
        f.write(encoded_data.decode('base64'))
    return 'Image saved successfully'

if __name__ == '__main__':
    app.run(debug=True)
