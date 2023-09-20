import json
import webbrowser
from pprint import pprint
from threading import Timer

from flask import Flask, render_template, request, redirect, url_for, jsonify

from data_card_reader_csv_template import DataCardReader

app = Flask(__name__)

# global data
rawdata = None
data_card_metadata = None
data_all = None
train_data = None
test_data = None
provenance = None


@app.route('/datacard')
def index():
    '''
    raw_data_file_location = 'data/data/heart/heart.csv'
    training_data_file_location = 'data/data/heart/heart_train.csv'
    testing_data_file_location = 'data/data/heart/heart_test.csv'

    card = DataCardReader()
    output = card.create(raw_data_file_location, training_data_file_location, testing_data_file_location)
    rawdata = card.create_json(output)
    '''

    global rawdata
    global data_card_metadata
    global data_all
    global train_data
    global test_data
    global provenance

    rawdata = json.load(open('/Users/luoyu/PycharmProjects/DataModelCard/card/data/sample_card_template.json'))

    data_card_metadata = rawdata['data_card_metadata']
    data_all = rawdata['data']
    train_data = rawdata['train_data']
    test_data = rawdata['test_data']
    provenance = rawdata['provenance']

    return render_template('data_card_update.html', datacardmetadata=jsonify(data_card_metadata).json,
                           dataall=jsonify(data_all).json, traindata=jsonify(train_data).json,
                           testdata=jsonify(test_data).json, provenance=jsonify(provenance).json)


@app.route('/update/datacardmetadata', methods=['POST'])
def update_data_card_metadata():
    updated_data = request.json
    global data
    data = updated_data
    print(data)
    return jsonify(success='Success', output=data)

@app.route('/update/data', methods=['POST'])
def update_data():
    updated_data = request.json
    global data
    data = updated_data
    print(data)
    return jsonify(success='Success', output=data)

@app.route('/update/provenance', methods=['POST'])
def update_provenance():
    updated_data = request.json
    global data
    data = updated_data
    print(data)
    return jsonify(success='Success', output=data)

@app.route('/update/traindata', methods=['POST'])
def update_train_data():
    updated_data = request.json
    global data
    data = updated_data
    print(data)
    return jsonify(success='Success', output=data)

@app.route('/update/testdata', methods=['POST'])
def update_test_data():
    updated_data = request.json
    global data
    data = updated_data
    print(data)
    return jsonify(success='Success', output=data)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/datacard')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=5000)