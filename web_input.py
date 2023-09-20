from flask import Flask, render_template, request, jsonify
import webbrowser
from threading import Timer


from data_card_reader_csv_template import DataCardReader

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def data_card_input_page():
    return render_template('datainput.html')

@app.route('/update', methods=['POST'])
def update_data():
    global data
    data = request.json
    return jsonify(success=True)

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        raw_data_file_location = request.form['rawdata']
        training_data_file_location = request.form['trainingdata']
        testing_data_file_location = request.form['testingdata']

        raw_data_file_location = 'data/data/heart/heart.csv'
        training_data_file_location = 'data/data/heart/heart_train.csv'
        testing_data_file_location = 'data/data/heart/heart_test.csv'


        card = DataCardReader()
        output = card.create(raw_data_file_location, training_data_file_location, testing_data_file_location)
        output = card.create_json(output)
        print(output)
        return render_template('dataoutputupdatev2.html',enumerate=enumerate, data=output)


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=5000)
