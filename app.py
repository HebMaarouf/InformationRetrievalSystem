from flask import Flask, render_template, request
from search_dataset import search_arabic_dataset, search_english_dataset

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # get the search query from the request
    search_text = request.form['search_text']

    # get the desired dataset from the request
    dropdown_value = request.form['dropdown']

    # perform the search operation in the selected dataset
    precision,related_docs=  search_english_dataset(search_text) if dropdown_value=="en" else search_arabic_dataset(search_text)
    
    # modify the data structure
    modified_data = [(index, key, value) for index, (key, value) in enumerate(related_docs.items())]

    # calcualte the number of relevant docs
    total_length = len(related_docs)

    # return the result to the page to let the user view it
    return render_template('result.html', precision=precision,result_dict =modified_data,total_length=total_length)

if __name__ == '__main__':
    app.run()