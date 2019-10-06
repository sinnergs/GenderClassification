import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model._make_predict_function()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    def result(prediction):
        if prediction == 0:
            return "Female"
        else:
            return "Male"

    def word2vec(urname):
        char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g',
                    'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']
        char2idx = {}
        index = 0
        for ch in char_set:
            char2idx[ch] = index
            index += 1
        vector_length = 39
        max_word_len = 20
        words = []
        one_hots_word = []

        for ch in urname:
            vec = np.zeros(vector_length)
            vec[char2idx[ch]] = 1
            one_hots_word.append(vec)
        for _ in range(max_word_len - len(urname)):
            vec = np.zeros(vector_length)
            vec[char2idx['END']] = 1
            one_hots_word.append(vec)
        one_hots_word = np.array(one_hots_word)

        return one_hots_word

    features = [x for x in request.form.values()][0]
    print(features)
    ohw_feature = word2vec(features)
    reshaped_feature = ohw_feature.reshape(1, 20, 39)

    prediction = np.argmax(model.predict(reshaped_feature))

    output = result(prediction)

    return render_template('index.html', result="The following name is of {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
