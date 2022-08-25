#!/usr/bin/env python
# coding: utf-8

# In[9]:


from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np


# In[10]:


app = Flask(__name__)

model=pickle.load(open('C:/Users/SHIVA/Downloads/model_pickle2.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("C:/Users/SHIVA/Downloads/index.html")
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('C:/Users/SHIVA/Downloads/index.html', prediction_text='Fare Amount should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




