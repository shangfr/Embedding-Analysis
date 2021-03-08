# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:10:22 2021

@author: shangfr
"""
import os
import json
from flask import Flask, jsonify,render_template,request

from word_net import get_word_net
#from flask_cors import CORS

app = Flask(__name__)
#CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/v1.0/keywords', methods=['GET','POST'])
def get_keydic():
        if request.method == 'GET':
            file_name = 'keywords/keywords.json'
            with open(file_name,'r',encoding='utf8')as fp:
                json_data = json.load(fp)
            return jsonify({'data': json_data})

@app.route('/api/v1.0/wordfrq', methods=['GET','POST'])
def get_wordfrq():
        if request.method == 'GET':
            file_name = 'keywords/word_frq.json'
            with open(file_name,'r',encoding='utf8')as fp:
                json_data = json.load(fp)
            return jsonify(json_data)
        
    
@app.route('/api/v1.0/wordnet', methods=['GET','POST'])
def get_policy():
        if request.method == 'GET':
            file_name = 'keywords/policy.json'
            with open(file_name,'r',encoding='utf8')as fp:
                json_data = json.load(fp)
            return jsonify({'data': json_data})
        #else:
        elif request.method == 'POST':
            postdata = json.loads(request.get_data(as_text=True))
            json_data = get_word_net(postdata)
            return jsonify({'data': json_data})       

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)