# coding=utf-8

import RSCCore as core
from flask import Flask
import json
from flask import request
from flask import jsonify

__app__ = Flask(__name__)


@__app__.route("/getSuggestions", methods=["POST"])
def get_suggestions():
    if request.method == "POST":
        content = request.get_data()
        content = content.decode("utf8")
        content_dir = json.loads(content)
        #Per una questione di efficienza non sono state cambiati i nomi della variabili, di seguito l'associazione che è stata usata
        #alla varabile movies corrisponde books
        #alla varabile entities corrisponde entities
        #alla varabile moviestoIgnore corrisponde booktoIgnore
        print(content_dir)
        suggestions = core.get_suggestion(preferences_IDs=content_dir["movies"],
                                              pref_entity=content_dir["entities"],
                                              movie_to_ignore=content_dir["movietoIgnore"],
                                              negative_entity=content_dir["negativeEntity"],
                                              pref_aspects = content_dir["prefAspects"],
                                              neg_aspects = content_dir["negAspects"],
                                              rec_list_size= content_dir["recListSize"])
        #print("book", content_dir["movies"])
        #print("bookIGNORE", content_dir["movietoIgnore"])
        #print("entities", content_dir["entities"])
        #print("negativeEntity", content_dir["negativeEntity"])
        return jsonify(results=suggestions)


@__app__.route("/selectModel/<int:selected_model>")
def select_model(selected_model):
    status = core.select_model(selected_model=selected_model)
    if status == 200:
        return "OK"
    else:
        return "No model loaded"


@__app__.route("/updateDataset", methods=["POST"])
def updateDataset():
    if request.method == "POST":
        content = request.get_data()
        try:
            content = json.loads(content)
            title = content["title"]
            ID = content["ID"]
            plot = content["plot"]
            cast = content["cast"]
            genres = content["genres"]
            directors = content["directors"]
            if core.update_dataset(ID=ID, title=title, plot=plot, cast=cast, genres=genres, directors=directors) == 400:
                return "Dataset not updated"
        except Exception:
            return "Format Error"
        return "OK"


@__app__.route("/getSuggestionsFromSentence", methods=["POST"])
def getSuggestionsFromSentence():
    if request.method == "POST":
        content = request.get_data()
        content = content.decode("utf8")
        content_dir = json.loads(content)
        try:
            evaluate_sim_word = content_dir["expand"]
            suggestions = core.get_suggestions_from_sentence(sentences=content_dir["sentences"],
                                                             pref_entity=content_dir["entities"],
                                                             evaluate_sim_word=evaluate_sim_word)
            return jsonify(results=suggestions)
        except Exception:
            try:
                suggestions = core.get_suggestions_from_sentence(sentences=content_dir["sentences"],
                                                                 pref_entity=content_dir["entities"],
                                                                 evaluate_sim_word=False
                                                                 )
                return jsonify(results=suggestions)
            except Exception:
                suggestions = "Format Error"
                return jsonify(results=suggestions)


if __name__ == '__main__':
    __app__.run(host='0.0.0.0')
