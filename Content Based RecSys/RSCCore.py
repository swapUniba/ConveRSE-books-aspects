# coding=utf-8

import csv
import queue
import threading

import gensim.parsing.preprocessing as __pp__
import numpy
import pandas as pd
#import spacy

import Models.Doc2Vec.Doc2Vec as __d2v__
import Models.TFIDF.TFIDF as __tfidf__
import Models.Word2Vec.Word2Vec as __w2v__

global __doc2vec__, __most_similar__
global __word2vec__, __w2c_pre_trained__
global __fasttext__, __ft_pre_trained__
global __tfidf_model__, __tfidf_index__, __tfidf_dictionary__
global __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__
global __id_Model__  # il numero che identifica il modello selezionato
global __returned_queue__  # returned_queue.get()
global __queue_nlp__, __npl__, __local_w2v__, __started_thread__

__CUSTOM_FILTERS__ = [lambda x: x.lower(), __pp__.strip_tags,
                      __pp__.strip_punctuation,
                      __pp__.remove_stopwords,
                      __pp__.split_alphanum,
                      __pp__.strip_multiple_whitespaces]


def __preprocessing__(trama):
    pp_trama = __pp__.preprocess_string(trama, __CUSTOM_FILTERS__)
    return pp_trama


def __pre_load__():
    global __queue_nlp__, __npl__, __local_w2v__, __started_thread__
    __started_thread__ = True
    global __queue_nlp__
    try:
        if __local_w2v__ is not None and __npl__ is not None:
            return
        else:
            raise Exception
    except Exception:
        #nlp = spacy.load("en_core_web_sm")
        w2v = __w2v__.load_model(None, None, True)
        #__queue_nlp__.put([nlp, w2v])
        return


def __update_file__(index, append):
    global __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__
    if append:
        with open('Dataset/books_info.csv', "a", newline='', encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            # ID,Title,Tokens,Genres,Cast,Directors
            writer.writerow(
                [__books_IDs__[index], __books_titles__[index], __tokenized_description__[index], __films_genres__[index],
                 __books_subjects__[index], __books_authors__[index]])
            csvfile.close()
            return 200
    with open('Dataset/books_info.csv', "r+", newline='', encoding="utf8") as csvfile:
        fieldnames = ['ID', 'Title', "Tokens"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, (ID, title, plot, genres, cast, directors) in enumerate(
                zip(__books_IDs__, __books_titles__, __tokenized_description__, __films_genres__, __books_subjects__,
                    __books_authors__)):
            writer.writerow(
                {"ID": ID, "Title": title, "Tokens": plot, "Genres": genres, "Cast": cast, "Directors": directors})
            if i > index:
                csvfile.close()
                return 200
    csvfile.close()


# Questa funzione ci permette di caricare in memoria l'intero dataset dataset diviso in token.
# Questa funzione va sostituita qualora si decidesse di non utilizzare il file già presente.
# Per sicurezza effettua il processing qualora
# non risulti diviso in token
def __tonkens_from_documents_gensim__():
    documents = []
    genres = []
    titles = []
    authors = []
    subjects = []
    n_features = []
    pp_docs = []
    IDs = []
    with open('Dataset/books_info.csv', newline='', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Description = row["Description"]
            Title = row["Title"]
            IDs.append(row["ID"])
            titles.append(Title)
            new_tokens = []
            raw_subjects = row["Subjects"].splitlines()
            raw_genres = row["Genres"].splitlines()
            raw_authors = row["Authors"].splitlines()
            book_subjects = list()
            film_genres = list()
            film_authors = list()
            for listcast in raw_subjects:
                row_IDS = listcast.split(",")
                for ID in row_IDS:
                    book_subjects.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            subjects.append(book_subjects)
            for listgeneres in raw_genres:
                row_IDS = listgeneres.split(",")
                for ID in row_IDS:
                    film_genres.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            genres.append(film_genres)
            for listdir in raw_authors:
                row_IDS = listdir.split(",")
                for ID in row_IDS:
                    film_authors.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            authors.append(film_authors)
            # for ID in raw_authors:
            #     authors.append(str(ID.replace(" ", "").replace("""'""", """""")))
            for token in Description.split(','):
                new_tokens.append(token.replace("""'""", """"""))
            n_features.append(len(new_tokens))
            documents.append(Description)
            pp_docs.append(__preprocessing__(Description))
    csvfile.close()
    return pp_docs, IDs, titles, subjects, genres, authors


def __tonkens_from_documents_gensim__():
    documents = []
    genres = []
    titles = []
    subjects = []
    n_features = []
    pp_docs = []
    IDs = []
    aspects = []
    authors = []
    with open('Dataset/books_info.csv', newline='', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Description = row["Description"]
            Title = row["Title"]
            IDs.append(row["ID"])
            titles.append(Title)
            new_tokens = []
            raw_subjects = row["Subjects"].splitlines()
            raw_genres = row["Genres"].splitlines()
            raw_authors = row["Authors"].splitlines()
            raw_aspects = row["Aspects"].splitlines()
            book_subjects = list()
            book_genres = list()
            books_authors = list()
            books_asp = list()
            for listcast in raw_subjects:
                row_IDS = listcast.split(",")
                for ID in row_IDS:
                    book_subjects.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            subjects.append(book_subjects)
            for listgeneres in raw_genres:
                row_IDS = listgeneres.split(",")
                for ID in row_IDS:
                    book_genres.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            genres.append(book_genres)
            for listdir in raw_authors:
                row_IDS = listdir.split(",")
                for ID in row_IDS:
                    books_authors.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            authors.append(books_authors)
            for listasp in raw_aspects:
                row_IDS = listasp.split(",")
                for ID in row_IDS:
                    # books_asp è la lista di aspetti per un libro
                    books_asp.append(ID.replace(" ", "").replace("""'""", """""").replace("[", "").replace("]", ""))
            # lista di ogni film e per ognuno contiene la lista dei suoi aspetti
            aspects.append(books_asp)
            for token in Description.split(','):
                new_tokens.append(token.replace("""'""", """"""))
            n_features.append(len(new_tokens))
            documents.append(Description)
            pp_docs.append(__preprocessing__(Description))
    csvfile.close()
    return pp_docs, IDs, titles, subjects, genres, authors, aspects


# Funzione esposta che permette di selezionare il modello con cui ottenere risultati. I valori di selected_model sono:
# 1 per usare Doc2Vec con il metodo most_similar.
# 2 per usare Doc2Vec la similarità è con il centroide.
# 3 per usare Word2Vec per utilizzare un modello pre-addestrato (word2vec-google-news-300).
# 4 per usare Word2Vec.
# 5 per usare FastText per utilizzare un modello pre-addestrato (cc.en.300.bin).
# 6 per usare FastText.
# 7 per usare tfidf.
def select_model(selected_model):
    global __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__
    global __id_Model__
    global __returned_queue__
    global __queue_nlp__, __npl__, __local_w2v__, __started_thread__
    __queue_nlp__ = queue.Queue()
    __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__ = __tonkens_from_documents_gensim__()
    # Selezione del modello DOC2VEC
    if selected_model == 1 or selected_model == 2:
        global __doc2vec__, __most_similar__
        try:
            if __doc2vec__ is not None:
                print("Already Loaded")  # Si evita di ricaricare il modello
            else:
                raise Exception
        except Exception:
            __doc2vec__ = None
            __returned_queue__ = queue.Queue()
            thread = threading.Thread(target=__d2v__.load_model,
                                      args=(__tokenized_description__, "Models/Doc2Vec/doc2vec_model", __returned_queue__))
            thread.start()
        if selected_model == 1:
            __most_similar__ = True
        else:
            __most_similar__ = False
        __id_Model__ = selected_model
        try:
            if __started_thread__:
                return 200
            else:
                raise Exception
        except Exception:
            print('')
            #threading.Thread(target=__pre_load__).start()
        return 200
    # Selezione del modello WORD2VEC
    if selected_model == 3 or selected_model == 4:
        global __word2vec__, __w2c_pre_trained__
        if selected_model == 3:
            __w2c_pre_trained__ = True
        else:
            __w2c_pre_trained__ = False
        try:
            if __word2vec__ is not None and __id_Model__ == selected_model:
                print("Already Loaded")  # Si evita di ricaricare il modello
            else:
                raise Exception
        except Exception:
            __word2vec__ = None
            __returned_queue__ = queue.Queue()
            thread = threading.Thread(target=__w2v__.load_model,
                                      args=(__tokenized_description__, "Models\Word2Vec\word2vec_model",
                                            __w2c_pre_trained__, __returned_queue__))
            thread.start()
        __id_Model__ = selected_model
        try:
            if __started_thread__:
                return 200
            else:
                raise Exception
        except Exception:
            print('')
           # threading.Thread(target=__pre_load__).start()
        return 200
    # Selezione del modello FASTTEXT
    if selected_model == 5 or selected_model == 6:
        global __fasttext__, __ft_pre_trained__
        __returned_queue__ = queue.Queue()
        if selected_model == 5:
            __ft_pre_trained__ = True
            try:
                if __fasttext__ is not None and selected_model == __id_Model__:
                    print("Already Loaded")  # Si evita di ricaricare il modello
                else:
                    raise Exception
            except Exception:
                __fasttext__ = None
                thread = threading.Thread(target=__ft__.create_model_fasttext_fb,
                                          args=(__fasttext__, __returned_queue__))
                thread.start()
        else:
            __ft_pre_trained__ = False
            try:
                if __fasttext__ is not None and selected_model == __id_Model__:
                    print("Already Loaded")
                else:
                    raise Exception
            except Exception:
                __fasttext__ = None
                thread = threading.Thread(target=__ft__.load_model, args=(__tokenized_description__, "Models/FastText"
                                                                                               "/fasttext_model",
                                                                          __returned_queue__))
                thread.start()
        __id_Model__ = selected_model
        try:
            if __started_thread__:
                return 200
            else:
                raise Exception
        except Exception:
            print('')
            #threading.Thread(target=__pre_load__).start()
        return 200
    # Selezione del modello __tfidf__
    if selected_model == 7:
        global __tfidf_model__, __tfidf_index__, __tfidf_dictionary__
        try:
            if __tfidf_model__ is not None and __tfidf_index__ is not None and __tfidf_dictionary__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __tfidf_model__ = None
            __tfidf_index__ = None
            __tfidf_dictionary__ = None
            __returned_queue__ = queue.Queue()
            thread = threading.Thread(target=__tfidf__.load_model,
                                      args=(__tokenized_description__, "Models/TFIDF/tfidf_model",
                                            "Models/TFIDF/matrix_tfidf",
                                            "Models/TFIDF/dictionary_tfidf",
                                            __returned_queue__))
            thread.start()
        __id_Model__ = selected_model
        try:
            if __started_thread__:
                return 200
            else:
                raise Exception
        except Exception:
            print('')
            #threading.Thread(target=__pre_load__).start()
        return 200
    else:
        __tokenized_description__ = None
        __books_titles__ = None
        __books_IDs__ = None
        return 404  # MODELLO NON TROVATO


def get_suggestions_from_sentence(sentences, evaluate_sim_word, pref_entity):
    recommends_from_senteces = []
    if len(sentences) > 0:
        recommends_from_senteces = __get_suggestion_from_sentence__(senteces=sentences,
                                                                    evaluate_sim_word=evaluate_sim_word)
    else:
        for i in range(len(__books_IDs__)):
            recommends_from_senteces.append({"Rank": i + 1, "ID": __books_IDs__[i], "Value": 0})
    list_value = []
    list_IDs = []
    value_cos_temp = list()
    recommends_from_entity = list()
    if len(pref_entity) > 0:
        for i in range(len(__books_IDs__)):
            value_cos_temp.append(recommends_from_senteces[i]["Value"] + 0.1)
        recommends_from_entity = __get_suggestion_from_entity__(pref_entity, films_IDs=__books_IDs__,
                                                                films_cast=__books_subjects__,
                                                                films_genres=__films_genres__,
                                                                films_directors=__books_authors__,
                                                                film_values=value_cos_temp)
    else:
        for i in range(len(__books_IDs__)):
            recommends_from_entity.append({"Rank": i + 1, "ID": __books_IDs__[i], "Value": 0})
    if len(value_cos_temp) == 0:
        for i in range(len(__books_IDs__)):
            if recommends_from_senteces[i]["ID"] == recommends_from_entity[i]["ID"]:
                list_IDs.append(recommends_from_senteces[i]["ID"])
                list_value.append(
                    recommends_from_senteces[i]["Value"] + recommends_from_entity[i]["Value"])
    else:
        for i in range(len(__books_IDs__)):
            list_IDs.append(recommends_from_entity[i]["ID"])
            list_value.append(recommends_from_entity[i]["Value"])
    value, IDs = zip(*sorted(zip(list_value, list_IDs), reverse=True))
    recommends_entity = list()
    output_to_print = list()
    for i in range(5):
        recommends_entity.append({"Rank": len(recommends_entity) + 1, "ID": IDs[i]})
        output_to_print.append([i + 1, __books_titles__[__books_IDs__.index(IDs[i])], value[i], IDs[i]])
    print("--------------" + str(__id_Model__) + "--------------")
    df = pd.DataFrame(output_to_print, columns=["rank", "title", "cosine_similarity", "ID"])
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    return recommends_entity


# Funzionalità esposta. In input c'è una lista di IDs per cui è presente una preferenza.
# NB: Anche se è una preferenza, deve essere una lista.
# In caso non sia sata chiamata la funzione select_model() allora __film_IDs è null quindi si sollega l'eccezione per
# cui sarà restituito "ERROR_FILM_NOT_FOUND"
def get_suggestion(preferences_IDs, pref_entity, movie_to_ignore, negative_entity, pref_aspects, neg_aspects,
                   rec_list_size):
    print('ASPETTI: \n')
    print(pref_aspects)
    print('FINE ASPETTI')
    IDs_pref = list()
    tokenized_pref = list()
    if len(preferences_IDs) + len(pref_entity) == 0:
        return 400
    for id in sorted(preferences_IDs):
        try:
            index = __books_IDs__.index(id)
            IDs_pref.append(id)
            tokenized_pref.append(__tokenized_description__[index])
        except Exception:
            print(id, ' non trovato')
            continue
    recommends_from_movie = []
    recommends_from_entity = []
    recommends_from_aspects = []
    if len(preferences_IDs) > 0:
        recommends_from_movie = __get_suggestion_from_movie__(IDs_pref, tokenized_pref)
    else:
        for i in range(len(__books_IDs__)):
            recommends_from_movie.append({"Rank": i + 1, "ID": __books_IDs__[i], "Value": 0})
    value_cos_temp = []
    if len(pref_aspects) > 0 or len(neg_aspects) > 0:
        for i in range(len(__books_IDs__)):
            value_cos_temp.append(recommends_from_movie[i]["Value"] + 0.1)
        recommends_from_aspects = __get_suggestion_from_aspects__(aspects=pref_aspects, films_IDs=__books_IDs__,
                                                                  film_values=value_cos_temp,
                                                                  neg_aspects=neg_aspects,
                                                                  film_aspects=__books_aspects__)
    else:
        recommends_from_aspects = recommends_from_movie

    value_cos_temp = []
    if len(pref_entity) > 0 or len(negative_entity) > 0:
        for i in range(len(__books_IDs__)):
            value_cos_temp.append(recommends_from_aspects[i]["Value"] + 0.1)
        recommends_from_entity = __get_suggestion_from_entity__(pref_entity, films_IDs=__books_IDs__,
                                                                films_cast=__books_subjects__,
                                                                films_genres=__films_genres__,
                                                                films_directors=__books_authors__,
                                                                film_values=value_cos_temp,
                                                                neg_entities=negative_entity)
    else:
        for i in range(len(__books_IDs__)):
            recommends_from_entity.append({"Rank": i + 1, "ID": __books_IDs__[i], "Value": 0})
    list_value = []
    list_IDs = []
    if len(value_cos_temp) == 0:
        for i in range(len(__books_IDs__)):
            if recommends_from_movie[i]["ID"] == recommends_from_entity[i]["ID"]:
                list_IDs.append(recommends_from_movie[i]["ID"])
                list_value.append(
                    recommends_from_movie[i]["Value"] + recommends_from_entity[i]["Value"])
    else:
        for i in range(len(__books_IDs__)):
            list_IDs.append(recommends_from_entity[i]["ID"])
            list_value.append(recommends_from_entity[i]["Value"])
    value, IDs = zip(*sorted(zip(list_value, list_IDs), reverse=True))
    recommends_entity = list()
    output_to_print = list()
    for i in range(rec_list_size + len(IDs_pref) + len(movie_to_ignore)):
        if IDs[i] in IDs_pref or IDs[i] in movie_to_ignore:
            continue
        if len(recommends_entity) == rec_list_size:
            break
        recommends_entity.append({"Rank": len(recommends_entity) + 1, "ID": IDs[i]})
        output_to_print.append(
            [len(recommends_entity) + 1, __books_titles__[__books_IDs__.index(IDs[i])], value[i], IDs[i]])
    print("--------------" + str(__id_Model__) + "--------------")
    df = pd.DataFrame(output_to_print, columns=["rank", "title", "cosine_similarity", "ID"])
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    return recommends_entity


def __calculate_entity_bias__(IDs_pref, recommends):
    ACTOR_BIAS = 0.15
    DIRECTOR_BIAS = 0.20
    GENRE_BIAS = 0.25
    ASPECT_BIAS = 0.17
    global __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__
    global __id_Model__
    mean_value = 0
    for rec in recommends:
        mean_value += rec["Value"]
    mean_value = mean_value / len(recommends)
    entity_cast_pref = list()
    entity_genres_pref = list()
    entity_directors_pref = list()
    entity_aspects_pref = list()
    for ID in IDs_pref:
        film_cast = __books_subjects__[__books_IDs__.index(ID)]
        film_generes = __films_genres__[__books_IDs__.index(ID)]
        film_directors = __books_authors__[__books_IDs__.index(ID)]
        film_aspects = __books_aspects__[__books_IDs__.index(ID)]
        for actor in film_cast:
            entity_cast_pref.append(actor)
        for genre in film_generes:
            entity_genres_pref.append(genre)
        for director in film_directors:
            entity_directors_pref.append(director)
    recommend_movies = list()
    for i, film in enumerate(recommends):
        film_ID = film["ID"]
        film_value = film["Value"]
        film_cast = __books_subjects__[__books_IDs__.index(film_ID)]
        film_genres = __films_genres__[__books_IDs__.index(film_ID)]
        film_directors = __books_authors__[__books_IDs__.index(film_ID)]
        film_aspects = __books_aspects__[__books_IDs__.index(ID)]
        if len(film_cast) > 0 and len(entity_cast_pref) > 0:
            for actor in entity_cast_pref:
                if actor in film_cast:
                    # print("Attore " + actor)
                    film_value += mean_value * ACTOR_BIAS
        if len(film_directors) > 0 and len(entity_directors_pref) > 0:
            for director in entity_directors_pref:
                if director in film_directors:
                    # print("Regista "+ director)
                    film_value += mean_value * DIRECTOR_BIAS
        if len(film_genres) > 0 and len(entity_genres_pref) > 0:
            for genre in entity_genres_pref:
                if genre in film_genres:
                    # print("Genere "+genre)
                    film_value += mean_value * GENRE_BIAS
        if len(film_aspects) > 0 and len(entity_aspects_pref) > 0:
            for aspect in entity_aspects_pref:
                if aspect in film_aspects:
                    # print("Aspetto "+aspect)
                    film_value += mean_value * ASPECT_BIAS
        recommend_movies.append({"Rank": i + 1, "ID": film_ID, "Value": film_value})
    return recommend_movies


# Funzione che effettivamente si occupa di generare le raccomandazioni in base al modello.
# DA NON CHIAMARE
def __get_rec__(IDs_pref, tokenized_pref):
    global __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__
    global __id_Model__
    global __returned_queue__
    if __id_Model__ == 1 or __id_Model__ == 2:
        global __doc2vec__, __most_similar__
        try:
            if __doc2vec__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __doc2vec__ = __returned_queue__.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __d2v__.get_recommendations_doc2vec(token_strings=tokenized_pref, documents=__tokenized_description__,
                                                         titles=__books_titles__, IDs=__books_IDs__,
                                                         modelDoC=__doc2vec__,
                                                         most_similar=__most_similar__, prefIDs=IDs_pref)
    elif __id_Model__ == 3 or __id_Model__ == 4:
        global __word2vec__, __w2c_pre_trained__
        try:
            if __word2vec__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __word2vec__ = __returned_queue__.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __w2v__.get_recommendations_word2vec(token_strings=tokenized_pref, documents=__tokenized_description__,
                                                          titles=__books_titles__, IDs=__books_IDs__,
                                                          modelWord=__word2vec__,
                                                          pretrained=__w2c_pre_trained__, prefIDs=IDs_pref)
    elif __id_Model__ == 5 or __id_Model__ == 6:
        global __fasttext__, __ft_pre_trained__
        try:
            if __fasttext__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __fasttext__ = __returned_queue__.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __ft__.get_recommendations_fastText(token_strings=tokenized_pref, documents=__tokenized_description__,
                                                         titles=__books_titles__, IDs=__books_IDs__,
                                                         modelFastText=__fasttext__,
                                                         pretrained=__ft_pre_trained__, prefIDs=IDs_pref)
    elif __id_Model__ == 7:
        global __tfidf_model__, __tfidf_index__, __tfidf_dictionary__
        try:
            if __tfidf_model__ is not None and __tfidf_index__ is not None and __tfidf_dictionary__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            loaded = __returned_queue__.get()
            __tfidf_model__ = loaded[0]
            __tfidf_index__ = loaded[1]
            __tfidf_dictionary__ = loaded[2]
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __tfidf__.get_recommendations_tfidf(token_strings=tokenized_pref, documents=__tokenized_description__,
                                                         titles=__books_titles__, IDs=__books_IDs__,
                                                         dictionary=__tfidf_dictionary__,
                                                         tfidfmodel=__tfidf_model__, index=__tfidf_index__,
                                                         prefIDs=IDs_pref)
    return recommends


def update_dataset(ID, title, plot, cast, genres, directors):
    append = True
    global __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__
    try:
        if __tokenized_description__ is not None and __books_IDs__ is not None and __books_titles__ is not None:
            print("Movie Info Already Loaded")  # Già caricati in memoria le informazioni sui film
        else:
            raise Exception
    except Exception:
        __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__ = __tonkens_from_documents_gensim__()
    try:
        index = __books_IDs__.index(ID)
        __books_IDs__.remove(ID)
        __books_titles__.remove(__books_titles__[index])
        __tokenized_description__.remove(__tokenized_description__[index])
        __books_subjects__.remove(__books_subjects__[index])
        __films_genres__.remove(__films_genres__[index])
        __books_authors__.remove(__films_genres__[index])
        append = False
    except ValueError:
        index = len(__books_IDs__)
    __tokenized_description__.insert(index, __preprocessing__(plot))
    __books_IDs__.insert(index, ID)
    __books_titles__.insert(index, title)
    __books_subjects__.insert(index, cast)
    __films_genres__.insert(index, genres)
    __books_authors__.insert(index, directors)
    if __update_file__(index, append) == 200:
        return 200
    else:
        return 400  # file non aggiornato


def __get_suggestion_from_movie__(IDs_pref, tokenized_pref):
    recommends = __get_rec__(IDs_pref, tokenized_pref)
    recommends_entity = __calculate_entity_bias__(IDs_pref, recommends)
    return recommends_entity


def __get_suggestion_from_sentence__(senteces, evaluate_sim_word):
    global __queue_nlp__, __npl__, __local_w2v__
    already_loaded = False
    try:
        if __local_w2v__ is not None and __npl__ is not None:
            already_loaded = True
            raise Exception
        else:
            if not evaluate_sim_word and not already_loaded:
                __queue_nlp__ = queue.Queue()
                __pre_load__()
                raise Exception
    except Exception:
        if not already_loaded:
            returned_value = __queue_nlp__.get()
            __npl__ = returned_value[0]
            __local_w2v__ = returned_value[1]
        try:
            complete_words = []
            for sentence in senteces:
                doc = __npl__(sentence)
                nouns = []
                singles= list()
                for ent in doc.ents:
                    singles = singles + str(ent).split(" ")
                    nouns.append(str(ent))
                for token in doc:
                    if str(token) in singles:
                        continue
                    if token.lemma_ in ["film", "movie", "like", "love", "appreciate", "I"] or token.is_stop or token.is_punct:
                        continue
                    if evaluate_sim_word:
                        try:
                            sim_words = __local_w2v__.most_similar(token.lemma_, topn=5)
                        except Exception:
                            continue
                        for word in sim_words:
                            nouns.append(word[0])
                    nouns.append(token.lemma_)
                complete_words.append(nouns)
            print(complete_words)
            recommends = __get_rec__(None, complete_words)
            # recommends = __get_rec__(None, __preprocessing__(sentence))
            return recommends
        except Exception as e:
            print(str(e))
            return 400


def __get_suggestion_from_entity__(entities, films_IDs, films_cast, films_genres, films_directors, film_values, neg_entities):
    recommend_movies = []
    # mean_value = numpy.mean(film_values)
    for i, ID in enumerate(films_IDs):
        sim_value = film_values[i]
        films_entities = numpy.concatenate((films_cast[i], films_genres[i], films_directors[i]))
        for entity in entities:
            if entity in films_entities:
                sim_value += film_values[i] * 0.4
            if entity in neg_entities:
                sim_value -= film_values[i] * 0.4
        recommend_movies.append({"Rank": i + 1, "ID": ID, "Value": sim_value})
    return recommend_movies

def __get_suggestion_from_aspects__(aspects, neg_aspects, films_IDs, film_values, film_aspects):
    recommend_movies = []
    for i, ID in enumerate(films_IDs):
        sim_value = film_values[i]
        for aspect in film_aspects[i]:
            if aspect in aspects:
                sim_value += film_values[i] * 0.4
            if aspect in neg_aspects:
                sim_value -= film_values[i] * 0.4
        recommend_movies.append({"Rank": i + 1, "ID": ID, "Value": sim_value})
    return recommend_movies


if __name__ == '__main__':
    select_model(7)
    __tokenized_description__, __books_IDs__, __books_titles__, __books_subjects__, __films_genres__, __books_authors__, __books_aspects__ = __tonkens_from_documents_gensim__()
    preferences_IDs = []
    pref_entity = []
    movie_to_ignore = []
    negative_entity = []
    pref_aspects = []
    neg_aspects = []
    rec_list_size = 5
    #get_suggestion(preferences_IDs, pref_entity, movie_to_ignore, negative_entity, pref_aspects, neg_aspects, rec_list_size)
    # ADDESTRAMENTO D2V
    #pp_docs, IDs, titles, subjects, genres, authors = __tonkens_from_documents_gensim__()
    #__d2v__.load_model(pp_docs, "Models/Doc2Vec/doc2vec_model")

    # ADDESTRAMENTO FastText
    #pp_docs, IDs, titles, subjects, genres, authors =__tonkens_from_documents_gensim__()
    #__ft__.load_model(pp_docs, "Models/FastText/fasttext_model")

    # ADDESTRAMENTO TFIDF
    # pp_docs, IDs, titles, subjects, genres, authors =  __tonkens_from_documents_gensim__()
    # __tfidf__.load_model(pp_docs, "Models/TFIDF/tfidf_model",
    #                                                   "Models/TFIDF/matrix_tfidf",
    #                                                    "Models/TFIDF/dictionary_tfidf")

    # ADDESTRAMENTO W2V
    # pp_docs, IDs, titles, subjects, genres, authors =  __tonkens_from_documents_gensim__()
    # __w2v__.load_model(pp_docs, "Models\Word2Vec\word2vec_model",pretrained=False)



    #    preferences = list()
    # with open("""C:\\Users\petru\Documents\Tesi\createProfile\\the_godfather_test.csv""",  newline='', encoding="utf8") as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         preferences.append(row["ID"])
    # csvfile.close()
    # for i in range(1, 8):
   # select_model(7)
    # "Q172975", "Q26698156", "Q182254"
    # "Q40831", "Q157443", "Q193815", "Q1329
    #get_suggestion(preferences_IDs=[], pref_entity=[], negative_entity=[], movie_to_ignore=[])
#     # Q102244-Q102438 Harry Potter 1-2
#     # Q192724-Q163872 Iron Man-Cavalire Oscuro #TFIDF forse dovuta alla lunghezza della trama di Batman rispetto
#     # Q190525-Q220713 Memento-American Pie
#     # Q47075-Q36479 Scarface-Re Leone
#     # Q13099455-Q27894574-Q63985561-Q274167-Q219315 Maze Runner-Bohemian Rhapsody-Tenet-L'esorcista-Hangover
#     # Q155476-Q1392744-Q188652-Q1930376-Q483815 The Fast and the Furious-The Wolf of Wall Street-Rocky-Classic Albums: Nirvana – Nevermind-Shrek
