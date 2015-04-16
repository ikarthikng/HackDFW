__author__ = 'Karthik'

import nltk

def read_train_file():
    _train_file = {}

    f = open("E:\Masters\HackDFW\Hackaton-Users_Train.tsv", "r")

    for train in f:
        userID, label = train.split()
        _train_file[userID] = label

    return _train_file

def read_screen_user_file():
    _user_set = set()

    f = open("E:\Masters\HackDFW\Hackaton-Users_Screens.tsv", "r")
    i = 0
    for screen_user in f:
        screen, user = screen_user.split()
        _user_set.add(user)

    return _user_set

def features(number):
    return {'user_id':number[:-3]}

if __name__ == "__main__":
    _output_userIds = set()
    train_file = read_train_file()
    featuresets = [(features(n), c) for (n,c) in train_file.items()]
    #print train_file
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    #print nltk.classify.accuracy(classifier, featuresets[:1000])
    #print classifier.classify(features('3750857'))
    #print classifier.show_most_informative_features(5)
    _user_set = read_screen_user_file()
    _output_userIds = _user_set - set(train_file.keys())
    f = open('problem.txt', 'wb')
    for line in _output_userIds:
        f.write(str(line)+"\t"+str(classifier.classify(features(line)))+"\n")
        #print line,classifier.classify(features(line))
    f.close()







