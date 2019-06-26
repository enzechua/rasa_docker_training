import shutil
import sys
import os
import requests

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer, Metadata, Interpreter
from rasa_nlu import config
import pprint
import spacy

print(spacy.load("en")("hello"))

def remove():
    shutil.rmtree("./models/nlu")

def train_nlu(data='./data/nlu_data.md', configs='nlu_config.yml'
              , model_dir='./models/test'):
    training_data = load_data(data)  # load NLU training sample
    trainer = Trainer(config.load(configs))  # train the pipeline first
    interpreter = trainer.train(training_data)  # train the model
    model_directory = trainer.persist(model_dir, project_name="nlu",
                                      fixed_model_name="rasa_nlu")  # store in directory


def run_nlu(text):
    interpreter = Interpreter.load('./models/nlu/rasa_nlu')
    pprint.pprint(interpreter.parse(text))


if __name__ == '__main__':
    if sys.argv[1] == "train":
        if os.path.exists("./models/nlu"):
            remove()
        train_nlu('./data/nlu.md', 'config.yml', './models/')

    elif sys.argv[1] == "run":
        full_sentence = ""
        for text in sys.argv[1:]:
            full_sentence += text + " "
        run_nlu(full_sentence)
