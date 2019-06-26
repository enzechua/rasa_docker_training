from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import rasa_core
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.utils import EndpointConfig
from rasa_core.run import serve_application
from rasa_core import config
import sys
import os
import shutil


logger = logging.getLogger(__name__)

def remove():
    shutil.rmtree("./models/rasa_core")


def train_core(domain_file='domain.yml',
                   model_path='./models/rasa_core',
                   training_data_file='./data/stories.md'):
    agent = Agent(domain_file, policies=[MemoizationPolicy(), KerasPolicy(max_history=3, epochs=200, batch_size=50)])
    data = agent.load_data(training_data_file)

    agent.train(data)

    agent.persist(model_path)
    return agent


def run_core(serve_forever=True):
    interpreter = RasaNLUInterpreter('./models/rasa_core')
    action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
    agent = Agent.load('./models/rasa_core', interpreter=interpreter, action_endpoint=action_endpoint)
    rasa_core.run.serve_application(agent, channel='cmdline')

    return agent


if __name__ == '__main__':
    if sys.argv[1] == "train":
        if os.path.exists("./models/core"):
            remove()
        train_core()

    elif sys.argv[1] == "run":
        full_sentence = ""
        for text in sys.argv[1:]:
            full_sentence += text + " "
        run_core(full_sentence)