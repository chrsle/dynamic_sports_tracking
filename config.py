#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import configparser

class Config:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    @property
    def database_config(self):
        return self.config['DATABASE']

    @property
    def model_config(self):
        return self.config['MODEL']

    @property
    def training_config(self):
        return self.config['TRAINING']

    @property
    def evaluation_config(self):
        return self.config['EVALUATION']


if __name__ == "__main__":
    config = Config('dynamic_sports_tracking/config.ini')

    print("Database config: ", config.database_config)
    print("Model config: ", config.model_config)
    print("Training config: ", config.training_config)
    print("Evaluation config: ", config.evaluation_config)

