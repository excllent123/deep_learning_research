

class Model_average():
    def __init__(self):
        self.model=[]
        self.proba=None

    def add_model(self, jsonPath, weightPath):
        # model have to be loaded weight
        json_file = open(jsonPath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # struc model
        model = model_from_json(loaded_model_json)
        # loading model weight
        model.load_weights(weightPath)
        # add model
        self.model.append(model)

    def predict_proba(self, iuPut):
        for model in self.model:
            predictProba = model.predict_proba(iuPut)[0]
            if self.proba is None:
                self.proba=predictProba
            else:
                self.proba+=predictProba
        result = self.proba/float(len(self.model))
        self.proba=None
        return [result]




