import csv
from datetime import datetime

from data_card_schema import *
from jsonschema import validate

class DataCardReaderSample:

    def __init__(self):
        self.header = None

    def create_provenance(self):
        provenance = Provenance(
            provenance_field='test',
            link_to_ids=['test']
        )
        return provenance

    def create_data_card_metadata(self):
        data_card_metadata = DataCardMetadata(
            name='test',
            version='v1.0.0',
            keyword=['test','draft'],
            description='this is a test data card',
            publisher='Yu Luo',
            location='dummy location',
            persistent_identifier='pid/1',
            deposit_date= datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        )
        return data_card_metadata

    def create_data_metadata(self):
        data_metadata = DataMetadata(
            name='Heart Disease',
            owner='kaggle',
            location='url',
            description='this is a test dataset',
            persistent_identifier='pid/1',
            dataset_characteristics=['Multivariate'],
            subject_area=['health'],
            associated_tasks=['classfication', 'regression'],
            attribute_type=['categorical', 'integer'],
            instances_number=200,
            attributes_number=10
        )

        return data_metadata

    def create_attribute_explainer(self):
        attribute_explainer = AttributeExplainer(
            list_of_attribute=['continuous'],
            additional_info='this is a test'

        )

        return attribute_explainer

    def create_analyses(self):
        analyses = Analyses(
            function_name='mean',
            result='123',
            bias_explainer='good result'
        )
        return analyses

    def create_feature_info(self, attribute_explainer, analyses):
        feature_info = FeatureInfo(
            name='age',
            type='Integer',
            demographic='Age',
            description='N/A',
            units='N/A',
            missing_value=False,
            attribute_explainer=attribute_explainer,
            analyses=analyses
        )

        return feature_info

    def create_data(self, data_metadata, feature_info, provenance):
        data = Data(
            data_metadata=data_metadata,
            features=feature_info,
            provenance=provenance
        )

        return data

    def create_data_card(self, card_metadata, data, train_data, test_data, provenance):
        data_card = DataCard(
            data_card_metadata=card_metadata,
            data=data,
            train_data=train_data,
            test_data=test_data,
            provenance=provenance
        )

        return data_card

    def create_train_data(self, feature_info, provenance):
        train_data = TrainData(
            name='train data',
            location='file location',
            instances_number=40,
            persistent_identifier='1',
            features=feature_info,
            provenance=provenance
        )

        return train_data

    def create_test_data(self, feature_info, provenance):
        test_data = TestData(
            name='test data',
            location='file loation',
            instances_number=30,
            persistent_identifier='1',
            features=feature_info,
            provenance=provenance
        )

        return test_data

if __name__ == '__main__':

    filename = 'data/data/heart/heart.csv'
    card = DataCardReaderSample()
    provenance = [card.create_provenance()]
    data_card_metadata = card.create_data_card_metadata()
    data_metadata = card.create_data_metadata()
    attribute_explainer = card.create_attribute_explainer()
    analyses = card.create_analyses()
    feature_info = card.create_feature_info(attribute_explainer, [analyses])
    data = card.create_data(data_metadata, [feature_info], provenance)
    train_data = card.create_train_data([feature_info], provenance)
    test_data = card.create_test_data([feature_info], provenance)
    data_card = card.create_data_card(data_card_metadata, data, train_data, test_data, provenance)
    print(validate_save(data_card, './card/data/sample_card.json'))



