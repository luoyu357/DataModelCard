import csv
import pandas as pd
from datetime import datetime
from data_card_schema import *
import os


class ReaderDifferentHelper:

    def __init__(self, file_path, column_names, row_data):
        self.filename = os.path.basename(file_path)
        self.data = pd.DataFrame(row_data, columns=column_names)
        self.statistical_data = self.data.describe()
        self.types = self.data.dtypes


    def get_name(self):
        return self.filename

    def get_head(self):
        return list(self.data.columns)

    def get_column(self, name_of_column):
        return self.data[name_of_column]

    def get_instance_number(self):
        return int(self.data.size)

    def get_attribute_number(self):
        return int(self.data.columns.size)

    #contains count, max, min, mean, std, 25, 50, 75 ...
    def get_statistical_info(self, name_of_column):
        if name_of_column not in self.statistical_data.keys():
            return None
        return self.statistical_data[name_of_column]

    def get_types(self):
        return [str(i) for i in self.types.unique()]

    def get_type_column(self, name_of_column):
        return str(self.types[name_of_column])

    def get_date(self):
        return datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    def check_missing(self, column_data):
        return bool(column_data.isnull().sum() != 0)

    def percentage_of_missing(self, column_data):
        return column_data.isnull().sum() / column_data.size

    def percentage_of_zero(self, column_data):
        return (column_data == 0).sum() / column_data.size

    def get_list_of_attributes(self, name_of_column):
        list_of_attributes = self.data[name_of_column].unique()

        if len(list_of_attributes) >= 10:
            return ['continuous']
        else:
            return [str(i) for i in list_of_attributes]

class DataCardReader:

    def __init__(self, file_path, row_data, columns_name, train_path, row_data_train, columns_name_train, test_path,
                 row_data_test, columns_name_test, save):

        self.readerhelp_data = ReaderDifferentHelper(file_path, row_data, columns_name)
        data_card_metadata = self.create_data_card_metadata(name=self.readerhelp_data.get_name(),
                                                            location=file_path,
                                                            deposite_date=self.readerhelp_data.get_date())
        data_metadata = self.create_data_metadata(name=self.readerhelp_data.get_name(),
                                                  location=file_path,
                                                  attribute_type=self.readerhelp_data.get_types(),
                                                  instances_number=self.readerhelp_data.get_instance_number(),
                                                  attributes_number=self.readerhelp_data.get_attribute_number())

        feature_info =self.create_all_feature_info(self.readerhelp_data)

        provenance = []
        temp_provenance = self.create_provenance()
        provenance.append(temp_provenance)

        data = self.create_data(data_metadata, feature_info, provenance)

        self.readerhelp_train = ReaderDifferentHelper(train_path, row_data_train, columns_name_train)
        train_feature_info = self.create_all_feature_info(self.readerhelp_train)
        train_data = self.create_train_data(train_feature_info,
                                            provenance,
                                            name=self.readerhelp_train.get_name(),
                                            location=train_path,
                                            instance_number=self.readerhelp_train.get_instance_number())

        self.readerhelp_test = ReaderDifferentHelper(test_path, row_data_test, columns_name_test)
        test_feature_info = self.create_all_feature_info(self.readerhelp_test)
        test_data = self.create_test_data(test_feature_info,
                                          provenance,
                                          name=self.readerhelp_test.get_name(),
                                          location=test_path,
                                          instance_number=self.readerhelp_test.get_instance_number())

        data_card = self.create_data_card(data_card_metadata, data, train_data, test_data, provenance)

        print(validate_save(data_card, save))


    def create_all_feature_info(self, data):
        header = data.get_head()

        feature_info = []
        for column_name in header:
            attribute_explainer = self.create_attribute_explainer(
                list_of_attribute=data.get_list_of_attributes(column_name))
            analyses = []
            statistical_info = data.get_statistical_info(column_name)
            if statistical_info is not None:
                for function_name in statistical_info.keys():
                    temp_analyses = self.create_analyses(function_name=function_name,
                                                         result=str(statistical_info.get(function_name)))
                    analyses.append(temp_analyses)
            # additial function
            percentage_of_zero = data.percentage_of_zero(data.get_column(column_name))
            analyses.append(self.create_analyses(function_name='Percentage of Zero',
                                                 result=str(percentage_of_zero)))
            percentage_of_missing = data.percentage_of_missing(
                data.get_column(column_name))
            analyses.append(self.create_analyses(function_name='Percentage of Missing',
                                                 result=str(percentage_of_missing)))

            feature_info.append(self.create_feature_info(attribute_explainer,
                                                         analyses,
                                                         name=column_name,
                                                         type_info=data.get_type_column(column_name),
                                                         missing_value=data.check_missing(
                                                             data.get_column(column_name))
                                                         ))
        return feature_info

    def create_data_card_metadata(self, name='Input Required', version='Input Required',
                                  keyword=['Input Required'], description='Input Required', publisher='Input Required',
                                  location='Input Required', persistent_identifier='Input Required',
                                  deposite_date=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")):
        data_card_metadata = DataCardMetadata(
            name=name,
            version=version,
            keyword=keyword,
            description=description,
            publisher=publisher,
            location=location,
            persistent_identifier=persistent_identifier,
            deposit_date= deposite_date
        )
        return data_card_metadata

    def create_data_metadata(self, name= 'Input Required', owner='Input Required', location='Input Required',
                             description='Input Required', persistent_identifier='Input Required',
                             dataset_characteristics=['Input Required'],
                             subject_area=['Input Required'], associated_tasks=['Input Required'],
                             attribute_type=['Input Required'], instances_number=0, attributes_number=0):
        data_metadata = DataMetadata(
            name=name,
            owner=owner,
            location=location,
            description=description,
            persistent_identifier=persistent_identifier,
            dataset_characteristics=dataset_characteristics,
            subject_area=subject_area,
            associated_tasks=associated_tasks,
            attribute_type=attribute_type,
            instances_number=instances_number,
            attributes_number=attributes_number
        )

        return data_metadata

    def create_attribute_explainer(self, list_of_attribute=['Input Required'],
                                   additional_info='Input Required'):
        attribute_explainer = AttributeExplainer(
            list_of_attribute=list_of_attribute,
            additional_info=additional_info

        )

        return attribute_explainer

    def create_analyses(self, function_name='Input Required',
                        result='Input Required',
                        bias_explainer='Input Required'):
        analyses = Analyses(
            function_name=function_name,
            result=result,
            bias_explainer=bias_explainer
        )
        return analyses

    def create_feature_info(self, attribute_explainer, analyses,
                            name='Input Required',
                            type_info='Input Required',
                            demographic='Input Required',
                            description='Input Required',
                            units='Input Required',
                            missing_value=False
                            ):
        feature_info = FeatureInfo(
            name=name,
            type=type_info,
            demographic=demographic,
            description=description,
            units=units,
            missing_value=missing_value,
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

    def create_train_data(self, feature_info, provenance,
                          name='Input Required',
                          location='Input Required',
                          persistent_identifier='Input Required',
                          instance_number=0):
        train_data = TrainData(
            name=name,
            location=location,
            instances_number=instance_number,
            persistent_identifier=persistent_identifier,
            features=feature_info,
            provenance=provenance
        )

        return train_data

    def create_test_data(self, feature_info, provenance,
                          name='Input Required',
                          location='Input Required',
                          persistent_identifier = 'Input Required',
                          instance_number=0):
        test_data = TestData(
            name=name,
            location=location,
            instances_number=instance_number,
            persistent_identifier=persistent_identifier,
            features=feature_info,
            provenance=provenance
        )

        return test_data

    def create_provenance(self, provenance_field='Input Required',
                          link_to_ids=["Input Required"]):
        provenance = Provenance(
            provenance_field=provenance_field,
            link_to_ids=link_to_ids
        )
        return provenance

if __name__ == '__main__':

    file_path = 'data/data/adult/adult.data'
    data = pd.read_csv(file_path, sep=', ', header=None)
    row_data = []
    for index, row in data.iterrows():
        temp = []
        for item in row:
            if isinstance(item, float):
                temp.append(float(item))
            elif isinstance(item, int):
                temp.append(int(item))
            else:
                if item in  " !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~":
                    temp.append(float('nan'))
                else:
                    temp.append(item)
        row_data.append(temp)

    name = pd.read_csv('data/data/adult/old.adult.names', sep=': ', header=None)
    column_names = list(name.iloc[:,0])


    train_path = 'data/data/adult/adult.data'

    test_path = 'data/data/adult/adult.test'

    data = pd.read_csv(test_path, sep=', ', header=None)
    row_data_test = []
    for index, row in data.iterrows():
        temp = []
        for item in row:
            if isinstance(item, int):
                temp.append(int(item))
            elif isinstance(item, float):
                temp.append(float(item))
            else:
                if item in  " !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~":
                    temp.append(float('nan'))
                else:
                    temp.append(item)
        row_data_test.append(temp)

    save = './card/data/adult.json'

    card = DataCardReader(file_path, column_names, row_data, train_path, column_names, row_data, test_path, column_names, row_data_test, save)



    '''
    file_path = './sample/adult/adult.data'
    row_data = []
    column_names = []
    train_path = './sample/adult/adult.data'
    row_data_train = []
    column_names_train = []
    test_path = './sample/adult/adult.test'
    row_data_test = []
    column_names_test = []
    card = DataCardReader(file_path, train_path, test_path)
    '''




