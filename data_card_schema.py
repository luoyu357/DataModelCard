from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable
from dataclasses import dataclass
import json
from json import JSONEncoder
#need to install jsonschema package
from jsonschema import validate
import os.path

SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'data_card_schema.json')

@dataclass
class Provenance:
    provenance_field: Optional[str] = 'Provenance Name'
    link_to_ids: Optional[List[str]] = field(default_factory=list)

@dataclass
class DataCardMetadata:
    name: Optional[str] = "Input Required (machine)"
    version: Optional[str] = "Input Required (user)"
    keyword: Optional[List[str]] = field(default_factory=list)
    description: Optional[str] = "Input Required (user)"
    publisher: Optional[str] = "Input Required (user)"
    location: Optional[str] = "Input Required (machine)"
    licence: Optional[str] = "Input Required (user)"
    persistent_identifier: Optional[str] = "Input Required (machine)"
    deposit_date: Optional[str] = "Input Required (machine)"




#this class is going to describe the metadata of data
@dataclass
class DataMetadata:
    name: Optional[str] = "Input Required (machine)"
    owner: Optional[str] = "Input Required (user)"
    location: Optional[str] = "Input Required (machine)"
    description: Optional[str] = "Input Required (user)",
    persistent_identifier: Optional[str] = "Input Required (machine)",
    dataset_characteristics: Optional[List[str]] = field(default_factory=list)
    subject_area: Optional[List[str]] = field(default_factory=list)
    associated_tasks: Optional[List[str]] = field(default_factory=list)
    attribute_type: Optional[List[str]] = field(default_factory=list)
    instances_number: Optional[int] = 0
    attributes_number: Optional[int] = 0




#list the content of an attribute
@dataclass
class AttributeExplainer:
    list_of_attribute: Optional[List[str]] = field(default_factory=list)
    additional_info: Optional[str] = "Input Required (user)"

#show the analyses about one feature
@dataclass
class Analyses:
    function_name: Optional[str] = "Input Required (machine)"
    result: Optional[str] = "Input Required (machine)"
    bias_explainer: Optional[List[str]] = field(default_factory=list)

#focus on one feature
@dataclass
class FeatureInfo:
    attribute_explainer: Optional[AttributeExplainer]
    name: Optional[str] = "Input Required (machine)"
    type: Optional[str] = "Input Required (machine)"
    demographic: Optional[str] = "Input Required (user/machine)"
    description: Optional[str] = "Input Required (user)"
    units: Optional[str] = "Input Required (user)"
    missing_value: Optional[bool] = field(default=False)
    analyses: Optional[List[Analyses]] = field(default_factory=list)


#this class is going to describe the data itself
@dataclass
class Data:
    data_metadata: DataMetadata
    features: List[FeatureInfo] = field(default_factory=list)
    provenance: Optional[List[Provenance]] = field(default_factory=list)


@dataclass
class TrainData:
    name: Optional[str] = "Input Required (machine)"
    location: Optional[str] = "Input Required (machine)"
    instances_number: Optional[int] = 0
    persistent_identifier: Optional[str] = "Input Required (machine)"
    features: Optional[List[FeatureInfo]] = field(default_factory=list)
    provenance: Optional[List[Provenance]] = field(default_factory=list)


@dataclass
class TestData:
    name: Optional[str] = "Input Required (machine)"
    location: Optional[str] = "Input Required (machine)"
    instances_number: Optional[int] = 0
    persistent_identifier: Optional[str] = "Input Required (machine)"
    features: Optional[List[FeatureInfo]] = field(default_factory=list)
    provenance: Optional[List[Provenance]] = field(default_factory=list)

#build the data card
@dataclass
class DataCard:
    data_card_metadata: DataCardMetadata
    data: Data
    train_data: TrainData
    test_data: TestData
    provenance: Optional[List[Provenance]] = field(default_factory=list)



def validate_save(data_card, save):
    dc = json.dumps(data_card, cls=DataCardJSONEncoder, indent=4)

    try:
        with open(SCHEMA_JSON, 'r') as schema_file:
            schema = json.load(schema_file)

        validate(json.loads(dc), schema)

        dc = json.loads(dc)
        with open(save, 'w') as json_file:
            json.dump(dc, json_file, indent=4)
        return True
    except Exception as e:
        print(e)
        return False


def makejson(data_card):
    dc = json.dumps(data_card, cls=DataCardJSONEncoder, indent=4)

    try:
        with open(SCHEMA_JSON, 'r') as schema_file:
            schema = json.load(schema_file)

        validate(json.loads(dc), schema)
        dc = json.loads(dc)
        return dc
    except Exception as e:
        print(e)




class DataCardJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (FeatureInfo, Data, DataCard, DataMetadata, DataCardMetadata, Analyses, AttributeExplainer, TestData, TrainData, Provenance)):
            return obj.__dict__
        return super().default(obj)
