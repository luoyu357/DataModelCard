from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable
from dataclasses import dataclass
import json
from json import JSONEncoder
#need to install jsonschema package
from jsonschema import validate
import os


SCHEMA_JSON = os.path.join(os.path.dirname(__file__), 'schema', 'model_card_schema.json')


@dataclass
class Provenance:
    provenance_field: Optional[str] = 'Provenance Name'
    link_to_ids: Optional[List[str]] = field(default_factory=list)


@dataclass
class ModelCardMetadata:
    name: Optional[str] = "Input Required (machine)"
    version: Optional[str] = "Input Required (user)"
    keyword: Optional[List[str]] = field(default_factory=list)
    description: Optional[str] = "Input Required (user)"
    publisher: Optional[str] = "Input Required (user)"
    location: Optional[str] = "Input Required (machine)"
    licence: Optional[str] = "Input Required (user)"
    persistent_identifier: Optional[str] = "Input Required (machine)"
    deposit_date: Optional[str] = "Input Required (machine)"


@dataclass
class DataCollection:
    data_source_annotation: Optional[str] = "Data Card PID (machine)"

@dataclass
class DataProcessing:
    data_clearn_algorithm: Optional[str] = "Input Required (machine)"
    feature_engineering_algorithm: Optional[str] = "Input Required (machine)"
    data_transformation_algorithm: Optional[str] = "Input Required (machine)"
    data_augmentation_algorithm: Optional[str] = "Input Required (machine)"


@dataclass
class ModelMetadata:
    name: Optional[str] = "Input Required (user)"
    version: Optional[str] = "Input Required (user)"
    description: Optional[str] = "Input Required (user)"
    owner: Optional[str] = "Input Required (user)"
    location: Optional[str] = "Input Required (machine)"
    license: Optional[str] = "Input Required (user)"



@dataclass
class ModelSelection:
    model_metadata: Optional[ModelMetadata] = field(default_factory=list)
    algorithm_selection_algorithm: Optional[str] = "Input Required (machine)"
    hyperparameter_tuning_algorithm: Optional[str] = "Input Required (machine)"

@dataclass
class PerformanceMetrics:
    metrics_name: Optional[str] = "Input Required (machine)"
    related_feature: Optional[str] = "Input Required/Feature ID (user/machine)"
    value: Optional[str] = "Input Required (machine)"

@dataclass
class ModelOutput:
    name: Optional[str] = "Input Required (user)"
    version: Optional[str] = "Input Required (user)"
    description: Optional[str] = "Input Required (user)"
    owner: Optional[str] = "Input Required (user)"
    location: Optional[str] = "Input Required (machine)"

@dataclass
class TrainingDataExplainer:
    key: Optional[str] = "Input Required (user)"
    value: Optional[str] = "Input Required (user)"

@dataclass
class ModelTraining:
    training_data: Optional[str] = "Training Data PID (machine)"
    training_data_explainer: Optional[List[TrainingDataExplainer]] = field(default_factory=list)
    validate_data_algorithm: Optional[str] = "Input Required (machine)"
    mode_selection: Optional[ModelSelection] = field(default_factory=list)
    training_algorithm: Optional[str] = "Input Required (machine)"
    model_output: Optional[ModelOutput] = "Input Required (machine)"

@dataclass
class AdversarialTesting:
    independent_of_attributes: Optional[List[str]] = field(default_factory=list)
    feasibility: Optional[bool] = field(default=False)
    performance_metrics: Optional[List[PerformanceMetrics]] = field(default_factory=list)


@dataclass
class BiasAndFairnessEvaluation:
    fairness_metrics: Optional[List[PerformanceMetrics]] = field(default_factory=list)
    adversarial_testing: Optional[List[AdversarialTesting]] = field(default_factory=list)
    interpretability_tools: Optional[List[PerformanceMetrics]] = field(default_factory=list)
    feedback_mechanisms: Optional[List[PerformanceMetrics]] = field(default_factory=list)


@dataclass
class ModelEvaluation:
    testing_data: Optional[str] = "Testing Data PID (machine)"
    evaluate_data_algorithm: Optional[str] = "Input Required (machine)"
    performance_metrics: Optional[List[PerformanceMetrics]] = field(default_factory=list)
    bias_and_fairness_evaluation: Optional[BiasAndFairnessEvaluation] = field(default_factory=list)



@dataclass
class Deployment:
    key: Optional[str] = "Input Required (machine/user)"
    value: Optional[str] = "Input Required (machine/user)"


@dataclass
class ModelDeployment:
    deployment_platform: Optional[List[Deployment]] = field(default_factory=list)
    APIs: Optional[str] = "Input Required (machine/user)"


@dataclass
class ModelCard:
    model_card_metadata: ModelCardMetadata
    data_collection: DataCollection
    data_processing: DataProcessing
    model_selection: ModelSelection
    model_training: ModelTraining
    model_evaluation: ModelEvaluation
    model_deployment: ModelDeployment
    provenance: Optional[List[Provenance]] = field(default_factory=list)


def validate_save(model_card, save):
    dc = json.dumps(model_card, cls=ModelCardJSONEncoder, indent=4)

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

class ModelCardJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Provenance, ModelCard, ModelMetadata, DataCollection, DataProcessing, ModelSelection, ModelOutput,
                            ModelTraining, ModelEvaluation, ModelDeployment,
                            PerformanceMetrics, TrainingDataExplainer, BiasAndFairnessEvaluation,
                            Deployment, ModelCardMetadata, AdversarialTesting)):
            return obj.__dict__
        return super().default(obj)
