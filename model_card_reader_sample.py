from model_card_schema import *


class ModelCardReaderSample:

    def create_provenance(self):
        provenance = Provenance(
            provenance_field='provenance name',
            link_to_ids='anny id'
        )

        return provenance

    def create_model_card_metadata(self):
        model_card_metadata = ModelCardMetadata(
            name='test',
            version='v1',
            keyword=['test'],
            description='this is a test',
            publisher='Yu Luo',
            location='file system',
            licence='license',
            persistent_identifier='pid/1',
            deposit_date=datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        )

        return model_card_metadata

    def create_data_collection(self):
        data_collection = DataCollection(
            data_source_annotation='Data Card PID'
        )

        return data_collection

    def create_data_processing(self):
        data_processing = DataProcessing(
            data_clearn_algorithm= 'code',
            feature_engineering_algorithm= 'code',
            data_transformation_algorithm= 'code',
            data_augmentation_algorithm= 'code'
        )

        return data_processing

    def create_model_metadata(self):
        model_metadata = ModelMetadata(
            name='classification for adult data',
            version='v1',
            description='this is a test',
            owner='Yu Luo',
            location='url of document or other things',
            license='license'
        )

        return model_metadata


    def create_model_selection(self, model_metadata):
        model_selection = ModelSelection(
            model_metadata=model_metadata,
            algorithm_selection_algorithm='code',
            hyperparameter_tuning_algorithm='code'
        )

        return model_selection

    def create_performance_metrics(self):
        performance_metrics = PerformanceMetrics(
            metrics_name='Loss',
            related_feature='any feature',
            value='0'
        )

        return performance_metrics

    def create_model_output(self):
        model_output = ModelOutput(
            name='output model file',
            version='v1',
            description='this is a test',
            owner='Yu Luo',
            location='where we save the model file (pt or other files)'
        )

        return model_output

    def create_trainning_data_explainer(self):
        training_data_explainer = TrainingDataExplainer(
            key='target feature',
            value='feature'
        )

        return training_data_explainer

    def create_model_trainning(self, training_data_explainer, model_selection, model_output):
        model_training = ModelTraining(
            training_data='Training Data PID',
            training_data_explainer=training_data_explainer,
            validate_data_algorithm='code',
            mode_selection=model_selection,
            training_algorithm='code',
            model_output=model_output
        )

        return model_training

    def create_adversarial_testing(self, performance_metrics):
        adversarial_testing = AdversarialTesting(
            independent_of_attributes=['none'],
            feasibility=True,
            performance_metrics=performance_metrics
        )

        return adversarial_testing

    def create_bias_and_fairness_evaluation(self, fairness_metrics, adversarial_testing,
                                            interpretability_tools, feedback_mechanisms):
        bias_and_fairness_evaluation = BiasAndFairnessEvaluation(
            fairness_metrics=fairness_metrics,
            adversarial_testing=adversarial_testing,
            interpretability_tools=interpretability_tools,
            feedback_mechanisms=feedback_mechanisms
        )

        return bias_and_fairness_evaluation

    def create_model_evaluation(self, performance_metrics, bias_and_fairness_evaluation):
        model_evaluation = ModelEvaluation(
            testing_data='Testing Data PID',
            evaluate_data_algorithm='code',
            performance_metrics=performance_metrics,
            bias_and_fairness_evaluation=[bias_and_fairness_evaluation]
        )

        return model_evaluation


    def create_deployment(self):
        deployment = Deployment(
            key='system',
            value='Ubuntu'
        )

        return deployment

    def create_model_deployment(self, deployment):
        model_deployment = ModelDeployment(
            deployment_platform=deployment,
            APIs='https'
        )

        return model_deployment

    def create_model_card(self, model_card_metadata, data_collection, data_processing,
                          model_selection, model_training, model_evaluation,
                          model_deployment, provenance):
        model_card = ModelCard(
            model_card_metadata=model_card_metadata,
            data_collection=data_collection,
            data_processing=data_processing,
            model_selection=model_selection,
            model_training=model_training,
            model_evaluation=model_evaluation,
            model_deployment=model_deployment,
            provenance=provenance
        )

        return model_card

if __name__ == '__main__':
    card = ModelCardReaderSample()
    provenance = [card.create_provenance()]
    model_card_metadata = card.create_model_card_metadata()
    data_collection = card.create_data_collection()
    data_processing = card.create_data_processing()
    model_metadata = card.create_model_metadata()
    model_selection = card.create_model_selection(model_metadata)
    performance_metrics = [card.create_performance_metrics()]
    model_output = card.create_model_output()
    training_data_explainer = [card.create_trainning_data_explainer()]
    model_training = card.create_model_trainning(training_data_explainer, model_selection, model_output)
    adversarial_testing = [card.create_adversarial_testing(performance_metrics)]
    bias_and_fairness_evaluation = card.create_bias_and_fairness_evaluation(performance_metrics, adversarial_testing,
                                                                            performance_metrics, performance_metrics)
    model_evaluation = card.create_model_evaluation(performance_metrics, bias_and_fairness_evaluation)
    deployment = [card.create_deployment()]
    model_deployment = card.create_model_deployment(deployment)
    model_card = card.create_model_card(model_card_metadata, data_collection, data_processing,
                                        model_selection, model_training,
                                        model_evaluation, model_deployment, provenance)
    print(validate_save(model_card, './card/model/sample_card.json'))