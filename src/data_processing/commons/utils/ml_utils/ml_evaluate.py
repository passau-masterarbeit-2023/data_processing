import json
import logging
from typing import Any
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

from commons.results.base_result_writer import BaseResultWriter
from ..utils import time_measure_result

    
def evaluate(
        clf: Any, 
        test_samples: list[list[int]], 
        test_labels: list[int],
        logger : logging.Logger,
        result_saver: BaseResultWriter,
):
    """
    Evaluate the model.
    """
    # evaluate the model
    with time_measure_result('evaluate_model_score', logger):
        
        # make predictions on the test data
        y_pred = clf.predict(test_samples)

        # print information about the labels (test and predicted labels)
        logger.info(
            "Sample of predicted labels: %s \n versus actual labels: %s", y_pred[:20], test_labels[:20]
        )
        logger.info(
            "Number of predicted 1 labels: {} \n versus number of predicted 0 labels: {}".format(
                sum(y_pred), len(y_pred) - sum(y_pred)
            )
        )

        # calculate the accuracy of the model
        accuracy = accuracy_score(test_labels, y_pred)
        logger.info("Accuracy: {:.2f}%".format(accuracy * 100))
        result_saver.set_result("accuracy", str(accuracy))

        # Get the classification report in a dictionary format
        clf_report = classification_report(test_labels, y_pred, output_dict=True)

        # print the classification report as a dict, with 4 spaces as indentation
        logger.info(json.dumps(clf_report, indent=4))

        # Save classification report metrics
        # TODO: not sure about the 1 here, need to check
        result_saver.set_result("precision", str(clf_report['1']['precision']))
        result_saver.set_result("recall", str(clf_report['1']['recall']))
        result_saver.set_result("f1_score", str(clf_report['1']['f1-score']))
        result_saver.set_result("support", str(clf_report['1']['support']))

        # calculate the confusion matrix
        cm = confusion_matrix(test_labels, y_pred)
        logger.info("Confusion Matrix: ")
        logger.info("True Positives: %d", cm[1, 1])
        logger.info("True Negatives: %d", cm[0, 0])
        logger.info("False Positives: %d", cm[0, 1])
        logger.info("False Negatives: %d", cm[1, 0])

        # save to results
        result_saver.set_result("true_positives", str(cm[1, 1]))
        result_saver.set_result("true_negatives", str(cm[0, 0]))
        result_saver.set_result("false_positives", str(cm[0, 1]))
        result_saver.set_result("false_negatives", str(cm[1, 0]))

        # calculate the false positive rate and true positive rate
        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)

        # calculate the area under the ROC curve
        roc_auc = auc(fpr, tpr)
        logger.info("AUC: {:.2f}".format(roc_auc))
        result_saver.set_result("auc", str(roc_auc))