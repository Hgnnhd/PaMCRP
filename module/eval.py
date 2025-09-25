from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from config import *
from lifelines.utils.btree import _BTree
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc
from lifelines import KaplanMeierFitter
import numpy as np

def parallel_bootstrap(func, data, n_iterations=1000, n_jobs=-1):
    with Pool(n_jobs) as pool:
        results = pool.map(func, [data] * n_iterations)
    return np.mean(results), np.percentile(results, [2.5, 97.5])

def bootstrap_c_index_parallel(data):
    censor_times, predictions, true_labels, censor_distribution = data
    indices = np.random.choice(len(censor_times), len(censor_times), replace=True)
    return concordance_index(censor_times[indices], predictions[indices], true_labels[indices], censor_distribution)
def bootstrap_auc_parallel(data):
    y_true, y_scores = data
    indices = np.random.choice(len(y_true), len(y_true), replace=True)
    return roc_auc_score(y_true[indices], y_scores[indices])
def get_censoring_dist(time_at_event, golds):

    times, event_observed = time_at_event, golds
    all_observed_times = set(times)
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    return censoring_dist
def concordance_index(event_times, predicted_scores, event_observed=None, censoring_dist=None):
    """
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The c-index is the average of how often a model says X is greater than Y when, in the observed
    data, X is indeed greater than Y. The c-index also handles how to handle censored values
    (obviously, if Y is censored, it's hard to know if X is truly greater than Y).


    The concordance index is a value between 0 and 1 where:

    - 0.5 is the expected result from random predictions,
    - 1.0 is perfect concordance and,
    - 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Parameters
    ----------
    event_times: iterable
         a length-n iterable of observed survival times.
    predicted_scores: iterable
        a length-n iterable of predicted scores - these could be survival times, or hazards, etc. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
    event_observed: iterable, optional
        a length-n iterable censorship flags, 1 if observed, 0 if not. Default None assumes all observed.

    Returns
    -------
    c-index: float
      a value between 0 and 1.

    References
    -----------
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.

    """
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = 1 - np.asarray(predicted_scores, dtype=float)


    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")

    num_correct, num_tied, num_pairs = _concordance_summary_statistics(event_times, predicted_scores, event_observed, censoring_dist)

    return _concordance_ratio(num_correct, num_tied, num_pairs)
def _concordance_ratio(num_correct, num_tied, num_pairs):
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs
def _concordance_summary_statistics(
    event_times, predicted_event_times, event_observed, censoring_dist
):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.module.concordance_index first.
    """
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    observed_times = set(event_times)


    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]

    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = {}
    for time in observed_times:
        times_to_compare[time] = _BTree(np.unique(died_pred[:, int(time)]))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare, censoring_dist)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix, weight = _handle_pairs(died_truth, died_pred, died_ix, times_to_compare, censoring_dist)
            for pred in died_pred[died_ix:next_ix]:
                for time in observed_times:
                    times_to_compare[time].insert(pred[int(time)])
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs * weight
        num_correct += correct * weight
        num_tied += tied * weight

    return (num_correct, num_tied, num_pairs)
def _handle_pairs(truth, pred, first_ix, times_to_compare, censoring_dist):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    truth_time = truth[first_ix]
    weight = 1./(censoring_dist[truth_time]**2)
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare[truth_time]) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare[truth_time].rank(pred[i][int(truth_time)])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix, weight)


def evaluate_top5(model, test_loader, test=False, threshold=0.5):
    model.eval()
    roc_data = {}
    all_predictions = []
    all_labels = []
    all_predictions_prob = []
    all_masks = []
    all_golds = []
    all_genders = []
    all_top5_golds = []

    all_predictions_2 = []
    all_predictions_5 = []

    all_predictions_2_prob = []

    with torch.no_grad():
        for inputs, aux, labels, masks, time, gold, top5_gold in (test_loader):
            inputs = inputs.long()
            outputs, output_cls, output_top5, *extra = model(inputs, time, aux)

            predictions_prob = torch.stack([(output) for output in outputs],dim=-1).cpu().numpy()
            predictions = (predictions_prob >= threshold).astype(int)
            output_cls_prob = torch.softmax(output_cls, dim=1)
            all_predictions_2_prob.extend(output_cls_prob[:,1].cpu().numpy())

            prodiction_2 = torch.argmax(output_cls, dim=1)
            prodiction_5 = torch.argmax(output_top5, dim=1)

            all_predictions_2.extend(prodiction_2.cpu().numpy())
            all_predictions_5.extend(prodiction_5.cpu().numpy())

            all_predictions.extend(predictions)
            all_predictions_prob.extend(predictions_prob)
            all_labels.extend(labels.long().cpu().numpy())
            all_masks.extend(masks.cpu().numpy())
            all_golds.extend(gold.cpu().numpy())
            all_genders.extend(aux[:, 0].cpu().numpy())
            all_top5_golds.extend(top5_gold.cpu().numpy())


    all_masks = np.array(all_masks)
    all_genders = np.array(all_genders)
    all_top5_golds = np.array(all_top5_golds)
    all_golds = np.array(all_golds)
    all_predictions_prob = np.array(all_predictions_prob)
    all_predictions_2_prob = np.array(all_predictions_2_prob)

    all_predictions = np.array(all_predictions)
    all_predictions_2 = np.array(all_predictions_2)
    all_predictions_5 = np.array(all_predictions_5)

    accuracy = accuracy_score(all_golds, all_predictions_2)

    accuracy_5 = accuracy_score(all_top5_golds, all_predictions_5)

    cm = confusion_matrix(all_golds, all_predictions_2)
    cm_5 = confusion_matrix(all_top5_golds, all_predictions_5)

    censor_times = np.argmax(all_masks[:, ::-1] == 1, axis=1)
    censor_times = all_masks.shape[1] - 1 - censor_times

    results = {}
    c_indices = {}
    c_index_cis = {}
    censor_distribution = get_censoring_dist(censor_times, all_golds)

    for category in range(1, 6):
        #print(f"Calculating metrics for category {category}")

        category_mask = all_top5_golds == category
        class_0_mask = all_top5_golds == 0


        if category == 2:
            gender_mask = all_genders == 1
        elif category == 4:
            gender_mask = all_genders == 0
        else:
            gender_mask = np.ones_like(all_genders, dtype=bool)

        combined_mask = np.logical_or(category_mask, class_0_mask) & gender_mask

        category_censor_times = censor_times[combined_mask]
        category_predictions = all_predictions_prob[combined_mask, :, category - 1]
        category_true_labels = all_golds[combined_mask]

        if test:
            data = (category_censor_times, category_predictions, category_true_labels, censor_distribution)
            mean_c_index, (ci_lower, ci_upper) = parallel_bootstrap(bootstrap_c_index_parallel, data, n_iterations=1000, n_jobs=64)
            c_indices[category] = mean_c_index
            c_index_cis[category] = (ci_lower, ci_upper)
        else:
            c_index = concordance_index(category_censor_times, category_predictions, category_true_labels,
                                        censor_distribution)
            c_indices[category] = c_index
            ci_lower, ci_upper = 0.0, 0.0
            c_index_cis[category] = (ci_lower, ci_upper)
        results[f"C_index_{category}"] = c_indices[category]
        results[f"C_index_CI_{category}"] = c_index_cis[category]

    aucs = {}
    auc_cis = {}
    for followup, eval_time in enumerate(evaluation_times):
        censor_times = np.argmax(all_masks[:, ::-1] == 1, axis=1)
        censor_times = all_masks.shape[1] - 1 - censor_times

        golds_for_eval = []
        probs_for_eval = []
        genders_for_eval = []
        top5_golds_for_eval = []

        def include_exam_and_determine_label(followup, censor_time, gold):
            valid_pos = gold and censor_time <= followup
            valid_neg = censor_time >= followup
            included, lab = (valid_pos or valid_neg), int(valid_pos)
            return included, lab

        for prob_arr, censor_time, gold, gender, top5_gold in zip(all_predictions_prob, censor_times, all_golds,
                                                                  all_genders, all_top5_golds):

            include, label = include_exam_and_determine_label(followup, censor_time, gold)

            if include:
                probs_for_eval.append(prob_arr[followup])
                golds_for_eval.append(label)
                genders_for_eval.append(gender)
                top5_golds_for_eval.append(top5_gold)

        golds_for_eval = np.array(golds_for_eval)
        probs_for_eval = np.array(probs_for_eval)
        genders_for_eval = np.array(genders_for_eval)
        top5_golds_for_eval = np.array(top5_golds_for_eval)

        # Calculate AUC for each top5_gold category
        for category in range(1, 6):
            category_mask = top5_golds_for_eval == category
            class_0_mask = top5_golds_for_eval == 0


            if category == 2:
                gender_mask = genders_for_eval == 1
            elif category == 4:
                gender_mask = genders_for_eval == 0
            else:
                gender_mask = np.ones_like(genders_for_eval, dtype=bool)

            combined_mask = np.logical_or(category_mask, class_0_mask) & gender_mask
            combined_golds = golds_for_eval[combined_mask]
            combined_probs = probs_for_eval[combined_mask, category - 1]

            if len(np.unique(combined_golds)) > 1:  # Ensure we have both positive and negative samples

                fpr, tpr, _ = roc_curve(combined_golds, combined_probs)
                roc_auc = auc(fpr, tpr)

                if category not in roc_data:
                    roc_data[category] = {}
                roc_data[category][eval_time] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

                if test:
                    data = (combined_golds, combined_probs)
                    mean_auc, (ci_lower, ci_upper) = parallel_bootstrap(bootstrap_auc_parallel, data, n_iterations=1000,
                                                                        n_jobs=64)
                    aucs[category] = mean_auc
                else:
                    auc_roc = roc_auc_score(combined_golds, combined_probs)
                    aucs[category] = auc_roc
                    ci_lower, ci_upper = 0.0, 0.0

                results[f"Time_{eval_time}_{category}_AUC"] = float(aucs[category])
                results[f"Time_{eval_time}_{category}_AUC_CI"] = (float(ci_lower), float(ci_upper))

    return results,roc_data

