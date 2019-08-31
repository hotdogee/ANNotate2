class ColoredLoggingTensorHook(tf.train.LoggingTensorHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.

    The tensors will be printed to the log, with `INFO` severity. If you are not
    seeing the logs, you might want to add the following line after your imports:

    ```python
      tf.logging.set_verbosity(tf.logging.INFO)
    ```

    Note that if `at_end` is True, `tensors` should not include any tensor
    whose evaluation produces a side effect such as consuming additional inputs.
    """

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            if elapsed_secs is not None:
                tf.logging.info("%s (%.3f sec)", self._formatter(
                    tensor_values), elapsed_secs)
            else:
                tf.logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append("%s = %s" % (tag, tensor_values[tag]))
            if elapsed_secs is not None:
                tf.logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
            else:
                tf.logging.info("%s", ", ".join(stats))
        np.set_printoptions(**original)


class SaveEvaluationResultHook(tf.train.SessionRunHook):
    """Saves evaluation results to disk for external use.
    Saves one file per batch in JSON format
    Remove padding for each sequence example and save:
    * protien sequence data
    * correct class
    * correct class prediction rank
    * correct class prediction probability
    * rank 1 prediction class
    * rank 1 prediction probability
    * rank N prediction class
    * rank N prediction probability

    logits shape=(batch_size, sequence_length, num_classes), dtype=float32

    """

    def __init__(self, model, output_file, post_evaluation_fn=None, predictions=None):
        """Initializes this hook.
        Args:
          model: The model for which to save the evaluation predictions.
          output_file: The output filename which will be suffixed by the current
            training step.
          post_evaluation_fn: (optional) A callable that takes as argument the
            current step and the file with the saved predictions.
          predictions: The predictions to save.
        """
        self._model = model
        self._output_file = output_file
        self._post_evaluation_fn = post_evaluation_fn
        self._predictions = predictions

    def begin(self):
        if self._predictions is None:
            self._predictions = misc.get_dict_from_collection("predictions")
        if not self._predictions:
            raise RuntimeError("The model did not define any predictions.")
        self._global_step = tf.train.get_global_step()
        if self._global_step is None:
            raise RuntimeError("Global step should be created to use SaveEvaluationPredictionHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs([self._predictions, self._global_step])

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        predictions, self._current_step = run_values.results
        self._output_path = "{}.{}".format(self._output_file, self._current_step)
        with io.open(self._output_path, encoding="utf-8", mode="a") as output_file:
            for prediction in misc.extract_batches(predictions):
                self._model.print_prediction(prediction, stream=output_file)

    def end(self, session):
        tf.logging.info("Evaluation predictions saved to %s", self._output_path)
        if self._post_evaluation_fn is not None:
            self._post_evaluation_fn(self._current_step, self._output_path)
