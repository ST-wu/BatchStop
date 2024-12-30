import tensorflow as tf
import numpy as np

class AdaptiveBatchStopper(tf.keras.callbacks.Callback):
    def __init__(self, monitor='accuracy', init_window_size=20, adaptive_threshold_factor=0.5, decline_patience=3, min_batches=20, adapt_rate=0.1):
        super(AdaptiveBatchStopper, self).__init__()
        self.monitor = monitor
        self.init_window_size = init_window_size
        self.adaptive_threshold_factor = adaptive_threshold_factor
        self.decline_patience = decline_patience
        self.min_batches = min_batches
        self.adapt_rate = adapt_rate
        
        # initial
        self.history = []
        self.base_value = None
        self.decline_count = 0
        self.stopped_batch = None
        self.adaptive_threshold = None
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        self.batch_count += 1
        
        if self.batch_count < self.min_batches:
            return
        
        if self.base_value is None:
            self.base_value = current_value
            self.adaptive_threshold = abs(self.base_value * self.adaptive_threshold_factor)
        
        self.history.append(current_value)
        
        if len(self.history) >= self.init_window_size:
            recent_window = np.array(self.history[-self.init_window_size:])
            recent_std = np.std(recent_window)
            
            self.adaptive_threshold = max(self.adaptive_threshold * (1 - self.adapt_rate), recent_std * self.adaptive_threshold_factor)
            change_rate = np.mean(np.diff(recent_window))
            
            if abs(change_rate) < self.adaptive_threshold:
                if change_rate < 0:
                    self.decline_count += 1
                else:
                    self.decline_count = 0
                
                if self.decline_count >= self.decline_patience:
                    print(f"\nProposed early stopping at batch {self.batch_count} of epoch, change_rate={change_rate:.5f}, adaptive_threshold={self.adaptive_threshold:.5f}")
                    self.model.stop_training = True
                    self.stopped_batch = self.batch_count

    def on_epoch_end(self, epoch, logs=None):
        self.history = []
        self.base_value = None
        self.decline_count = 0
        self.stopped_batch = None
        self.batch_count = 0
        self.model.stop_training = False
