import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = self.cosine_decay_with_warmup(global_step=self.global_step,
                                           learning_rate_base=self.learning_rate_base,
                                           total_steps=self.total_steps,
                                           warmup_learning_rate=self.warmup_learning_rate,
                                           warmup_steps=self.warmup_steps,
                                           hold_base_rate_steps=self.hold_base_rate_steps)
        # K.set_value(self.model.optimizer.lr, lr)
        self.model.optimizer.learning_rate = lr

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0:
            print(f'\nEpoch %d:    lr rate - %.6f' % (epoch + 1, self.learning_rates[-1]), flush=True)

    @staticmethod
    def cosine_decay_with_warmup(global_step,
                                 learning_rate_base,
                                 total_steps,
                                 warmup_learning_rate=0.0,
                                 warmup_steps=0,
                                 hold_base_rate_steps=0):
        """Cosine decay schedule with warm up period.

        Cosine annealing learning rate as described in:
          Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
          ICLR 2017. https://arxiv.org/abs/1608.03983
        In this schedule, the learning rate grows linearly from warmup_learning_rate
        to learning_rate_base for warmup_steps, then transitions to a cosine decay
        schedule.

        Arguments:
            global_step {int} -- global step.
            learning_rate_base {float} -- base learning rate.
            total_steps {int} -- total number of training steps.

        Keyword Arguments:
            warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
            warmup_steps {int} -- number of warmup steps. (default: {0})
            hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                        before decaying. (default: {0})
        Returns:
          a float representing learning rate.

        Raises:
          ValueError: if warmup_learning_rate is larger than learning_rate_base,
            or if warmup_steps is larger than total_steps.
        """

        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                             'warmup_steps.')
        learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
            np.pi *
            (global_step - warmup_steps - hold_base_rate_steps
             ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)


if __name__ == '__main__':
    num_steps = 20 * 100
    lr_base = 0.1
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=lr_base,
                                            total_steps=num_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=2 * 100,
                                            hold_base_rate_steps=0,
                                            verbose=1)

    import matplotlib.pyplot as plt

    # Create a model.
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Generate dummy data.
    data = np.random.random((500, 100))
    labels = np.random.randint(10, size=(500, 1))

    # Convert labels to categorical one-hot encoding.
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, one_hot_labels, epochs=20, batch_size=5,
              verbose=0, callbacks=[warm_up_lr])

    plt.plot(warm_up_lr.learning_rates)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('lr', fontsize=20)
    plt.axis([0, num_steps, 0, lr_base * 1.1])
    plt.xticks(np.arange(0, num_steps, 50))
    plt.grid()
    plt.title('Cosine decay with warmup', fontsize=20)
    plt.tight_layout()
    plt.show()

    print('')
